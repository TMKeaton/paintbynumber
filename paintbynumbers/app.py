import os
import platform
import textwrap
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from skimage.segmentation import slic
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from scipy.optimize import minimize

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)
OUTPUT_DIR = "exports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Config
# -----------------------------
DEFAULT_WIDTH = 1080
INITIAL_K = 24
FINAL_K = 20
DEFAULT_SLIC_N = 500
DEFAULT_SLIC_COMPACT = 30.0
DEFAULT_MIN_AREA = 200  # merge smaller polygons

BASE_COLORS = [
    (255, 234, 0),   # Cadmium lemon
    (255, 184, 28),  # Cadmium yellow medium
    (227, 38, 54),   # Cadmium red
    (255, 105, 180), # Light permanent rose
    (0, 47, 167),    # Ultramarine blue
    (0, 123, 167),   # Cerulean blue
    (101, 67, 33),   # Burnt umber
    (255, 255, 255)  # Titanium white
]
BASE_COLOR_NAMES = [
    "Cadmium Lemon", "Cadmium Yellow Medium", "Cadmium Red", "Light Permanent Rose",
    "Ultramarine Blue", "Cerulean Blue", "Burnt Umber", "Titanium White"
]

# -----------------------------
# Portable font loader
# -----------------------------
def find_system_font():
    system = platform.system()
    candidates = []
    if system == "Windows":
        candidates = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\arialbd.ttf"
        ]
    elif system == "Darwin":  # macOS
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf"
        ]
    else:  # Linux
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

FONT_PATH = find_system_font()
def load_font(size):
    if FONT_PATH:
        try:
            return ImageFont.truetype(FONT_PATH, int(size))
        except Exception:
            return ImageFont.load_default()
    return ImageFont.load_default()

# -----------------------------
# Color mixing
# -----------------------------
def rgb_to_cmy(rgb): return 1 - np.array(rgb) / 255.0
def cmy_to_rgb(cmy): return (1 - np.array(cmy)) * 255.0

def mix_colors(target_rgb, base_cmy):
    target_cmy = rgb_to_cmy(target_rgb)
    def objective(weights):
        mixed_cmy = np.dot(weights, base_cmy)
        return np.sum((mixed_cmy - target_cmy) ** 2)
    bounds = [(0, 1)] * len(base_cmy)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    initial = np.ones(len(base_cmy)) / len(base_cmy)
    result = minimize(objective, initial, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else initial

# -----------------------------
# Geometry + polygons
# -----------------------------
class ColoredPolygon:
    def __init__(self, polygon: Polygon, color_id: int):
        self.polygon = polygon
        self.color_id = color_id

def generate_polygons_from_mask(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    polys = []
    def to_poly(idx):
        outer = contours[idx].squeeze()
        if outer.ndim != 2 or outer.shape[0] < 3:
            return None
        holes = []
        child = hierarchy[idx][2]
        while child != -1:
            ring = contours[child].squeeze()
            if ring.ndim == 2 and ring.shape[0] >= 3:
                holes.append(ring.tolist())
            child = hierarchy[idx][0]
        poly = Polygon(outer, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if not poly.is_empty else None
    for i in range(len(contours)):
        if hierarchy[i][3] == -1:
            p = to_poly(i)
            if p is not None:
                if p.geom_type == "MultiPolygon":
                    polys.extend(list(p.geoms))
                else:
                    polys.append(p)
    return polys

def polygons_from_labels(labels, colors):
    polys = []
    for idx in range(len(colors)):
        m = np.zeros_like(labels, np.uint8)
        m[labels == idx] = 255
        pgs = generate_polygons_from_mask(m)
        polys.extend([ColoredPolygon(p, idx) for p in pgs])
    return polys

def merge_small(polygons, min_area):
    big = [p for p in polygons if p.polygon.area >= min_area]
    small = [p for p in polygons if p.polygon.area < min_area]
    merged = big[:]
    for s in small:
        target = min(merged, key=lambda g: s.polygon.centroid.distance(g.polygon.centroid), default=None)
        if target:
            new_poly = unary_union([target.polygon, s.polygon])
            if not new_poly.is_valid:
                new_poly = new_poly.buffer(0)
            if new_poly.geom_type == "MultiPolygon":
                new_poly = max(list(new_poly.geoms), key=lambda g: g.area)
            target.polygon = new_poly
    return merged

# -----------------------------
# Drawing
# -----------------------------
def outline_image(polygons, shape):
    h, w = shape
    canvas = np.full((h, w), 255, np.uint8)
    for pl in polygons:
        if pl.polygon.is_empty: continue
        geoms = pl.polygon.geoms if isinstance(pl.polygon, MultiPolygon) else [pl.polygon]
        for g in geoms:
            ext = np.int32(list(zip(*g.exterior.xy)))
            if len(ext) >= 3:
                cv2.polylines(canvas, [ext], True, 0, 1, lineType=cv2.LINE_AA)
            for ring in g.interiors:
                interior = np.int32(list(ring.coords))
                if interior.shape[0] >= 3:
                    cv2.polylines(canvas, [interior], True, 0, 1, lineType=cv2.LINE_AA)
    return Image.fromarray(canvas)

def draw_numbers(polygons, outline_image_pil):
    w, h = outline_image_pil.size
    min_dim = min(w, h)
    font_size = max(8, min(10, int(min_dim / 80)))
    font = load_font(font_size)
    d = ImageDraw.Draw(outline_image_pil)
    for pl in polygons:
        if pl.polygon.is_empty:
            continue
        c = pl.polygon.representative_point()
        text = str(pl.color_id + 1)
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text((c.x - tw/2, c.y - th/2), text, fill="black", font=font)
    return outline_image_pil

def colored_result(polygons, colors, shape):
    h, w = shape
    out = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    for pl in polygons:
        geom = pl.polygon
        geoms = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
        for g in geoms:
            mask = Image.new("L", (w, h), 0)
            d = ImageDraw.Draw(mask)
            d.polygon(list(g.exterior.coords), fill=255)
            for ring in g.interiors:
                d.polygon(list(ring.coords), fill=0)
            out.paste(Image.new("RGBA", (w, h), colors[pl.color_id] + (255,)), mask=mask)
    return out

def generate_color_palette(colors, weights):
    cols = 4
    swatch_w, swatch_h = 420, 240
    spacing_x, spacing_y = 80, 120
    rows = int(np.ceil(len(colors) / cols))
    W = cols * (swatch_w + spacing_x) + spacing_x
    H = rows * (swatch_h + spacing_y) + spacing_y + 40
    pal = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(pal)
    name_font = load_font(26)
    recipe_font = load_font(20)
    badge_font = load_font(16)
    x, y = spacing_x, spacing_y
    for i, (c, w) in enumerate(zip(colors, weights), 1):
        draw.rectangle((x, y, x + swatch_w, y + swatch_h), fill=c + (255,), outline="black", width=3)
        badge_r = 20
        badge_x = x + 12
        badge_y = y + 12
        draw.ellipse((badge_x - badge_r, badge_y - badge_r, badge_x + badge_r, badge_y + badge_r),
                     fill=(255,255,255,230), outline="black", width=2)
        badge_text = str(i)
        bbox = badge_font.getbbox(badge_text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((badge_x - tw/2, badge_y - th/2), badge_text, fill="black", font=badge_font)
        name = BASE_COLOR_NAMES[i-1] if i-1 < len(BASE_COLOR_NAMES) else f"Color {i}"
        name_bbox = name_font.getbbox(name)
        n_w = name_bbox[2] - name_bbox[0]
        draw.text((x + (swatch_w - n_w) / 2, y + swatch_h + 8), name, fill="black", font=name_font)
        recipe = ", ".join(f"{n}: {w[j]*100:.1f}%" for j, n in enumerate(BASE_COLOR_NAMES) if w[j] > 0.01)
        sample_bbox = recipe_font.getbbox("M")
        avg_char_w = max(6, sample_bbox[2] - sample_bbox[0])
        max_chars = max(20, int((swatch_w - 20) / avg_char_w))
        wrapped = textwrap.wrap(recipe, width=max_chars)
        for k, line in enumerate(wrapped[:6]):
            draw.text((x + 10, y + swatch_h + 40 + k * (recipe_font.size + 6)),
                      line, fill="black", font=recipe_font)
        if i % cols == 0:
            x = spacing_x
            y += swatch_h + spacing_y
        else:
            x += swatch_w + spacing_x
    return pal

# -----------------------------
# Core pipeline
# -----------------------------
def generate_painting(input_path, output_dir=OUTPUT_DIR):
    arr = np.array(Image.open(input_path).convert("RGB"))
    h, w = arr.shape[:2]
    img_flat = arr.reshape(-1, 3)
    kmeans = KMeans(n_clusters=INITIAL_K, random_state=0).fit(img_flat)
    labels = kmeans.labels_
    initial_centers = kmeans.cluster_centers_.astype(np.uint8)
    segments = slic(arr, n_segments=DEFAULT_SLIC_N, compactness=DEFAULT_SLIC_COMPACT, start_label=0)
    uniq = np.unique(segments)
    seg_means = np.array([arr[segments == sval].mean(axis=0) for sval in uniq])
    base_cmy = np.array([rgb_to_cmy(c) for c in BASE_COLORS])
    weights = np.array([mix_colors(seg, base_cmy) for seg in seg_means])
    mixed_colors = cmy_to_rgb(np.dot(weights, base_cmy)).astype(np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, lbls, centers = cv2.kmeans(mixed_colors.astype(np.float32), K=FINAL_K,
                                  bestLabels=None, criteria=criteria,
                                  attempts=10, flags=cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    colors = [tuple(map(int, c)) for c in centers]
    cluster_weights = []
    for i in range(len(centers)):
        cluster_segs = weights[lbls.flatten() == i]
        avg_weights = np.mean(cluster_segs, axis=0) if len(cluster_segs) > 0 else np.zeros(len(base_cmy))
        cluster_weights.append(avg_weights)
    seg_labels = np.zeros_like(segments, dtype=np.int32)
    for i, sval in enumerate(uniq):
        seg_labels[segments == sval] = lbls[i][0]
    polys = polygons_from_labels(seg_labels, colors)
    polys = merge_small(polys, DEFAULT_MIN_AREA)
    outline = outline_image(polys, (h, w))
    outline_nums = draw_numbers(polys, outline.copy())
    result = colored_result(polys, colors, (h, w))
    palette = generate_color_palette(colors, cluster_weights)
    outline_nums.save(os.path.join(output_dir, "outline.png"))
    result.save(os.path.join(output_dir, "result.png"))
    palette.save(os.path.join(output_dir, "swatches.png"))
    palette_rgb = palette.convert("RGB")
    outline_nums.save(os.path.join(output_dir, "outline.pdf"), "PDF")
    palette_rgb.save(os.path.join(output_dir, "swatches.pdf"), "PDF")
    return output_dir

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            input_path = os.path.join(OUTPUT_DIR, "input.jpg")
            file.save(input_path)
            generate_painting(input_path, OUTPUT_DIR)
            return redirect(url_for("results"))
    return '''
    <!DOCTYPE html>
    <html>
    <body>
      <h2>Upload an image to generate Paint by Numbers</h2>
      <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Generate</button>
      </form>
    </body>
    </html>
    '''

@app.route("/results")
def results():
    files = os.listdir(OUTPUT_DIR)
    files = [f for f in files if f.endswith((".png", ".pdf"))]
    links = "".join(f'<li><a href="/download/{f}">{f}</a></li>' for f in files)
    return f"<h2>Your Paint By Numbers Bundle</h2><ul>{links}</ul><a href='/'>Upload another</a>"

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
