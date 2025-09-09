from flask import Flask, render_template, request, send_file
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from skimage.segmentation import slic
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import os, zipfile, io, platform
from textwrap import wrap

app = Flask(__name__)

# Ensure export folder exists
OUTPUT_DIR = "exports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Config
DEFAULT_WIDTH = 1024
INITIAL_K = 24
FINAL_K = 20
DEFAULT_SLIC_N = 500
DEFAULT_SLIC_COMPACT = 30.0
DEFAULT_MIN_AREA = 200  # merge smaller polygons

# Fonts
try:
    system = platform.system()
    font_path = None
    if system == "Windows":
        font_path = "C:\\Windows\\Fonts\\arial.ttf"
    elif system == "Darwin":
        font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
        if not os.path.exists(font_path):
            font_path = "/Library/Fonts/Arial.ttf"
    elif system == "Linux":
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        if not os.path.exists(font_path):
            font_path = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
        if not os.path.exists(font_path):
            font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"

    if font_path and os.path.exists(font_path):
        FONT_NUMBERS = ImageFont.truetype(font_path, 28)
        FONT_SWATCH = ImageFont.truetype(font_path, 36)
    else:
        FONT_NUMBERS = ImageFont.load_default()
        FONT_SWATCH = ImageFont.load_default()
except Exception:
    FONT_NUMBERS = ImageFont.load_default()
    FONT_SWATCH = ImageFont.load_default()

# Base palette
BASE_COLORS = [
    (255, 234, 0), (255, 184, 28), (227, 38, 54), (255, 105, 180),
    (0, 47, 167), (0, 123, 167), (101, 67, 33), (255, 255, 255)
]
BASE_COLOR_NAMES = [
    "Cadmium Lemon", "Cadmium Yellow Medium", "Cadmium Red", "Light Permanent Rose",
    "Ultramarine Blue", "Cerulean Blue", "Burnt Umber", "Titanium White"
]

# --- helpers (rgb, cmy, mixing, polygons, drawing) ---
def rgb_to_cmy(rgb): return 1 - np.array(rgb) / 255.0
def cmy_to_rgb(cmy): return (1 - np.array(cmy)) * 255.0
def mix_colors(target_rgb, base_cmy):
    target_cmy = rgb_to_cmy(target_rgb)
    def objective(weights): return np.sum((np.dot(weights, base_cmy) - target_cmy) ** 2)
    bounds = [(0, 1)] * len(base_cmy)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    initial = np.ones(len(base_cmy)) / len(base_cmy)
    result = minimize(objective, initial, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else initial

class ColoredPolygon:
    def __init__(self, polygon: Polygon, color_id: int):
        self.polygon = polygon
        self.color_id = color_id

def generate_polygons_from_mask(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None: return []
    hierarchy = hierarchy[0]
    polys = []
    def to_poly(idx):
        outer = contours[idx].squeeze()
        if outer.ndim != 2 or outer.shape[0] < 3: return None
        holes, child = [], hierarchy[idx][2]
        while child != -1:
            ring = contours[child].squeeze()
            if ring.ndim == 2 and ring.shape[0] >= 3:
                holes.append(ring.tolist())
            child = hierarchy[child][0]
        poly = Polygon(outer, holes)
        if not poly.is_valid: poly = poly.buffer(0)
        return poly if not poly.is_empty else None
    for i in range(len(contours)):
        if hierarchy[i][3] == -1:
            p = to_poly(i)
            if p is not None:
                polys.extend(list(p.geoms) if p.geom_type == "MultiPolygon" else [p])
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
    big, small, merged = [p for p in polygons if p.polygon.area >= min_area], \
                         [p for p in polygons if p.polygon.area < min_area], []
    merged = big[:]
    for s in small:
        target = min(merged, key=lambda g: s.polygon.centroid.distance(g.polygon.centroid), default=None)
        if target:
            new_poly = unary_union([target.polygon, s.polygon]).buffer(0)
            if new_poly.geom_type == "MultiPolygon":
                new_poly = max(list(new_poly.geoms), key=lambda g: g.area)
            target.polygon = new_poly
    return merged

def outline_image(polygons, shape):
    h, w = shape
    canvas = np.full((h, w), 255, np.uint8)
    for pl in polygons:
        if pl.polygon.is_empty: continue
        geoms = pl.polygon.geoms if isinstance(pl.polygon, MultiPolygon) else [pl.polygon]
        for g in geoms:
            ext = np.int32(list(zip(*g.exterior.xy)))
            cv2.polylines(canvas, [ext], True, 0, 1, lineType=cv2.LINE_AA)
            for ring in g.interiors:
                cv2.polylines(canvas, [np.int32(list(ring.coords))], True, 0, 1, lineType=cv2.LINE_AA)
    return Image.fromarray(canvas)

def draw_numbers(polygons, outline):
    d = ImageDraw.Draw(outline)
    for pl in polygons:
        if pl.polygon.is_empty: continue
        c = pl.polygon.representative_point()
        text = str(pl.color_id + 1)
        bbox = FONT_NUMBERS.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text((c.x - tw/2, c.y - th/2), text, fill="black", font=FONT_NUMBERS)
    return outline

def colored_result(polygons, colors, shape):
    h, w = shape
    out = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    for pl in polygons:
        mask = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(mask)
        d.polygon(list(pl.polygon.exterior.coords), fill=255)
        for ring in pl.polygon.interiors: d.polygon(list(ring.coords), fill=0)
        out.paste(Image.new("RGBA", (w, h), colors[pl.color_id] + (255,)), mask=mask)
    return out

def generate_color_palette(colors, weights):
    cols, swatch_w, swatch_h = 2, 320, 220
    spacing_x, spacing_y = 120, 180
    rows = int(np.ceil(len(colors) / cols))
    W, H = cols * (swatch_w + spacing_x) + spacing_x, rows * (swatch_h + spacing_y) + spacing_y
    pal, draw = Image.new("RGBA", (W, H), (255, 255, 255, 255)), ImageDraw.Draw(Image.new("RGBA", (W, H)))
    pal = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(pal)
    try:
        FONT_NUM = ImageFont.truetype("arial.ttf", 40)
        FONT_TEXT = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        FONT_NUM, FONT_TEXT = FONT_NUMBERS, FONT_SWATCH
    x, y = spacing_x, spacing_y
    for i, (c, w) in enumerate(zip(colors, weights), 1):
        draw.rectangle((x, y, x + swatch_w, y + swatch_h), fill=c + (255,), outline="black", width=4)
        num, bbox = str(i), FONT_NUM.getbbox(str(i))
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x + swatch_w/2 - tw/2, y + swatch_h/2 - th/2), num, fill="black", font=FONT_NUM)
        recipe = ", ".join(f"{n}: {w[j]*100:.1f}%" for j, n in enumerate(BASE_COLOR_NAMES) if w[j] > 0.01)
        for j, line in enumerate(wrap(recipe, width=40)):
            draw.text((x, y + swatch_h + 15 + j*32), line, fill="black", font=FONT_TEXT)
        if i % cols == 0: x, y = spacing_x, y + swatch_h + spacing_y
        else: x += swatch_w + spacing_x
    return pal

# --- core pipeline ---
def generate_painting(img):
    arr = np.array(img)
    h, w = arr.shape[:2]
    img_flat = arr.reshape(-1, 3)
    kmeans = KMeans(n_clusters=INITIAL_K, random_state=0).fit(img_flat)
    segments = slic(arr, n_segments=DEFAULT_SLIC_N, compactness=DEFAULT_SLIC_COMPACT, start_label=0)
    uniq = np.unique(segments)
    seg_means = np.array([arr[segments == sval].mean(axis=0) for sval in uniq])
    base_cmy = np.array([rgb_to_cmy(c) for c in BASE_COLORS])
    weights = np.array([mix_colors(seg, base_cmy) for seg in seg_means])
    mixed_colors = cmy_to_rgb(np.dot(weights, base_cmy)).astype(np.uint8)
    _, lbls, centers = cv2.kmeans(mixed_colors.astype(np.float32), K=FINAL_K, bestLabels=None,
                                  criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                  attempts=10, flags=cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    colors = [tuple(map(int, c)) for c in centers]
    cluster_weights = [np.mean(weights[lbls.flatten() == i], axis=0) if np.any(lbls.flatten() == i) else np.zeros(len(base_cmy)) for i in range(len(centers))]
    seg_labels = np.zeros_like(segments, dtype=np.int32)
    for i, sval in enumerate(uniq): seg_labels[segments == sval] = lbls[i][0]
    polys = merge_small(polygons_from_labels(seg_labels, colors), DEFAULT_MIN_AREA)
    outline, outline_nums = outline_image(polys, (h, w)), None
    outline_nums = draw_numbers(polys, outline.copy())
    result, palette = colored_result(polys, colors, (h, w)), generate_color_palette(colors, cluster_weights)
    outline_nums.save(os.path.join(OUTPUT_DIR, "outline.png"))
    result.save(os.path.join(OUTPUT_DIR, "result.png"))
    palette.save(os.path.join(OUTPUT_DIR, "swatches.png"))

# --- flask routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            img = Image.open(file.stream).convert("RGB")
            if DEFAULT_WIDTH:
                r = DEFAULT_WIDTH / float(img.width)
                img = img.resize((DEFAULT_WIDTH, int(img.height * r)), Image.Resampling.LANCZOS)
            generate_painting(img)
            memory_file = io.BytesIO()
            with zipfile.ZipFile(memory_file, "w") as zf:
                for f in ["outline.png", "result.png", "swatches.png"]:
                    zf.write(os.path.join(OUTPUT_DIR, f), f)
            memory_file.seek(0)
            return send_file(memory_file, as_attachment=True, download_name="bundle.zip")
    return '''
    <h1>Upload an image</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*">
      <input type="submit" value="Process">
    </form>
    '''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
