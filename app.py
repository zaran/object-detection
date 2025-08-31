import io
import base64
from collections import Counter
from typing import Dict

from flask import Flask, request, render_template, send_file, abort, jsonify
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load once at startup (yolov8n: nano, fast on CPU)
model = YOLO("yolov8n.pt")

def draw_boxes(img: Image.Image,
               boxes_xyxy: np.ndarray,
               cls_ids: np.ndarray,
               confs: np.ndarray,
               names: Dict[int, str]) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, img.width // 100))
    except Exception:
        font = ImageFont.load_default()

    line_w = max(2, img.width // 400)
    for (x1, y1, x2, y2), cls, conf in zip(boxes_xyxy, cls_ids, confs):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{names.get(int(cls), str(int(cls)))} {conf:.2f}"
        draw.rectangle([(x1, y1), (x2, y2)], outline=(20, 20, 20), width=line_w)
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        tx1, ty1 = x1, max(0, y1 - th - 4)
        draw.rectangle([(tx1, ty1), (tx1 + tw + 6, ty1 + th + 4)], fill=(20, 20, 20))
        draw.text((tx1 + 3, ty1 + 2), label, font=font, fill=(255, 255, 255))
    return img

def pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/detect")
def detect():
    if "image" not in request.files:
        abort(400, description="No file part named 'image'.")
    file = request.files["image"]
    if file.filename == "":
        abort(400, description="No selected file.")
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        abort(400, description="Could not read image. Upload a valid JPG/PNG.")

    # Optional confidence slider support from the form (defaults to 0.25)
    conf = request.form.get("conf", type=float) or 0.25

    # Run inference on CPU
    results = model.predict(image, conf=conf, device="cpu", verbose=False)
    r0 = results[0]

    # Extract detections
    if r0.boxes is None or len(r0.boxes) == 0:
        xyxy = np.zeros((0, 4), dtype=np.float32)
        cls_ids = np.zeros((0,), dtype=np.int32)
        confs = np.zeros((0,), dtype=np.float32)
        names = r0.names
    else:
        xyxy = r0.boxes.xyxy.cpu().numpy()
        cls_ids = r0.boxes.cls.cpu().numpy().astype(int)
        confs = r0.boxes.conf.cpu().numpy()
        names = r0.names

    annotated = draw_boxes(image, xyxy, cls_ids, confs, names)
    out_bytes = pil_to_jpeg_bytes(annotated)

    # Raw image for programmatic usage
    if request.args.get("return") == "image":
        return send_file(io.BytesIO(out_bytes), mimetype="image/jpeg")

    # Build a small human summary (e.g., person: 2, dog: 1)
    labels = [names[int(c)] for c in cls_ids]
    counts = Counter(labels)
    summary = [f"{k}: {v}" for k, v in counts.items()]

    b64_img = base64.b64encode(out_bytes).decode("utf-8")
    return render_template("result.html", b64_image=b64_img, summary=summary)

# Friendly errors for the browser
@app.errorhandler(400)
def bad_request(e):
    return render_template("result.html", b64_image=None, summary=None, error=str(e.description)), 400

if __name__ == "__main__":
    # Dev server (use Gunicorn in Docker)
    app.run(host="0.0.0.0", port=5000, debug=True)
