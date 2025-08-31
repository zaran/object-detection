# Flask YOLOv8 Object Detector (CPU)

Lightweight Flask app that accepts an uploaded image, runs a **YOLOv8n** model on CPU, and returns the same image with bounding boxes drawn. HTML is split into templates, and a Dockerfile is provided for easy deployment.

---

## Project Structure

```
object-detector/
├─ app.py
├─ requirements.txt
├─ Dockerfile
├─ templates/
│  ├─ index.html
│  └─ result.html
└─ static/
   └─ style.css
```

- **app.py** – minimal Flask server with `/`, `/detect`, and `/health`.
- **templates/** – split HTML (upload page and results page).
- **static/style.css** – tiny stylesheet for basic styling.
- **Dockerfile** – CPU-only build with Gunicorn and pre-downloaded model weights.
- **requirements.txt** – Python dependencies.

> Model: `yolov8n.pt` (nano) – very small and fast on CPU.


---

## Prerequisites

- **Python 3.10+** (3.11 recommended) for local dev
- **pip** (latest)
- **Docker** (optional, for containerized run)
- Network access at build/first-run time to download model weights

> On first run outside Docker, Ultralytics will download `yolov8n.pt` to your cache. The Dockerfile pre-downloads the weights during build to avoid startup delays.


---

## Quick Start (Local Development)

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install deps
pip install --upgrade pip
pip install -r requirements.txt

# 3) Run the dev server (Flask’s built-in)
python app.py
# Open http://localhost:5000
```

### Test it
1. Open `http://localhost:5000` in your browser.
2. Upload a JPG/PNG and click **Detect Objects**.
3. You’ll see an annotated image and a small count summary.
4. Programmatic download of the annotated image:
   ```bash
   curl -o out.jpg -F "image=@path/to/your.jpg" "http://localhost:5000/detect?return=image"
   ```


---

## Run in Docker (Production-style)

Build and run the CPU image that starts **Gunicorn** on port 8000.

```bash
# Build
docker build -t objdet:cpu .

# Run
docker run --rm -p 8000:8000 objdet:cpu
# Open http://localhost:8000
```

### Health check
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Annotated image via API
```bash
curl -o out.jpg -F "image=@path/to/your.jpg" "http://localhost:8000/detect?return=image"
```

> The Dockerfile pre-installs **CPU-only PyTorch** and downloads the `yolov8n.pt` weights in a build layer to keep startup fast.


---

## Endpoints

- `GET /` – Upload page (HTML form with optional confidence slider).
- `POST /detect` – Accepts `multipart/form-data` with field **image** and optional **conf** (0.10–0.90).
  - Default response: HTML results page.
  - Add query `?return=image` to receive **image/jpeg** bytes (annotated image).
- `GET /health` – JSON health probe: `{"status":"ok"}`.

**Example form fields**
- `image`: your file (JPG/PNG)
- `conf`: (optional) confidence threshold, e.g. `0.25`


---

## Configuration Notes

- The confidence threshold is adjustable via the UI slider or by posting `conf`.
- The server draws boxes using Pillow (no OpenCV dependency in drawing code). Ultralytics internally imports `cv2`, so the container uses **opencv-python-headless** to avoid GUI dependencies.


---

## Troubleshooting

### `ImportError: libGL.so.1: cannot open shared object file`
If you ever see this while importing `cv2` (OpenCV) in a container:
- Prefer **opencv-python-headless** (already listed in `requirements.txt`), **or**
- Install GUI libs in your image: `apt-get install -y libgl1 libglib2.0-0`

### Slow first request
If you didn’t pre-download weights, Ultralytics will fetch them on first run. The Dockerfile includes a step to prefetch `yolov8n.pt` during `docker build`.

### 413 “Request Entity Too Large” behind a reverse proxy
Increase the allowed body size (e.g. in Nginx: `client_max_body_size 25M;`).

### Gunicorn worker settings
The Dockerfile starts Gunicorn with `-w 2` (2 workers). For larger instances, you can change it at runtime:
```bash
docker run --rm -p 8000:8000 objdet:cpu \
  gunicorn -w 4 -b 0.0.0.0:8000 app:app
```
