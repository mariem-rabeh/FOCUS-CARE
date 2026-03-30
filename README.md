# FOCUS-CARE
Le projet FOCUS-CARE AI consiste à concevoir un prototype intelligent de détection du niveau de concentration et de stress chez les étudiants, basé sur les technologies de l'Internet des Objets (IoT) et de l'Intelligence Artificielle sur le Cloud.

---

## Fonctionnalité : Posture AI — Quick-Start

Real-time **sitting / standing** detection via YOLOv8 Pose, served as a REST API with a live web dashboard.

---

## Stack

| Layer | Tech |
|---|---|
| AI inference | YOLOv8 Nano Pose (`yolov8n-pose.pt`) |
| Backend | Python · Flask · OpenCV |
| Stream | MJPEG over HTTP (`/video_feed`) |
| Frontend | Vanilla HTML/CSS/JS · Chart.js |

---

## Setup (one-time)

```bash
# 1. Create & activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt
```

> `yolov8n-pose.pt` (~6 MB) downloads automatically on first run.

---

## Run

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Dashboard HTML |
| `GET` | `/video_feed` | MJPEG live stream |
| `GET` | `/status` | Quick posture JSON |
| `GET` | `/results` | Full metrics JSON (team contract) |

### `GET /results` — example response
```json
{
  "posture":       "sitting",
  "confidence":    0.87,
  "concentration": 0.70,
  "stress":        0.15,
  "ANGER":         0.0,
  "timestamp":     1711820400.12
}
```

---

## Configuration

Edit the constants at the top of `app.py`:

```python
CAMERA_INDEX  = 0    # 0 = default webcam; change if you have multiple cameras
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
JPEG_QUALITY  = 80   # 0-100
DETECTION_FPS = 5    # run YOLO every N frames (lower = faster stream, fewer detections)
```

---

## Project Structure

```
posture-ai/
├── app.py            ← Flask backend + YOLO inference
├── requirements.txt
├── README.md
└── static/
    └── index.html    ← Dashboard (served by Flask)
```
