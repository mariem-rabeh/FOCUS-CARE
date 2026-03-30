"""
Posture Detection API - Flask + YOLOv8 Pose
---------------------------------------------
Endpoints:
  GET /           → serves index.html
  GET /video_feed → MJPEG stream with live annotations
  GET /status     → JSON with current posture + confidence
  GET /results    → JSON with full detection data (compatible with dashboard task #2)
"""

import cv2
import threading
import time
import math
from flask import Flask, Response, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

# ─── Config ────────────────────────────────────────────────────────────────────
# Pour utiliser le téléphone: mettez l'URL de IP Webcam. Exemple: "http://192.168.1.10:8080/video"
# Pour la webcam du PC: mettez 0
CAMERA_SOURCE  = "http://192.168.0.40:4747/video" 
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480

JPEG_QUALITY   = 80         # 0-100
DETECTION_FPS  = 5          # run YOLO every N frames to keep it fast
MODEL_NAME     = "yolov8n-pose.pt"   # nano pose model (~6MB, auto-downloaded)
USE_DSHOW      = True       # Windows: use DirectShow backend (avoids MSMF errors)

# Si l'image de votre téléphone apparaît couchée, modifiez cette variable :
# Options : cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, ou None
ROTATE_FRAME   = cv2.ROTATE_90_CLOCKWISE

# ─── App Setup ─────────────────────────────────────────────────────────────────
app  = Flask(__name__, static_folder="static")
CORS(app)   # allow the HTML frontend to call the API from any origin

# ─── Global State ──────────────────────────────────────────────────────────────
state = {
    "posture":     "unknown",   # "standing" | "sitting" | "unknown"
    "confidence":  0.0,
    "person_count": 0,
    "timestamp":   time.time(),
    "frame_count": 0,
}
state_lock   = threading.Lock()
output_frame = None
frame_lock   = threading.Lock()

# ─── YOLO Model ────────────────────────────────────────────────────────────────
print(f"[INFO] Loading YOLO model: {MODEL_NAME}")
model = YOLO(MODEL_NAME)   # downloads automatically on first run
print("[INFO] Model loaded.")


# ─── Posture Logic ─────────────────────────────────────────────────────────────
def classify_posture(keypoints):
    """
    Determine sitting vs standing from YOLOv8 Pose keypoints.

    YOLOv8 pose gives 17 keypoints (COCO):
      0  nose          5  left_shoulder   6  right_shoulder
      7  left_elbow    8  right_elbow     9  left_wrist    10 right_wrist
      11 left_hip     12 right_hip       13 left_knee     14 right_knee
      15 left_ankle   16 right_ankle

    Strategy:
      • If we can see hip + knee + ankle → compare vertical ratios
        - If knee_y ≈ hip_y (small vertical difference relative to body height)
          → sitting  (legs bent horizontally)
        - Else → standing (legs extended downward)
      • Fallback: compare shoulder-to-hip height vs hip-to-ankle height
    """
    kp = keypoints       # shape (17, 3)  [x, y, confidence]
    CONF_THRESH = 0.3

    def visible(idx):
        return kp[idx][2] > CONF_THRESH

    def y(idx):
        return kp[idx][1]

    def x(idx):
        return kp[idx][0]

    # Try to use hip, knee, ankle
    left_ok  = visible(11) and visible(13) and visible(15)
    right_ok = visible(12) and visible(14) and visible(16)

    if left_ok or right_ok:
        if left_ok and right_ok:
            hip_y   = (y(11) + y(12)) / 2
            knee_y  = (y(13) + y(14)) / 2
            ankle_y = (y(15) + y(16)) / 2
        elif left_ok:
            hip_y, knee_y, ankle_y = y(11), y(13), y(15)
        else:
            hip_y, knee_y, ankle_y = y(12), y(14), y(16)

        body_height = abs(ankle_y - hip_y) + 1e-6
        knee_drop   = abs(knee_y - hip_y) / body_height

        # If knee is barely below hip (< 30% of hip-ankle distance) → sitting
        if knee_drop < 0.30:
            return "sitting", round(float(1 - knee_drop), 2)
        else:
            return "standing", round(float(knee_drop), 2)

    # Fallback: use shoulder vs hip
    shoulder_ok = visible(5) and visible(6)
    hip_ok      = visible(11) and visible(12)
    if shoulder_ok and hip_ok:
        sh_y  = (y(5) + y(6)) / 2
        hip_y = (y(11) + y(12)) / 2
        # In image coords Y increases downward
        torso = abs(hip_y - sh_y)
        # If torso looks very compressed relative to frame height → sitting
        if torso < FRAME_HEIGHT * 0.18:
            return "sitting", 0.55
        else:
            return "standing", 0.60

    return "unknown", 0.0


# ─── Camera Thread ─────────────────────────────────────────────────────────────
def camera_loop():
    global output_frame

    # On Windows, DirectShow (CAP_DSHOW) is much more reliable than the
    # default MSMF backend — avoids the "can't grab frame" / -1072875772 error.
    if isinstance(CAMERA_SOURCE, int) and USE_DSHOW:
        backend = cv2.CAP_DSHOW
        cap = cv2.VideoCapture(CAMERA_SOURCE + backend)
    else:
        # Pour le flux réseau (téléphone) ou autre
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce latency

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {CAMERA_SOURCE}! Trying fallback…")
        cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("[ERROR] No camera stream found. Please check your IP Webcam URL or USB connection.")
        return

    frame_idx  = 0
    last_label = "unknown"
    last_conf  = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # Correction de l'orientation pour l'image du téléphone
        if ROTATE_FRAME is not None:
            frame = cv2.rotate(frame, ROTATE_FRAME)

        frame_idx += 1

        # ── Run YOLO every DETECTION_FPS-th frame ──────────────────────────
        if frame_idx % DETECTION_FPS == 0:
            results = model(frame, verbose=False, conf=0.4)
            result  = results[0]

            persons    = 0
            best_label = "unknown"
            best_conf  = 0.0

            if result.keypoints is not None:
                for kp_tensor in result.keypoints.data:
                    kp = kp_tensor.cpu().numpy()   # (17, 3)
                    persons += 1
                    label, conf = classify_posture(kp)
                    if conf > best_conf:
                        best_conf  = conf
                        best_label = label

            last_label = best_label
            last_conf  = best_conf

            with state_lock:
                state["posture"]      = last_label
                state["confidence"]   = best_conf
                state["person_count"] = persons
                state["timestamp"]    = time.time()
                state["frame_count"]  = frame_idx

            # Draw bounding boxes + keypoints
            annotated = result.plot()
        else:
            annotated = frame.copy()

        # ── Overlay label on every frame ───────────────────────────────────
        color_map = {
            "standing": (0, 200, 80),
            "sitting":  (0, 140, 255),
            "unknown":  (180, 180, 180),
        }
        color = color_map.get(last_label, (200, 200, 200))
        label_text = f"{last_label.upper()}  ({last_conf*100:.0f}%)"

        # Background rectangle for readability
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(annotated, (8, 8), (tw + 20, th + 22), (20, 20, 20), -1)
        cv2.putText(annotated, label_text, (14, th + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # ── Encode to JPEG and store ───────────────────────────────────────
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        with frame_lock:
            output_frame = jpeg.tobytes()

    cap.release()


# ─── MJPEG Generator ───────────────────────────────────────────────────────────
def generate():
    """Yield MJPEG frames for the /video_feed endpoint."""
    while True:
        with frame_lock:
            frame = output_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.04)   # ~25 FPS max for the stream


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the HTML dashboard."""
    return send_from_directory("static", "index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream — embed directly in <img src='/video_feed'>."""
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    """Quick posture status."""
    with state_lock:
        return jsonify({
            "posture":      state["posture"],
            "confidence":   state["confidence"],
            "person_count": state["person_count"],
            "timestamp":    state["timestamp"],
        })


@app.route("/results")
def results():
    """
    Full results JSON — compatible with the team's GET /results contract.
    concentration & stress are placeholder scores derived from posture.
    """
    with state_lock:
        posture    = state["posture"]
        confidence = state["confidence"]
        ts         = state["timestamp"]

    # Simple heuristics for now (will be replaced by real AI in task #3)
    concentration = round(confidence * 0.8, 2) if posture == "sitting"  else round(confidence * 0.4, 2)
    stress        = round(0.3 - confidence * 0.1, 2) if posture == "standing" else round(0.15, 2)

    return jsonify({
        "posture":        posture,
        "confidence":     confidence,
        "concentration":  concentration,
        "stress":         max(0, stress),
        "ANGER":          0.0,          # placeholder for task #3
        "timestamp":      ts,
    })


# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start camera + detection in background thread
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    print("[INFO] Camera thread started.")
    print("[INFO] Open http://localhost:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
