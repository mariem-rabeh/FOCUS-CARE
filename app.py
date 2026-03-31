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
# Camo Studio agit comme une webcam virtuelle via USB ou Wi-Fi.
# Utilisez l'index 0, 1 ou 2 pour vous y connecter. (0 est généralement la webcam par défaut)
CAMERA_SOURCE  = 1
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480

JPEG_QUALITY   = 80         # 0-100
DETECTION_FPS  = 5          # run YOLO every N frames to keep it fast
MODEL_NAME     = "yolov8n-pose.pt"   # nano pose model (~6MB, auto-downloaded)
USE_DSHOW      = False      # Désactivé pour les Webcams Virtuelles (Camo Studio)

# Camo Studio s'occupe de l'orientation tout seul, donc on désactive la rotation forcée
ROTATE_FRAME   = None

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
    "stats": {"total": 0, "sitting": 0, "standing": 0, "stressed": 0, "focused": 0}
}
state_lock   = threading.Lock()
output_frame = None
frame_lock   = threading.Lock()

# ─── YOLO Model ────────────────────────────────────────────────────────────────
print(f"[INFO] Loading YOLO model: {MODEL_NAME}")
model = YOLO(MODEL_NAME)   # downloads automatically on first run
print("[INFO] Model loaded.")


# ─── Posture Logic ─────────────────────────────────────────────────────────────
def classify_posture(keypoints, box):
    kp = keypoints
    CONF_THRESH = 0.4

    def visible(idx): return kp[idx][2] > CONF_THRESH
    def y(idx):       return kp[idx][1]
    def x(idx):       return kp[idx][0]

    bx1, by1, bx2, by2 = box
    box_h = by2 - by1
    box_w = bx2 - bx1
    ratio = box_h / (box_w + 1e-6)

    shoulder_ok = visible(5) and visible(6)
    hip_ok      = visible(11) or visible(12)
    left_ok     = visible(11) and visible(13)
    right_ok    = visible(12) and visible(14)

    # Méthode 1 : Angle fémur hanche→genou
    if left_ok or right_ok:
        if left_ok and right_ok:
            hip_x  = (x(11) + x(12)) / 2
            hip_y  = (y(11) + y(12)) / 2
            knee_x = (x(13) + x(14)) / 2
            knee_y = (y(13) + y(14)) / 2
        elif left_ok:
            hip_x, hip_y   = x(11), y(11)
            knee_x, knee_y = x(13), y(13)
        else:
            hip_x, hip_y   = x(12), y(12)
            knee_x, knee_y = x(14), y(14)

        dx    = abs(knee_x - hip_x) + 1e-6
        dy    = knee_y - hip_y
        angle = math.degrees(math.atan2(dy, dx))

        hip_knee_dist = math.hypot(dx, abs(dy))
        box_h = by2 - by1
        femur_ratio = hip_knee_dist / (box_h + 1e-6)

        # Fauteuil relax : fémur court ET angle élevé → assis jambes étendues
        # Debout coupé   : fémur court MAIS angle élevé ET ankles invisibles → standing
        ankle_visible = visible(15) or visible(16)

        if angle < 55:
            # Fémur clairement horizontal → assis
            return "sitting", 0.88
        elif angle > 75:
            # Fémur clairement vertical → debout
            return "standing", 0.88
        else:
            # Zone ambiguë 55°–75° → utiliser femur_ratio + cheville
            if femur_ratio < 0.35 and not ankle_visible:
                # Jambes étendues + chevilles invisibles → fauteuil relax
                return "sitting", 0.75
            elif femur_ratio < 0.35 and ankle_visible:
                # Jambes courtes mais chevilles visibles → debout ou accroupi
                return "standing", 0.65
            elif angle < 65:
                return "sitting", 0.70
            else:
                return "standing", 0.70

    # Méthode 2 : Épaules + hanches visibles, genoux cachés
    if shoulder_ok and hip_ok:
        sh_y  = (y(5) + y(6)) / 2
        hip_y = (y(11) if visible(11) else y(12))
        torso = abs(hip_y - sh_y)
        if torso < FRAME_HEIGHT * 0.22:
            return "sitting", 0.70
        else:
            return "standing", 0.68

    # Méthode 3 : Épaules visibles, hanches cachées → bureau devant
    if shoulder_ok and not hip_ok:
        return "sitting", 0.75

    # Méthode 4 : Ratio bbox en dernier recours
    if ratio < 1.6:
        return "sitting", 0.55
    else:
        return "standing", 0.45

def analyze_behavior(kp, nose_history, box):
    """
    Analyse comportementale basée sur les heuristiques de position (Phase 3).
    Retourne: "Focused", "Stressed", ou "Neutral"
    """
    CONF_THRESH = 0.5
    def vis(idx): return kp[idx][2] > CONF_THRESH
    def ptx(idx): return kp[idx][0]
    def pty(idx): return kp[idx][1]

    x1, y1, x2, y2 = box
    box_w = max(1, x2 - x1)
    
    stress_points = 0
    focus_points  = 0

    # 1. Agitation vs Immobile (Nose History)
    if len(nose_history) >= 2:
        total_dist = 0
        for i in range(1, len(nose_history)):
            total_dist += math.hypot(nose_history[i][0] - nose_history[i-1][0], 
                                     nose_history[i][1] - nose_history[i-1][1])
        avg_dist = total_dist / (len(nose_history) - 1)
        
        # Mouvement par rapport à la taille de la personne
        if avg_dist > box_w * 0.08:   # Bouge beaucoup
            stress_points += 1
        elif avg_dist < box_w * 0.02: # Très stable
            focus_points += 1

    # 2. Mains près du visage
    if vis(0):
        # Distance Poignet (9,10) - Nez (0)
        dist_thresh = box_w * 0.35
        if vis(9) and math.hypot(ptx(9)-ptx(0), pty(9)-pty(0)) < dist_thresh:
            stress_points += 1
        if vis(10) and math.hypot(ptx(10)-ptx(0), pty(10)-pty(0)) < dist_thresh:
            stress_points += 1

    # 3. Visage tourné vs Droit
    if vis(0) and vis(3) and vis(4):
        # Les deux oreilles visibles : on regarde la symétrie
        d_left  = math.hypot(ptx(3)-ptx(0), pty(3)-pty(0))
        d_right = math.hypot(ptx(4)-ptx(0), pty(4)-pty(0))
        ratio = min(d_left, d_right) / (max(d_left, d_right) + 1e-6)
        if ratio < 0.4:
            stress_points += 1   # Tourné
        elif ratio > 0.7:
            focus_points += 1    # Droit de face
    elif vis(0) and (vis(3) ^ vis(4)): # XOR: une seule oreille visible
        stress_points += 1       # Très tourné (profil)

    # 4. Penché en avant (Concentration)
    if vis(0) and vis(5) and vis(6):
        shoulder_y = (pty(5) + pty(6)) / 2
        shoulder_w = abs(ptx(5) - ptx(6)) + 1e-6
        head_height = shoulder_y - pty(0)
        
        # Si la tête est basse par rapport aux épaules (pour écrire/regarder l'écran)
        if head_height < shoulder_w * 0.4:
            focus_points += 1

    # Décision finale
    if stress_points >= 2 or (stress_points > focus_points):
        return "Stressed"
    elif focus_points >= 2 or (focus_points > stress_points):
        return "Focused"
    else:
        return "Neutral"



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
    last_people = []
    last_stats = {"total": 0, "sitting": 0, "standing": 0}

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

            new_people = []
            best_label = "unknown"
            best_conf  = 0.0

            if result.boxes is not None and result.keypoints is not None:
                for box_idx, box_tensor in enumerate(result.boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box_tensor.cpu().numpy())
                    kp_tensor = result.keypoints.data[box_idx]
                    kp = kp_tensor.cpu().numpy()

                    label, conf = classify_posture(kp, (x1, y1, x2, y2))
                    if conf > best_conf:
                        best_conf  = conf
                        best_label = label

                    new_people.append({
                        "box": [x1, y1, x2, y2],
                        "label": label,
                        "kp": kp,
                        "matched": False
                    })

            # ── Tracking temporel (Lissage des boxes et labels) ──
            updated_people = []
            for old_p in last_people:
                best_match = None
                best_dist = 150  # Distance max en pixels pour associer la même personne
                
                ox_c = (old_p["box"][0] + old_p["box"][2]) / 2
                oy_c = (old_p["box"][1] + old_p["box"][3]) / 2

                for p in new_people:
                    if not p["matched"]:
                        nx_c = (p["box"][0] + p["box"][2]) / 2
                        ny_c = (p["box"][1] + p["box"][3]) / 2
                        dist = math.hypot(nx_c - ox_c, ny_c - oy_c)
                        if dist < best_dist:
                            best_dist = dist
                            best_match = p
                
                if best_match is not None:
                    best_match["matched"] = True
                    # 1. Lissage de la position du rectangle
                    alpha = 0.6 # 60% nouvelle position, 40% ancienne
                    n_box = best_match["box"]
                    o_box = old_p["box"]
                    s_box = [int(n_box[i]*alpha + o_box[i]*(1-alpha)) for i in range(4)]
                    
                    # 2. Lissage du Label de Posture
                    history = old_p.get("history", [old_p["label"]]) + [best_match["label"]]
                    if len(history) > 5:
                        history.pop(0)
                    smooth_label = max(set(history), key=history.count)
                    
                    # 3. Suivi du nez (Agitation) & Analyse comportementale
                    kp_new = best_match["kp"]
                    nose_hist = old_p.get("nose_history", [])
                    if kp_new[0][2] > 0.5: # Si nez visible, on l'ajoute à l'historique
                        nose_hist.append((kp_new[0][0], kp_new[0][1]))
                        if len(nose_hist) > 10: # Conserver 10 images d'historique 
                            nose_hist.pop(0)
                    
                    behav = analyze_behavior(kp_new, nose_hist, s_box)
                    behav_hist = old_p.get("behav_hist", [behav]) + [behav]
                    if len(behav_hist) > 5:
                        behav_hist.pop(0)
                    smooth_behav = max(set(behav_hist), key=behav_hist.count)
                    
                    updated_people.append({
                        "box": s_box,
                        "label": smooth_label,
                        "history": history,
                        "behavior": smooth_behav,
                        "behav_hist": behav_hist,
                        "nose_history": nose_hist,
                        "kp": kp_new,
                        "missed": 0
                    })
                else:
                    # Maintien temporaire si disparition brève
                    old_p["missed"] = old_p.get("missed", 0) + 1
                    if old_p["missed"] < 3: 
                        updated_people.append(old_p)
            
            # Ajouter les nouvelles personnes détectées
            for p in new_people:
                if not p["matched"]:
                    kp_new = p["kp"]
                    nose_hist = [(kp_new[0][0], kp_new[0][1])] if kp_new[0][2] > 0.5 else []
                    behav = analyze_behavior(kp_new, nose_hist, p["box"])
                    updated_people.append({
                        "box": p["box"],
                        "label": p["label"],
                        "history": [p["label"]],
                        "behavior": behav,
                        "behav_hist": [behav],
                        "nose_history": nose_hist,
                        "kp": kp_new,
                        "missed": 0
                    })
            
            last_people = updated_people
            
            sitting_count  = sum(1 for p in last_people if p["label"] == "sitting")
            standing_count = sum(1 for p in last_people if p["label"] == "standing")
            stress_count   = sum(1 for p in last_people if p.get("behavior") == "Stressed")
            focus_count    = sum(1 for p in last_people if p.get("behavior") == "Focused")

            last_stats = {
                "total": len(last_people),
                "sitting": sitting_count,
                "standing": standing_count,
                "stressed": stress_count,
                "focused": focus_count
            }

            with state_lock:
                state["posture"]      = best_label
                state["confidence"]   = best_conf
                state["person_count"] = len(last_people)
                state["timestamp"]    = time.time()
                state["frame_count"]  = frame_idx
                state["stats"]        = last_stats

        annotated = frame.copy()

        # ── Draw per-person bounding boxes ──────────────────────────────────
        for p in last_people:
            x1, y1, x2, y2 = p["box"]
            label = p["label"]
            behav = p.get("behavior", "Neutral")
            
            if behav == "Focused":
                box_color = (0, 200, 0) # Green
            elif behav == "Stressed":
                box_color = (0, 0, 255) # Red
            else:
                box_color = (255, 100, 50) # Orange-ish for neutral or unknown behavior
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw label background
            behavior_text = f" | {behav}" if behav != "Neutral" else ""
            text = f"Student {label.capitalize()}{behavior_text}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            
            bg_y = max(0, y1-25)
            cv2.rectangle(annotated, (x1, bg_y), (x1 + tw + 8, bg_y + th + 8), (255, 255, 255), -1)
            cv2.putText(annotated, text, (x1 + 4, bg_y + th + 4), cv2.FONT_HERSHEY_DUPLEX, 0.5, box_color, 1, cv2.LINE_AA)

        # ── Draw HUD / Dashboard overlay ───────────────────────────────────
        students = last_stats.get("total", 0)
        standing = last_stats.get("standing", 0)
        stressed = last_stats.get("stressed", 0)
        focused  = last_stats.get("focused", 0)
        
        activity   = int((standing / students * 100) if students > 0 else 0)
        stress_pct = int((stressed / students * 100) if students > 0 else 0)
        focus_pct  = int((focused / students * 100)  if students > 0 else 0)

        hud_w, hud_h = 320, 200
        hud_x, hud_y = FRAME_WIDTH - hud_w - 20, 20
        
        # Transparent overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

        # Draw HUD text
        def put_hud_text(img, msg, x, y, size=0.6, color=(255, 255, 255), thick=1):
            cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)

        put_hud_text(annotated, f"STUDENTS- {students}", hud_x + 15, hud_y + 35, 0.8)
        put_hud_text(annotated, f"ACTIVITY LEVEL {activity}%", hud_x + 15, hud_y + 70, 0.8)
        
        put_hud_text(annotated, "BEHAVIOUR ANALYSIS", hud_x + 15, hud_y + 115, 0.7, (0, 255, 255), 2)
        
        put_hud_text(annotated, f"FOCUS STATUS    {focus_pct}%", hud_x + 25, hud_y + 155, 0.65, (0, 255, 0), 2)
        put_hud_text(annotated, f"STRESS STATUS   {stress_pct}%", hud_x + 25, hud_y + 185, 0.65, (0, 0, 255), 2)

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
    Full results JSON — updated to return real AI behavioral logic (Phase 3).
    """
    with state_lock:
        posture    = state["posture"]
        confidence = state["confidence"]
        ts         = state["timestamp"]
        stats      = state.get("stats", {})

    total    = stats.get("total", 0)
    focused  = stats.get("focused", 0)
    stressed = stats.get("stressed", 0)

    concentration = round(focused / total, 2) if total > 0 else 0.0
    stress        = round(stressed / total, 2) if total > 0 else 0.0

    return jsonify({
        "posture":        posture,
        "confidence":     confidence,
        "concentration":  concentration,
        "stress":         stress,
        "ANGER":          stress,
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
