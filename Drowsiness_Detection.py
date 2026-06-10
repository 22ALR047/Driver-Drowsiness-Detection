import cv2
import numpy as np
import time
import os
import sys
import mediapipe as mp
import threading
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks import python as mp_tasks
from scipy.spatial import distance

try:
    import winsound
    winsound_available = True
except ImportError:
    winsound_available = False

# ══════════════════════════════════════════════════════════════════════════════
# ALARM SOUND SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
alarm_playing = False

def play_beep_thread(frequency, duration):
    global alarm_playing
    alarm_playing = True
    if winsound_available:
        try:
            winsound.Beep(frequency, duration)
        except Exception:
            pass
    alarm_playing = False

def trigger_alarm_sound(frequency=800, duration=300):
    global alarm_playing
    if not alarm_playing:
        t = threading.Thread(target=play_beep_thread, args=(frequency, duration), daemon=True)
        t.start()


# ══════════════════════════════════════════════════════════════════════════════
# SERVER FLAG CHECK
# ══════════════════════════════════════════════════════════════════════════════
if len(sys.argv) > 1 and sys.argv[1] == "--server":
    try:
        from server import app
        print("[INFO] Launching Flask server backend...")
        app.run(host='0.0.0.0', port=5000, debug=True)
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Failed to run Flask server: {e}")
        sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# DEEP LEARNING MODEL SETUP
# ══════════════════════════════════════════════════════════════════════════════
torch_available = False
dl_model = None
img_transforms = None
device = None

try:
    import torch
    from torchvision import transforms
    from PIL import Image
    from models.deep_learning_models import MultimodalDrowsinessClassifier, LANDMARK_SET
    
    torch_available = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl_model = MultimodalDrowsinessClassifier().to(device)
    model_weight_path = "models/drowsiness_model.pth"
    if os.path.exists(model_weight_path):
        dl_model.load_state_dict(torch.load(model_weight_path, map_location=device))
        print(f"[INFO] Loaded local deep learning weights from '{model_weight_path}'")
    else:
        print("[WARNING] Local model weights not found. Inference running with initialized weights.")
    dl_model.eval()
    
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
except Exception as e:
    print(f"[WARNING] PyTorch or model components not available: {e}")
    print("[INFO] System will run in rule-based fallback mode.")


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════
EAR_THRESHOLD   = 0.25   # EAR below this → eyes closing
MAR_THRESHOLD   = 0.60   # MAR above this → mouth open / yawning
YAW_LOW         = 0.35   # Yaw ratio below → head turned left
YAW_HIGH        = 0.65   # Yaw ratio above → head turned right

EAR_CONSEC_FRAMES = 20   # frames of low EAR before drowsy alert
MAR_CONSEC_FRAMES = 15   # frames of high MAR before yawn alert
YAW_CONSEC_FRAMES = 20   # frames of head turn before distraction alert

MODEL_PATH = "models/face_landmarker.task"

# ══════════════════════════════════════════════════════════════════════════════
# LANDMARK INDICES  (MediaPipe 468-point Face Mesh)
# ══════════════════════════════════════════════════════════════════════════════
# Eye landmarks – 6-point EAR
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Mouth landmarks – 6-point MAR
#   P1=left-corner  P2=upper-left  P3=upper-right
#   P4=right-corner P5=lower-right P6=lower-left
MOUTH = [61, 82, 312, 291, 317, 87]

# Head-yaw landmarks
NOSE_TIP    = 1
LEFT_CHEEK  = 234
RIGHT_CHEEK = 454

# ══════════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def compute_ear(landmarks, indices, w, h):
    """Eye Aspect Ratio from 6 landmark points."""
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def compute_mar(landmarks, indices, w, h):
    """Mouth Aspect Ratio from 6 landmark points."""
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    A = distance.euclidean(pts[1], pts[5])   # upper-left  ↔ lower-left
    B = distance.euclidean(pts[2], pts[4])   # upper-right ↔ lower-right
    C = distance.euclidean(pts[0], pts[3])   # left-corner ↔ right-corner
    return (A + B) / (2.0 * C)

def compute_yaw(landmarks, w, h):
    """
    Normalised head-yaw ratio.
    0.5 = looking straight ahead
    < YAW_LOW  = head turned left
    > YAW_HIGH = head turned right
    """
    nose_x  = landmarks[NOSE_TIP].x    * w
    left_x  = landmarks[LEFT_CHEEK].x  * w
    right_x = landmarks[RIGHT_CHEEK].x * w
    span = right_x - left_x
    if span < 1e-5:
        return 0.5
    return (nose_x - left_x) / span

def draw_contour(frame, landmarks, indices, w, h, color):
    pts  = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices], np.int32)
    hull = cv2.convexHull(pts)
    cv2.drawContours(frame, [hull], -1, color, 1)

# ══════════════════════════════════════════════════════════════════════════════
# HUD OVERLAY
# ══════════════════════════════════════════════════════════════════════════════
def draw_hud(frame, ear, mar, yaw, ear_flag, mar_flag, yaw_flag, dl_pred=None, dl_prob=0.0):
    H, W = frame.shape[:2]

    # Semi-transparent dark panel on the left
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 145), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def metric_color(value, low=None, high=None, good_high=True):
        """Green if OK, red if threshold breached."""
        if good_high:  # higher is better (EAR)
            return (0, 220, 0) if value >= low else (0, 0, 220)
        else:          # lower is better (MAR)
            return (0, 220, 0) if value <= high else (0, 0, 220)

    ear_col = (0, 220, 0) if ear >= EAR_THRESHOLD else (0, 0, 220)
    mar_col = (0, 220, 0) if mar <= MAR_THRESHOLD else (0, 0, 220)
    yaw_col = (0, 220, 0) if YAW_LOW <= yaw <= YAW_HIGH else (0, 165, 0)

    cv2.putText(frame, f"EAR : {ear:.3f}", (10, 28),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, ear_col, 2)
    cv2.putText(frame, f"MAR : {mar:.3f}", (10, 58),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, mar_col, 2)
    cv2.putText(frame, f"YAW : {yaw:.3f}", (10, 88),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, yaw_col, 2)

    # Display Deep Learning predictions in HUD
    if dl_pred:
        dl_col = (0, 220, 0) if dl_pred == "ALERT" else (0, 0, 220)
        cv2.putText(frame, f"AI  : {dl_pred} ({dl_prob*100:.1f}%)", (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, dl_col, 2)
    else:
        cv2.putText(frame, f"AI  : Offline", (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 128, 128), 2)

    # Status bar at the bottom
    alerts = []
    if ear_flag >= EAR_CONSEC_FRAMES: alerts.append("DROWSY")
    if mar_flag >= MAR_CONSEC_FRAMES: alerts.append("YAWNING")
    if yaw_flag >= YAW_CONSEC_FRAMES: alerts.append("DISTRACTED")
    if dl_pred == "DROWSY": alerts.append("AI ALERT")

    if alerts:
        label = " | ".join(alerts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        x = (W - tw) // 2
        # Red banner
        cv2.rectangle(frame, (0, H - 45), (W, H), (0, 0, 180), -1)
        cv2.putText(frame, f"⚠  {label}  ⚠", (x - 25, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

# ══════════════════════════════════════════════════════════════════════════════
# FACE LANDMARKER SETUP
# ══════════════════════════════════════════════════════════════════════════════
base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionTaskRunningMode.IMAGE,
    num_faces=1
)
detector = FaceLandmarker.create_from_options(options)

# ══════════════════════════════════════════════════════════════════════════════
# WEBCAM SETUP  (try multiple backends)
# ══════════════════════════════════════════════════════════════════════════════
cap = None
for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]:
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        print(f"[INFO] Opened camera (backend={backend})")
        break
    cap.release()

if cap is None or not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit(1)

# Warm-up – wait up to 3 s for the first real frame
print("[INFO] Warming up camera...")
for _ in range(30):
    ret, _ = cap.read()
    if ret:
        print("[INFO] Camera ready.")
        break
    time.sleep(0.1)
else:
    print("[ERROR] Camera opened but no frames received.")
    cap.release()
    exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
ear_flag = mar_flag = yaw_flag = 0
visual_features = []
landmarks_seq = []
dl_pred = None
dl_prob = 0.0

print("[INFO] Running. Press 'q' to quit.\n")
print(f"  EAR threshold  : < {EAR_THRESHOLD}  -> drowsy")
print(f"  MAR threshold  : > {MAR_THRESHOLD}  -> yawning")
print(f"  Yaw range      : [{YAW_LOW} - {YAW_HIGH}]  -> distracted if outside\n")

while True:
    ret, frame = cap.read()
    if not ret:
        # Try recovering from a transient camera glitch (up to 10 retries)
        recovered = False
        for _ in range(10):
            time.sleep(0.05)
            ret, frame = cap.read()
            if ret:
                recovered = True
                break
        if not recovered:
            print("[ERROR] Lost camera feed. Exiting.")
            break

    frame = cv2.resize(frame, (720, 540))
    H, W  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_img)

    if result.face_landmarks:
        lm = result.face_landmarks[0]

        # ── Compute metrics ────────────────────────────────────────────────
        left_ear = compute_ear(lm, LEFT_EYE,  W, H)
        right_ear= compute_ear(lm, RIGHT_EYE, W, H)
        ear_val  = (left_ear + right_ear) / 2.0

        mar_val  = compute_mar(lm, MOUTH, W, H)
        yaw_val  = compute_yaw(lm, W, H)

        # ── Local Deep Learning Inference ──────────────────────────────────
        if torch_available:
            try:
                # 1. Bounding box and face crop
                xs = [pt.x for pt in lm]
                ys = [pt.y for pt in lm]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                pad_x = (max_x - min_x) * 0.15
                pad_y = (max_y - min_y) * 0.15
                
                x1 = max(0, int((min_x - pad_x) * W))
                y1 = max(0, int((min_y - pad_y) * H))
                x2 = min(W, int((max_x + pad_x) * W))
                y2 = min(H, int((max_y + pad_y) * H))
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(crop_rgb).resize((64, 64))
                    img_tensor = img_transforms(pil_img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        feat = dl_model.visual_extractor(img_tensor).squeeze(0).cpu()
                    visual_features.append(feat)
                else:
                    visual_features.append(torch.zeros(128))
            except Exception:
                visual_features.append(torch.zeros(128))
                
            try:
                lm_pts = torch.tensor([[lm[idx].x, lm[idx].y, lm[idx].z] for idx in LANDMARK_SET], dtype=torch.float32)
                landmarks_seq.append(lm_pts)
            except Exception:
                landmarks_seq.append(torch.zeros(len(LANDMARK_SET), 3))
                
            T = 30
            if len(visual_features) > T:
                visual_features.pop(0)
            if len(landmarks_seq) > T:
                landmarks_seq.pop(0)
                
            if len(visual_features) >= 5:
                num_buf = len(visual_features)
                pad_len = T - num_buf
                
                feats_tensor = torch.stack(visual_features)
                lms_tensor = torch.stack(landmarks_seq)
                
                if pad_len > 0:
                    feats_pad = torch.zeros(pad_len, 128)
                    lms_pad = torch.zeros(pad_len, len(LANDMARK_SET), 3)
                    feats_tensor = torch.cat([feats_pad, feats_tensor], dim=0)
                    lms_tensor = torch.cat([lms_pad, lms_tensor], dim=0)
                    
                feats_tensor = feats_tensor.unsqueeze(0).to(device)
                lms_tensor = lms_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    temporal_features = dl_model.temporal_transformer(feats_tensor)
                    landmark_features = dl_model.landmark_stgcn(lms_tensor)
                    fused = torch.cat([temporal_features, landmark_features], dim=1)
                    logits = dl_model.classifier(fused)
                    probs = torch.softmax(logits, dim=1).squeeze(0)
                    dl_prob = float(probs[1])
                    
                dl_pred = "DROWSY" if dl_prob >= 0.5 else "ALERT"
            else:
                dl_pred = "ALERT"
                dl_prob = 0.0
        else:
            dl_pred = None
            dl_prob = 0.0

        # ── Update flags ───────────────────────────────────────────────────
        ear_flag = ear_flag + 1 if ear_val < EAR_THRESHOLD else 0
        mar_flag = mar_flag + 1 if mar_val > MAR_THRESHOLD else 0
        yaw_flag = yaw_flag + 1 if not (YAW_LOW <= yaw_val <= YAW_HIGH) else 0

        # ── Draw contours ──────────────────────────────────────────────────
        eye_col  = (0, 220, 0) if ear_val >= EAR_THRESHOLD else (0, 0, 220)
        mouth_col= (0, 220, 0) if mar_val <= MAR_THRESHOLD else (0, 165, 255)

        draw_contour(frame, lm, LEFT_EYE,  W, H, eye_col)
        draw_contour(frame, lm, RIGHT_EYE, W, H, eye_col)
        draw_contour(frame, lm, MOUTH,     W, H, mouth_col)

        # ── HUD ────────────────────────────────────────────────────────────
        draw_hud(frame, ear_val, mar_val, yaw_val, ear_flag, mar_flag, yaw_flag, dl_pred, dl_prob)

        # ── Audio alerts ───────────────────────────────────────────────────
        if ear_flag >= EAR_CONSEC_FRAMES or dl_pred == "DROWSY":
            trigger_alarm_sound(1000, 300) # Urgent alert for drowsiness
        elif mar_flag >= MAR_CONSEC_FRAMES or yaw_flag >= YAW_CONSEC_FRAMES:
            trigger_alarm_sound(600, 200)  # Moderate warning for yawning/distraction

    else:
        # No face
        cv2.rectangle(frame, (0, 0), (220, 40), (20, 20, 20), -1)
        cv2.putText(frame, "No face detected", (8, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        ear_flag = mar_flag = yaw_flag = 0
        visual_features.clear()
        landmarks_seq.clear()
        dl_pred = None
        dl_prob = 0.0

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
print("[INFO] Stopped.")
