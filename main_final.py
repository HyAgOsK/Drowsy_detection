import cv2
import numpy as np
import mediapipe as mp
import dlib
from math import hypot
from playsound import playsound
import threading
from collections import defaultdict
from ultralytics import YOLO
import time

# ========== CONFIGURAÃ‡Ã•ES ==========
EYE_CLOSED_THRESHOLD = 4.5
POSTURE_DISTANCE_THRESHOLD = 2.9
ALERT_FRAME_THRESHOLD = 10
ALERT_SOUND_PATH = "alert.wav"
YOLO_MODEL_PATH = "/home/hyago/Desktop/Sleep_app/Drowsy_detection/runs_yolov5/detect/train/weights/best.pt"
DLIB_LANDMARK_PATH = "shape_predictor_68_face_landmarks.dat"

# ========== INICIALIZAÃ‡ÃƒO ==========
cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_LANDMARK_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)

# Estado interno
state = {
    "closed_eyes_frames": 0,
    "leaning_forward_frames": 0,
    "alert_triggered": False
}
class_counts = defaultdict(int)

# ========== FUNÃ‡Ã•ES UTILITÃRIAS ==========

def play_alert_sound():
    threading.Thread(target=playsound, args=(ALERT_SOUND_PATH,), daemon=True).start()

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def midpoint_float(p1, p2):
    return float((p1[0] + p2[0]) / 2)

def get_blinking_ratio(eye_points, landmarks):
    left = (landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y)
    right = (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y)
    top = midpoint(landmarks.part(eye_points[1]), landmarks.part(eye_points[2]))
    bottom = midpoint(landmarks.part(eye_points[5]), landmarks.part(eye_points[4]))

    hor_len = hypot(left[0] - right[0], left[1] - right[1])
    ver_len = hypot(top[0] - bottom[0], top[1] - bottom[1])
    return hor_len / ver_len

# ========== PROCESSAMENTO DE FRAME ==========

def process_frame(frame):
    global state
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pose_results = pose.process(rgb)
    total_alert_score = 0

    # YOLO
    yolo_results = yolo_model.track(frame, stream=True, conf=0.25, verbose=False, tracker="./botsort.yaml")
    detected_classes = set()

    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            confidence = float(box.conf[0])

            detected_classes.add(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ComitÃª de classes YOLO
    for cls in detected_classes:
        if cls in ["drowsy", "phone", "smoking"]:
            class_counts[cls] += 1
            total_alert_score += 1
            if class_counts[cls] == ALERT_FRAME_THRESHOLD:
                play_alert_sound()
            cv2.putText(frame, f"ALERTA: {cls.upper()}", (50, 50 + 30 * total_alert_score), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif cls == "seatbelt":
            class_counts["seatbelt_ok"] += 1

    # Dlib - DetecÃ§Ã£o de piscadas
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > EYE_CLOSED_THRESHOLD:
            state["closed_eyes_frames"] += 1
        else:
            state["closed_eyes_frames"] = 0

        if state["closed_eyes_frames"] >= ALERT_FRAME_THRESHOLD:
            total_alert_score += 1
            cv2.putText(frame, f"ALERTA: OLHOS FECHADOS ({state['closed_eyes_frames']})", 
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if state["closed_eyes_frames"] == ALERT_FRAME_THRESHOLD:
                play_alert_sound()

    # MediaPipe - Postura
    if pose_results.pose_landmarks:
        nose = pose_results.pose_landmarks.landmark[0]
        l_shoulder = pose_results.pose_landmarks.landmark[11]
        r_shoulder = pose_results.pose_landmarks.landmark[12]

        nose_point = np.array([nose.x, nose.y, nose.z])
        l_point = np.array([l_shoulder.x, l_shoulder.y, l_shoulder.z])
        r_point = np.array([r_shoulder.x, r_shoulder.y, r_shoulder.z])

        mid = midpoint_float(l_point, r_point)
        dist = np.linalg.norm(nose_point - mid)

        if dist > POSTURE_DISTANCE_THRESHOLD:
            state["leaning_forward_frames"] += 1
        else:
            state["leaning_forward_frames"] = 0

        if state["leaning_forward_frames"] >= ALERT_FRAME_THRESHOLD:
            total_alert_score += 1
            cv2.putText(frame, f"ALERTA: INCLINADO ({state['leaning_forward_frames']})", 
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if state["leaning_forward_frames"] == ALERT_FRAME_THRESHOLD:
                play_alert_sound()

    # ComitÃª de decisÃ£o geral
    if total_alert_score >= 2:
        if not state["alert_triggered"]:
            play_alert_sound()
            state["alert_triggered"] = True
    else:
        state["alert_triggered"] = False

    return frame

# ========== LOOP PRINCIPAL ==========

while cap.isOpened():
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame)
    cv2.imshow("Sistema de Alerta Inteligente", frame)
    print(f"FPS: {1/(time.time() - start):.2f}")

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
