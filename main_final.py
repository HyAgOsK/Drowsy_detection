import cv2
import numpy as np
import mediapipe as mp
import dlib
from math import hypot
from playsound import playsound
import threading
from collections import defaultdict
from ultralytics import YOLO
import torch
import time

# Inicializações
cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
yolo_model = YOLO("best.pt")  # substitua com o seu modelo YOLO

# Parâmetros de limiar
BLINK_RATIO_THRESHOLD = 4.5
POSTURE_DIST_THRESHOLD = 2.9
FRAME_THRESHOLD = 10  # X frames consecutivos

# Sons
audio_path = "alert.wav"

# Estado
state = {
    "frames_with_closed_eyes": 0,
    "frames_leaning_forward": 0,
    "alert_triggered": False
}

class_counts = defaultdict(int)

# Funções auxiliares
def play_alert_sound():
    threading.Thread(target=playsound, args=(audio_path,), daemon=True).start()

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def midpoint2(p1, p2):
    return float((p1[0] + p2[0]) / 2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    hor = hypot(left[0] - right[0], left[1] - right[1])
    ver = hypot(top[0] - bottom[0], top[1] - bottom[1])
    return hor / ver

# Processamento do frame
def process_frame(frame):
    global state
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pose_results = pose.process(rgb)
    yolo_results = yolo_model.track(frame, stream=True, conf=0.25, verbose=False, tracker="./botsort.yaml")

    total_alert_score = 0
    classes_detected = set()

    # === YOLO ===
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            class_name = yolo_model.names[cls_id]
            confidence = box.conf[0]

            classes_detected.add(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Comitê YOLO
    for cls in ["drowsy", "phone", "smoking"]:
        if cls in classes_detected:
            class_counts[cls] += 1
            if class_counts[cls] >= FRAME_THRESHOLD:
                total_alert_score += 1
                cv2.putText(frame, f"YOLO ALERTA: {cls.upper()}", (50, 50 + 30 * list(classes_detected).index(cls)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                play_alert_sound()

    if "seatbelt" not in classes_detected:
        class_counts["seatbelt"] += 1
        if class_counts["seatbelt"] >= FRAME_THRESHOLD:
            cv2.putText(frame, f"ALERTA: Cinto de segurança não detectado",
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            play_alert_sound()
    else:
        class_counts["seatbelt"] = 0

    # === Detecção de olhos fechados (BR) ===
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blink_ratio = (left_eye + right_eye) / 2
        print("BR:", blink_ratio)

        if blink_ratio > BLINK_RATIO_THRESHOLD:
            state["frames_with_closed_eyes"] += 1
        else:
            state["frames_with_closed_eyes"] = 0

        if state["frames_with_closed_eyes"] >= FRAME_THRESHOLD:
            total_alert_score += 1
            cv2.putText(frame, "ALERTA: Olhos fechados", (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # === Detecção de postura (DSC) ===
    if pose_results.pose_landmarks:
        nose = pose_results.pose_landmarks.landmark[0]
        l_shoulder = pose_results.pose_landmarks.landmark[11]
        r_shoulder = pose_results.pose_landmarks.landmark[12]

        nose_pt = np.array([nose.x, nose.y, nose.z])
        mid_pt = np.array([
            (l_shoulder.x + r_shoulder.x) / 2,
            (l_shoulder.y + r_shoulder.y) / 2,
            (l_shoulder.z + r_shoulder.z) / 2,
        ])

        dist = np.linalg.norm(nose_pt - mid_pt)
        print("DSC:", dist)

        if dist > POSTURE_DIST_THRESHOLD:
            state["frames_leaning_forward"] += 1
        else:
            state["frames_leaning_forward"] = 0

        if state["frames_leaning_forward"] >= FRAME_THRESHOLD:
            total_alert_score += 1
            cv2.putText(frame, "ALERTA: Postura inclinada", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Comitê de decisão final
    if total_alert_score >= 2 and not state["alert_triggered"]:
        play_alert_sound()
        state["alert_triggered"] = True
    elif total_alert_score < 2:
        state["alert_triggered"] = False

    return frame

# === Loop principal ===
while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame)
    cv2.imshow("Monitoramento de Fadiga", frame)
    print(f"FPS: {1 / (time.time() - start_time):.2f}")

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
