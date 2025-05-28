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


# Verifica se a GPU está disponível e define o dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# captura de um video na pasta
cap = cv2.VideoCapture(0)
yolo_model = YOLO("../runs_yolov5/detect/train/weights/best.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

yolo_model.to(device)
print("Modelo na GPU:", next(yolo_model.model.parameters()).is_cuda)

# Variáveis de controle
alert_triggered = False
blinking_threshold = 4.5
eye_frame_threshold = 7
leaning_threshold_frames = 10
audio_path = "./alarm.wav"
class_counts = defaultdict(int)
frames_with_closed_eyes = 0
frames_leaning_forward = 0

# --- Funções auxiliares ---
def play_alert_sound():
    threading.Thread(target=playsound, args=(audio_path,), daemon=True).start()

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def midpoint2(p1, p2):
    return float((p1[0] + p2[0]) / 2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    return hor_line_length / ver_line_length

# --- Função principal de processamento ---
def process_frame(frame):
    global frames_with_closed_eyes, frames_leaning_forward, alert_triggered

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pose_results = pose.process(rgb)
    yolo_results = yolo_model.track(frame, stream=True, conf=0.25, verbose=False, device=device, tracker="./botsort.yaml")
    total_alert_score = 0

    # === YOLO detections ===
    classes_no_frame = set()
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = yolo_model.names[int(box.cls[0])]
            confidence = box.conf[0]

            classes_no_frame.add(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for cls in classes_no_frame:
        if cls in ["drowsy", "phone", "smoking"]:
            class_counts[cls] += 1
            total_alert_score += 1
            cv2.putText(frame, f"ALERTA: {cls.upper()}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if class_counts[cls] == 10:
                threading.Thread(target=play_alert_sound, daemon=True).start()
                pass
        elif cls == "seatbelt":
            class_counts["seatbelt_ok"] += 1

    # === Dlib blinking detection ===
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > blinking_threshold:
            frames_with_closed_eyes += 1
        else:
            frames_with_closed_eyes = 0

        if frames_with_closed_eyes >= eye_frame_threshold:
            total_alert_score += 1
            cv2.putText(frame, f"ALERTA: Olhos fechados ({frames_with_closed_eyes})",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if frames_with_closed_eyes == 10:
                threading.Thread(target=play_alert_sound, daemon=True).start()
                pass

    # === MediaPipe pose: postura ===
    if pose_results.pose_landmarks:
        nose = pose_results.pose_landmarks.landmark[0]
        l_shoulder = pose_results.pose_landmarks.landmark[11]
        r_shoulder = pose_results.pose_landmarks.landmark[12]

        nose_point = np.array([nose.x, nose.y, nose.z])
        l_point = np.array([l_shoulder.x, l_shoulder.y, l_shoulder.z])
        r_point = np.array([r_shoulder.x, r_shoulder.y, r_shoulder.z])
        mid = midpoint2(l_point, r_point)
        dist = np.linalg.norm(nose_point - mid)
        print("distancia dos ombros para o nariz", dist)
        if dist > 2.9:
            frames_leaning_forward += 1
        else:
            frames_leaning_forward = 0

        if frames_leaning_forward >= leaning_threshold_frames:
            total_alert_score += 1
            cv2.putText(frame, f"ALERTA: Inclinacao ({frames_leaning_forward})",
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if frames_leaning_forward == 10:
                threading.Thread(target=play_alert_sound, daemon=True).start()
                pass

    # === Comitê de decisão ===
    if total_alert_score >= 2:
        if not alert_triggered:
            play_alert_sound()
            alert_triggered = True
    else:
        alert_triggered = False

    return frame

# --- Loop principal ---
while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    
    if not ret:
        break
    #frame_resized = cv2.resize(frame, (640, 640))
    frame = process_frame(frame)
    cv2.imshow("Alerta - Comitê", frame)
    print(f"FPS: {1/(time.time() - start_time):.2f}")
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
