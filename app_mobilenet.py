import streamlit as st
import cv2
import numpy as np
import time
import psutil
import mediapipe as mp
import dlib
from math import hypot
from collections import deque
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
# --------------------
# Sidebar: ConfiguraÃ§Ãµes
# --------------------
st.sidebar.title("ðŸ”§ ConfiguraÃ§Ãµes")

model_resolution = st.sidebar.selectbox("ResoluÃ§Ã£o do modelo", [(320, 320), (640, 640), (1280, 720)])
confidence_threshold = st.sidebar.slider("ConfianÃ§a mÃ­nima", 0.0, 1.0, 0.35, 0.01)
iou_threshold = st.sidebar.slider("IoU threshold", 0.0, 1.0, 0.75, 0.01)
max_detections = st.sidebar.number_input("MÃ¡ximo detecÃ§Ãµes", min_value=1, max_value=100, value=3)
half_precision = st.sidebar.checkbox("Half precision (FP16)", value=False)
skip_frames = st.sidebar.number_input("Pular frames", min_value=0, max_value=30, value=0)

drowsy_fps_check = st.sidebar.number_input("FPS p/ checar drowsy", min_value=1, max_value=30, value=15)
drowsy_frame_threshold = st.sidebar.number_input("Frames p/ alerta drowsy", min_value=1, max_value=30, value=7)
eye_ratio_threshold = st.sidebar.slider("Limiar olhos", 1.0, 10.0, 4.5, 0.1)
posture_threshold = st.sidebar.slider("Limiar postura", 0.1, 3.0, 0.30, 0.1)

show_dashboard = st.sidebar.checkbox("Dashboard performance", value=True)
show_video = st.sidebar.checkbox("Exibir vÃ­deo", value=True)
start_button = st.sidebar.button("â–¶ï¸ Iniciar Sistema")
stop_button = st.sidebar.button("â¹ï¸ Parar Sistema")
reset_button = st.sidebar.button("ðŸ”„ Resetar MÃ©tricas")

st.title("ðŸ›°ï¸ Sistema Inteligente de Monitoramento")

if 'running' not in st.session_state:
    st.session_state.running = False
# --------------------
# InicializaÃ§Ã£o
# --------------------
fps_deque = deque(maxlen=60)
inf_time_deque = deque(maxlen=60)
detect_count_deque = deque(maxlen=60)
running = False
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
seatbelt_detected_this_frame = False
alert_placeholder = st.empty()
alert_placeholder_1 = st.empty()
frame_placeholder = st.empty()

# Estado - Adicionando controle persistente do cinto
if 'seatbelt_detected' not in st.session_state:
    st.session_state.seatbelt_detected = False

if reset_button:
    fps_deque.clear()
    inf_time_deque.clear()
    detect_count_deque.clear()
    st.session_state.seatbelt_detected = False
    
def load_labels(path):
    with open(path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
        return labels[1:] if labels [0] == '???' else labels
    

labels = load_labels('labelmap.txt')

interpreter = Interpreter(
    model_path='edgetpu.tflite',
    experimental_delegates=[load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = input_details[0]['dtype'] == np.float32 
input_mean, input_std = 127.5, 127.5

outname = output_details[0]['name']
boxes_idx, classes_idx, scores_idx = (1,3,0) if "StatefulPartitionedCall" in outname else (0, 1, 2)

 
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def blinking_ratio(pts, lm):
    left = (lm.part(pts[0]).x, lm.part(pts[0]).y)
    right = (lm.part(pts[3]).x, lm.part(pts[3]).y)
    top = midpoint(lm.part(pts[1]), lm.part(pts[2]))
    bottom = midpoint(lm.part(pts[5]), lm.part(pts[4]))
    hor = hypot(left[0] - right[0], left[1] - right[1])
    ver = hypot(top[0] - bottom[0], top[1] - bottom[1])
    return hor / ver if ver != 0 else 0



if start_button and not st.session_state.running:
    cap = cv2.VideoCapture("videofinal_720p.mp4")  # Ou 0 para webcam
    st.session_state.running = True
    running = True
    frames_closed = frames_lean = warning_score = drowsy_frames = 0
    seatbelt_frames = idx = 0


    if st.session_state.seatbelt_detected:
        alert_placeholder_1.success("? Cinto de seguranca detectado")

    dashboard = st.container()
    with dashboard:
        st.header("Dashboard de Performance")
        fps_metric = st.empty()
        inf_time_metric = st.empty()
        detect_count_metric = st.empty()
        memory_metric = st.empty()

    while running and cap is not None:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            st.warning("Video finalizado ou erro na captura.")
            break
        idx += 1
    
        if skip_frames and idx % (skip_frames + 1) != 0:
            continue

        w_res, h_res = model_resolution
        input_tensor = cv2.resize(frame, (w_res, h_res))
        input_tensor_rgb = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_tensor_rgb, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        elif input_details[0]['dtype'] == np.uint8:
            input_data = np.uint8(input_data)
        elif input_details[0]['dtype'] == np.float16:
            input_data = (np.float16(input_data) - input_mean) / input_std
        else:
            raise ValueError(f"Tipo de entrada nÃ£o suportado: {input_details[0]['dtype']}")

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interp_start = time.time()
        interpreter.invoke()
        interp_end = time.time()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        h, w, _ = frame.shape
        cnt = 0
        seatbelt_this_frame = False
    
        for i in range(len(scores)):
            if scores[i] < confidence_threshold or cnt >= max_detections:
                continue

            ymin, xmin = int(boxes[i][0] * h), int(boxes[i][1] * w)
            ymax, xmax = int(boxes[i][2] * h), int(boxes[i][3] * w)
            name = labels[int(classes[i])]

            if name not in ['phone', 'smoking', 'seatbelt', 'drowsy']:
                continue

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(frame, f"{name}: {scores[i]:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cnt += 1
            if name == 'seatbelt':
                seatbelt_detected_this_frame  = True
                # Marcar como detectado permanentemente
                if not st.session_state.seatbelt_detected:
                    st.session_state.seatbelt_detected = True
                    alert_placeholder_1.success("âœ… Cinto de seguranÃ§a detectado")
                alert_state = False
                seatbelt_frames = 0
                
            elif name == 'drowsy':
                drowsy_frames += 1
                if drowsy_frames >= drowsy_frame_threshold:
                    # Sempre mostrar alerta de sonolÃªncia
                    alert_placeholder.warning("âš ï¸ Alerta de sonolÃªncia")
                    alert_state = True
                    drowsy_frames = 0  
                    
            elif name == 'phone' or name == 'smoking':
                warning_score += 1
                if warning_score >= drowsy_frame_threshold:
                    # Sempre mostrar alerta
                    alert_placeholder.warning(f"âš ï¸ Alerta de {name}")
                    alert_state = True
                    warning_score = 0
                
        # Se o cinto nÃ£o foi detectado neste frame E nunca foi detectado antes
        if not seatbelt_detected_this_frame and not st.session_state.seatbelt_detected:
            seatbelt_frames += 1
            if seatbelt_frames >= drowsy_frame_threshold:
                alert_placeholder_1.warning("âš ï¸ Alerta: Cinto de seguranÃ§a nÃ£o detectado")
                alert_state = True
        else:
            # Resetar contador se cinto foi detectado ou jÃ¡ estÃ¡ marcado como detectado
            seatbelt_frames = 0
                
        if idx % drowsy_fps_check == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)
            for f in faces:
                lm = predictor(gray, f)
                br = blinking_ratio([36, 37, 38, 39, 40, 41], lm)
                print(f"RelaÃ§Ã£o de piscar: {br:.2f} limite: {eye_ratio_threshold:.2f}")
                if br > eye_ratio_threshold:
                    frames_closed += 1
                else:
                    frames_closed = 0
                if frames_closed >= drowsy_frame_threshold:
                    # Sempre mostrar alerta de olhos
                    alert_placeholder.warning("âš ï¸ ALERTA: Olhos fechados - SonolÃªncia detectada")
                    alert_state = True
                    cv2.putText(frame, "ALERTA: Drowsy eyes", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                n = results.pose_landmarks.landmark[0]
                ls = results.pose_landmarks.landmark[11]
                rs = results.pose_landmarks.landmark[12]
                mid = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
                dist = np.linalg.norm(np.array([n.x, n.y]) - np.array(mid))
                print(f"DistÃ¢ncia do nariz ao meio dos ombros: {dist:.2f} limite: {posture_threshold:.2f}")
                if dist < posture_threshold:
                    frames_lean += 1
                else:
                    frames_lean = 0
                if frames_lean >= drowsy_frame_threshold:
                    # Sempre mostrar alerta de postura
                    alert_placeholder.warning("âš ï¸ ALERTA: Postura inadequada")
                    alert_state = True
                    cv2.putText(frame, "ALERTA: Postura", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        fps = 1 / (time.time() - t0)
        fps_deque.append(fps)
        inf_time_deque.append((interp_end - interp_start) * 1000)
        detect_count_deque.append(cnt)

        if show_video:
            frame_placeholder.image(frame, channels='BGR', use_container_width=True)
        
        time.sleep(0.01)

        if show_dashboard:
            if fps_deque:
                fps_metric.metric("FPS (mÃ©dio)", f"{np.mean(fps_deque):.2f}")
                inf_time_metric.metric("InferÃªncia (ms)", f"{np.mean(inf_time_deque):.2f}")
                detect_count_metric.metric("DetecÃ§Ãµes/frame", f"{np.mean(detect_count_deque):.2f}")
                mem = psutil.Process().memory_info().rss / 1024 ** 2
                memory_metric.metric("MemÃ³ria (MB)", f"{mem:.2f}")
                time.sleep(0.01)
        if stop_button:
            st.session_state.running = False
            cap.release()

