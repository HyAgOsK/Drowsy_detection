import streamlit as st
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
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging
import psutil
import gc

@dataclass
class PerformanceConfig:
    """ConfiguraÃ§Ãµes de performance e otimizaÃ§Ã£o"""
    AVAILABLE_RESOLUTIONS: Dict[str, Tuple[int, int]] = None
    
    YOLO_CONFIDENCE: float = 0.25
    YOLO_IOU_THRESHOLD: float = 0.45
    YOLO_MAX_DETECTIONS: int = 5
    YOLO_HALF_PRECISION: bool = False
    
    FRAME_SKIP: int = 1  
    USE_MULTITHREADING: bool = True
    MEMORY_OPTIMIZATION: bool = True
    
    SHOW_CONFIDENCE: bool = True
    SHOW_CLASS_NAMES: bool = True
    BBOX_THICKNESS: int = 2
    
    def __post_init__(self):
        if self.AVAILABLE_RESOLUTIONS is None:
            self.AVAILABLE_RESOLUTIONS = {
                "240p (426x240)": (426, 240),
                "360p (640x360)": (640, 360),
                "480p (854x480)": (854, 480),
                "720p (1280x720)": (1280, 720),
                "1080p (1920x1080)": (1920, 1080),
                "Custom YOLO 320": (320, 320),
                "Custom YOLO 416": (416, 416),
                "Custom YOLO 512": (512, 512),
                "Custom YOLO 640": (640, 640),
                "Custom YOLO 832": (832, 832),
                "Custom YOLO 1024": (1024, 1024),
                "Custom YOLO 240": (240,240),
                "Custom YOLO 192": (192,192),
                "Custom YOLO 544": (544,544),
                "custom yolo 480": (480,480)
           
            }

@dataclass
class Config:
    EYE_CLOSED_THRESHOLD: float = 4.5
    POSTURE_DISTANCE_THRESHOLD: float = 0.90
    ALERT_FRAME_THRESHOLD: int = 15
    SEATBELT_DETECTION_FRAMES: int = 1
    ALERT_SOUND_PATH: str = "alarm.wav"
    YOLO_MODEL_PATH: str = "/home/hyago/Desktop/Sleep_app/Drowsy_detection/runs_yolov5/detect/train/weights/best_saved_model/best_full_integer_quant_edgetpu.tflite"
    DLIB_LANDMARK_PATH: str = "shape_predictor_68_face_landmarks.dat"
    MODEL_RESOLUTION: Tuple[int, int] = (640, 640)
    MONITORED_CLASSES: List[str] = None
    
    performance: PerformanceConfig = None

    def __post_init__(self):
        if self.MONITORED_CLASSES is None:
            self.MONITORED_CLASSES = ["drowsy", "phone", "smoking"]
        if self.performance is None:
            self.performance = PerformanceConfig()

@dataclass
class PerformanceMetrics:
    """MÃ©tricas de performance do sistema"""
    fps_current: float = 0.0
    fps_average: float = 0.0
    processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    frame_count: int = 0
    detection_count: int = 0
    resolution: str = ""
    
    def update_memory_cpu(self):
        """Atualiza mÃ©tricas de memÃ³ria e CPU"""
        process = psutil.Process()
        self.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        self.cpu_usage_percent = psutil.cpu_percent(interval=None)

@dataclass
class AlertState:
    closed_eyes_frames: int = 0
    leaning_forward_frames: int = 0
    seatbelt_on: bool = False
    seatbelt_frames: int = 0
    seatbelt_missing_frames: int = 0

class AlertSystem:
    def __init__(self, config: Config):
        self.config = config
        self.state = AlertState()
        self.class_counts = defaultdict(int)
        self.alert_triggered_by_class = defaultdict(bool)
        self.metrics = PerformanceMetrics()
        self._initialize_models()

    def _initialize_models(self):
        """Inicializa todos os modelos necessÃ¡rios"""
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.config.DLIB_LANDMARK_PATH)

            self.yolo_model = YOLO(self.config.YOLO_MODEL_PATH)
            
            if self.config.performance.YOLO_HALF_PRECISION:
                try:
                    self.yolo_model.half()
                except:
                    logging.warning("Half precision nÃ£o suportado, continuando com float32")

        except Exception as e:
            logging.error(f"Erro ao inicializar modelos: {e}")
            raise

    @staticmethod
    def play_alert_sound(sound_path: str):
        """Toca som de alerta em thread separada"""
        def _play():
            try:
                playsound(sound_path)
            except Exception as e:
                logging.warning(f"Erro ao tocar som: {e}")

        if threading.active_count() < 10: 
            threading.Thread(target=_play, daemon=True).start()

    @staticmethod
    def midpoint(p1, p2) -> Tuple[int, int]:
        """Calcula ponto mÃ©dio entre dois pontos"""
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

    @staticmethod
    def midpoint_float(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Calcula ponto mÃ©dio para arrays numpy"""
        return (p1 + p2) / 2

    def get_blinking_ratio(self, eye_points: List[int], landmarks) -> float:
        """Calcula razÃ£o de piscada para um olho"""
        left = (landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y)
        right = (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y)
        top = self.midpoint(landmarks.part(eye_points[1]), landmarks.part(eye_points[2]))
        bottom = self.midpoint(landmarks.part(eye_points[5]), landmarks.part(eye_points[4]))

        hor_len = hypot(left[0] - right[0], left[1] - right[1])
        ver_len = hypot(top[0] - bottom[0], top[1] - bottom[1])

        return hor_len / (ver_len + 1e-6)

    def process_yolo_detection(self, frame: np.ndarray) -> Tuple[set, List[Tuple[str, int]]]:
        """Processa detecÃ§Ã£o YOLO com configuraÃ§Ãµes avanÃ§adas"""
        alert_messages = []
        classes_seen_this_frame = set()
        detection_count = 0

        try:
            results = self.yolo_model.predict(
                frame,
                stream=True,
                conf=self.config.performance.YOLO_CONFIDENCE,
                iou=self.config.performance.YOLO_IOU_THRESHOLD,
                max_det=self.config.performance.YOLO_MAX_DETECTIONS,
                verbose=False,
                imgsz=list(self.config.MODEL_RESOLUTION),
                half=self.config.performance.YOLO_HALF_PRECISION
            )

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_id = int(box.cls[0])
                        class_name = self.yolo_model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        classes_seen_this_frame.add(class_name)
                        detection_count += 1

                        color_map = {
                            "drowsy": (0, 0, 255),      
                            "phone": (255, 165, 0),     
                            "smoking": (255, 0, 255),  
                            "seatbelt": (0, 255, 0)     
                        }
                        color = color_map.get(class_name, (255, 255, 255))

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 
                                    self.config.performance.BBOX_THICKNESS)
                        
                        label_parts = []
                        if self.config.performance.SHOW_CLASS_NAMES:
                            label_parts.append(class_name)
                        if self.config.performance.SHOW_CONFIDENCE:
                            label_parts.append(f"{confidence:.2f}")
                        
                        label = " ".join(label_parts)
                        if label:
                            (text_width, text_height), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                                        (x1 + text_width, y1), color, -1)
                            cv2.putText(frame, label, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            self.metrics.detection_count = detection_count

            for cls in self.config.MONITORED_CLASSES:
                if cls in classes_seen_this_frame:
                    self.class_counts[cls] += 1
                    if self.class_counts[cls] >= self.config.ALERT_FRAME_THRESHOLD:
                        if not self.alert_triggered_by_class[cls]:
                            self.play_alert_sound(self.config.ALERT_SOUND_PATH)
                            self.alert_triggered_by_class[cls] = True
                        alert_messages.append((f"ðŸš¨ {cls.upper()} - ALERTA!!!", self.class_counts[cls]))
                    else:
                        alert_messages.append((f"{cls.upper()} (anÃ¡lise)", self.class_counts[cls]))
                else:
                    self.class_counts[cls] = 0
                    self.alert_triggered_by_class[cls] = False

        except Exception as e:
            logging.warning(f"Erro na detecÃ§Ã£o YOLO: {e}")

        return classes_seen_this_frame, alert_messages

    def process_seatbelt_detection(self, classes_seen: set) -> List[Tuple[str, int]]:
        """Processa detecÃ§Ã£o de cinto de seguranÃ§a"""
        alert_messages = []

        if not self.state.seatbelt_on:
            if "seatbelt" in classes_seen:
                self.state.seatbelt_frames += 1
                self.state.seatbelt_missing_frames = 0

                if self.state.seatbelt_frames >= self.config.SEATBELT_DETECTION_FRAMES:
                    self.state.seatbelt_on = True
                    alert_messages.append(("âœ… CINTO DETECTADO", self.state.seatbelt_frames))
                else:
                    alert_messages.append(("ðŸ” Analisando cinto...", self.state.seatbelt_frames))
            else:
                self.state.seatbelt_missing_frames += 1
                self.state.seatbelt_frames = 0

                if self.state.seatbelt_missing_frames > self.config.ALERT_FRAME_THRESHOLD:
                    alert_messages.append(("âš ï¸ CINTO NÃƒO DETECTADO!", self.state.seatbelt_missing_frames))
                else:
                    alert_messages.append(("ðŸ” Procurando cinto...", self.state.seatbelt_missing_frames))
        else:
            alert_messages.append(("âœ… CINTO OK - Monitoramento ativo", 0))

        return alert_messages

    def process_eye_detection(self, gray_frame: np.ndarray) -> List[Tuple[str, int]]:
        """Processa detecÃ§Ã£o de olhos fechados"""
        alert_messages = []
        faces = self.detector(gray_frame)

        for face in faces:
            landmarks = self.predictor(gray_frame, face)

            left_eye_ratio = self.get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = self.get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

            if blinking_ratio > self.config.EYE_CLOSED_THRESHOLD:
                self.state.closed_eyes_frames += 1
            else:
                self.state.closed_eyes_frames = 0
                self.alert_triggered_by_class["olhos_fechados"] = False

            if self.state.closed_eyes_frames >= self.config.ALERT_FRAME_THRESHOLD:
                if not self.alert_triggered_by_class["olhos_fechados"]:
                    self.play_alert_sound(self.config.ALERT_SOUND_PATH)
                    self.alert_triggered_by_class["olhos_fechados"] = True
                alert_messages.append(("ðŸ˜´ OLHOS FECHADOS - ALERTA!", self.state.closed_eyes_frames))
            elif self.state.closed_eyes_frames > 0:
                alert_messages.append(("ðŸ‘ï¸ Analisando olhos...", self.state.closed_eyes_frames))

        return alert_messages

    def process_posture_detection(self, rgb_frame: np.ndarray) -> List[Tuple[str, int]]:
        """Processa detecÃ§Ã£o de postura inclinada"""
        alert_messages = []

        try:
            pose_results = self.pose.process(rgb_frame)

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                nose = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
                l_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
                r_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])

                shoulder_midpoint = self.midpoint_float(l_shoulder, r_shoulder)
                distance = np.linalg.norm(nose - shoulder_midpoint)

                if distance > self.config.POSTURE_DISTANCE_THRESHOLD:
                    self.state.leaning_forward_frames += 1
                else:
                    self.state.leaning_forward_frames = 0
                    self.alert_triggered_by_class["inclinado"] = False

                if self.state.leaning_forward_frames >= self.config.ALERT_FRAME_THRESHOLD:
                    if not self.alert_triggered_by_class["inclinado"]:
                        self.play_alert_sound(self.config.ALERT_SOUND_PATH)
                        self.alert_triggered_by_class["inclinado"] = True
                    alert_messages.append(("ðŸ“ POSTURA INCLINADA - ALERTA!", self.state.leaning_forward_frames))
                elif self.state.leaning_forward_frames > 0:
                    alert_messages.append(("ðŸ“ Analisando postura...", self.state.leaning_forward_frames))

        except Exception as e:
            logging.warning(f"Erro na detecÃ§Ã£o de postura: {e}")

        return alert_messages

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        """Processa um frame completo com otimizaÃ§Ãµes"""
        start_time = time.time()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        all_alerts = []

        classes_seen, yolo_alerts = self.process_yolo_detection(frame)
        all_alerts.extend(yolo_alerts)

        frame_skip = self.config.performance.FRAME_SKIP
        if self.metrics.frame_count % frame_skip == 0:
            seatbelt_alerts = self.process_seatbelt_detection(classes_seen)
            all_alerts.extend(seatbelt_alerts)

            eye_alerts = self.process_eye_detection(gray_frame)
            all_alerts.extend(eye_alerts)

            posture_alerts = self.process_posture_detection(rgb_frame)
            all_alerts.extend(posture_alerts)

        processing_time = time.time() - start_time
        self.metrics.processing_time_ms = processing_time * 1000
        self.metrics.fps_current = 1 / (processing_time + 1e-6)
        self.metrics.frame_count += 1
        
        if self.metrics.frame_count > 1:
            self.metrics.fps_average = (self.metrics.fps_average * (self.metrics.frame_count - 1) + 
                                      self.metrics.fps_current) / self.metrics.frame_count

        if self.config.performance.MEMORY_OPTIMIZATION and self.metrics.frame_count % 100 == 0:
            gc.collect()

        return frame, all_alerts

class StreamlitApp:
    def __init__(self):
        self.config = Config()
        self.alert_system = None
        self.is_running = False

    def create_sidebar_controls(self):
        """Cria controles avanÃ§ados na sidebar"""
        with st.sidebar:
            st.header("âš™ï¸ ConfiguraÃ§Ãµes AvanÃ§adas")
            
            st.subheader("ðŸ“ ResoluÃ§Ã£o e Performance")
            
            resolution_options = list(self.config.performance.AVAILABLE_RESOLUTIONS.keys())
            selected_resolution = st.selectbox(
                "ResoluÃ§Ã£o de Processamento",
                resolution_options,
                index=resolution_options.index("Custom YOLO 640")
            )
            self.config.MODEL_RESOLUTION = self.config.performance.AVAILABLE_RESOLUTIONS[selected_resolution]
            
            st.subheader("ðŸŽ¯ ConfiguraÃ§Ãµes YOLO")
            
            col1, col2 = st.columns(2)
            with col1:
                self.config.performance.YOLO_CONFIDENCE = st.slider(
                    "ConfianÃ§a MÃ­nima", 0.1, 0.9, 0.25, 0.05
                )
            with col2:
                self.config.performance.YOLO_IOU_THRESHOLD = st.slider(
                    "IoU Threshold", 0.1, 0.9, 0.45, 0.05
                )
            
            self.config.performance.YOLO_MAX_DETECTIONS = st.slider(
                "MÃ¡x. DetecÃ§Ãµes", 10, 100, 50, 5
            )
            
            st.subheader("âš¡ OtimizaÃ§Ãµes")
            
            self.config.performance.FRAME_SKIP = st.slider(
                "Pular Frames (1 = todos)", 1, 30, 1
            )
            
            col1, col2 = st.columns(2)
            with col1:
                self.config.performance.YOLO_HALF_PRECISION = st.checkbox(
                    "Half Precision", value=False
                )
            with col2:
                self.config.performance.MEMORY_OPTIMIZATION = st.checkbox(
                    "OtimizaÃ§Ã£o MemÃ³ria", value=True
                )
            
            st.subheader("ðŸŽ¨ Display")
            
            col1, col2 = st.columns(2)
            with col1:
                self.config.performance.SHOW_CONFIDENCE = st.checkbox(
                    "Mostrar ConfianÃ§a", value=True
                )
                self.config.performance.SHOW_CLASS_NAMES = st.checkbox(
                    "Mostrar Classes", value=True
                )
            with col2:
                self.config.performance.BBOX_THICKNESS = st.slider(
                    "Espessura Bbox", 1, 5, 2
                )
            
            st.subheader("ðŸ”§ ConfiguraÃ§Ãµes de DetecÃ§Ã£o")
            
            self.config.EYE_CLOSED_THRESHOLD = st.slider(
                "Limiar Olhos Fechados", 3.0, 6.0, 4.5, 0.05
            )
            self.config.POSTURE_DISTANCE_THRESHOLD = st.slider(
                "Limiar Postura", 0.8, 4.0, 0.9, 0.05
            )
            self.config.ALERT_FRAME_THRESHOLD = st.slider(
                "Frames para Alerta", 5, 30, 15
            )

    def create_performance_dashboard(self, metrics: PerformanceMetrics):
        """Cria dashboard de performance"""
        st.subheader("ðŸ“Š Dashboard de Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "FPS Atual",
                f"{metrics.fps_current:.1f}",
                delta=f"MÃ©dia: {metrics.fps_average:.1f}"
            )
        
        with col2:
            st.metric(
                "Tempo Processamento",
                f"{metrics.processing_time_ms:.1f}ms",
                delta=f"CPU: {metrics.cpu_usage_percent:.1f}%"
            )
        
        with col3:
            st.metric(
                "MemÃ³ria",
                f"{metrics.memory_usage_mb:.1f}MB",
                delta=f"ResoluÃ§Ã£o: {metrics.resolution}"
            )
        
        with col4:
            st.metric(
                "DetecÃ§Ãµes",
                f"{metrics.detection_count}",
                delta=f"Frame: {metrics.frame_count}"
            )

    def run(self):
        st.title("ðŸ›¡ï¸ Sistema de Alerta Inteligente - ConfiguraÃ§Ãµes AvanÃ§adas")
        
        self.create_sidebar_controls()
        
        if self.alert_system is None or st.sidebar.button("ðŸ”„ Aplicar ConfiguraÃ§Ãµes"):
            try:
                self.alert_system = AlertSystem(self.config)
                st.success("âœ… ConfiguraÃ§Ãµes aplicadas com sucesso!")
            except Exception as e:
                st.error(f"âŒ Erro ao aplicar configuraÃ§Ãµes: {e}")
                return
        
        col1, col2, col3 = st.columns(3)
        with col1:
            run = st.button("â–¶ï¸ Iniciar Sistema", type="primary")
        with col2:
            stop = st.button("â¹ï¸ Parar Sistema")
        with col3:
            reset_metrics = st.button("ðŸ”„ Reset MÃ©tricas")
        
        if reset_metrics and self.alert_system:
            self.alert_system.metrics = PerformanceMetrics()
            st.rerun()
        
        show_video = st.checkbox("ðŸ“¹ Exibir vÃ­deo", value=True)
        show_performance = st.checkbox("ðŸ“Š Dashboard Performance", value=True)
        
        if show_performance:
            performance_container = st.container()
        
        frame_container = st.container()
        alert_container = st.container()
        
        if run and not stop and self.alert_system:
            self.run_detection(show_video, show_performance, 
                             performance_container if show_performance else None,
                             frame_container, alert_container)

    def run_detection(self, show_video, show_performance, performance_container,
                     frame_container, alert_container):
        """Executa o loop principal de detecÃ§Ã£o com mÃ©tricas avanÃ§adas"""
        cap = cv2.VideoCapture("videofinal_480p.mp4")

        if not cap.isOpened():
            st.error("âŒ Erro ao abrir vÃ­deo")
            return

        self.alert_system.metrics.resolution = f"{self.config.MODEL_RESOLUTION[0]}x{self.config.MODEL_RESOLUTION[1]}"
        
        try:
            frame_placeholder = frame_container.empty()
            alert_placeholder = alert_container.empty()
            performance_placeholder = performance_container.empty() if show_performance else None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.info("ðŸ“¹ Fim do vÃ­deo")
                    break

                frame = cv2.resize(frame, self.config.MODEL_RESOLUTION,
                                 interpolation=cv2.INTER_LINEAR)

                processed_frame, alerts = self.alert_system.process_frame(frame)
                
                self.alert_system.metrics.update_memory_cpu()

                if show_video:
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB",
                                          caption=f"ResoluÃ§Ã£o: {self.config.MODEL_RESOLUTION[0]}x{self.config.MODEL_RESOLUTION[1]} | "
                                                 f"ConfianÃ§a: {self.config.performance.YOLO_CONFIDENCE} | "
                                                 f"IoU: {self.config.performance.YOLO_IOU_THRESHOLD}")

                if show_performance and performance_placeholder:
                    with performance_placeholder.container():
                        self.create_performance_dashboard(self.alert_system.metrics)

                with alert_placeholder.container():
                    if alerts:
                        st.subheader("ðŸ“¢ Status do Sistema")
                        
                        alert_types = {"ðŸš¨ ALERTAS CRÃTICOS": [], "â„¹ï¸ MONITORAMENTO": []}
                        
                        for msg, score in alerts:
                            if "ALERTA" in msg:
                                alert_types["ðŸš¨ ALERTAS CRÃTICOS"].append(f"**{msg}** | Score: {score}")
                            else:
                                alert_types["â„¹ï¸ MONITORAMENTO"].append(f"{msg} | Score: {score}")
                        
                        for category, messages in alert_types.items():
                            if messages:
                                st.markdown(f"**{category}**")
                                for msg in messages:
                                    st.markdown(f"- {msg}")
                    else:
                        pass

                time.sleep(0.01)

        except KeyboardInterrupt:
            st.info("ðŸ›‘ Sistema interrompido pelo usuÃ¡rio")
        except Exception as e:
            st.error(f"âŒ Erro durante processamento: {e}")
        finally:
            cap.release()
            if self.config.performance.MEMORY_OPTIMIZATION:
                gc.collect()

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    
    st.set_page_config(
        page_title="Sistema de Alerta Inteligente",
        page_icon="ðŸ›¡ï¸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    try:
        app = StreamlitApp()
        app.run()
    except Exception as e:
        st.error(f"âŒ Erro ao inicializar aplicaÃ§Ã£o: {e}")
        st.info("Verifique se todos os arquivos de modelo estÃ£o no local correto.")

