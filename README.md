
# 🚨 Risk Behavior Detection System for Drivers

This project uses **YOLOv5, YOLOv8, YOLOv11, and YOLOv12** (compatible formats: `.pt` [PyTorch], `.tflite` [TensorFlow Lite float32, float16, int8, and EdgeTPU]), along with **MediaPipe**, **Dlib**, and **OpenCV** to detect dangerous driver behaviors such as:

- Drowsiness  
- Mobile phone usage  
- No seatbelt  
- Excessive body leaning  
- Smoking  

⚠️ When suspicious behavior is detected, an **audible alert** is automatically triggered.

---

## ✅ Features

- Object detection using **YOLO** (`drowsy`, `phone`, `smoking`, `seatbelt`, `awake`)
- **Eye closure detection** using Dlib (facial landmarks and Blink Ratio — BR)
- **Posture analysis** using MediaPipe (calculating Nose-to-Shoulder Center ratio — DSC)
- **Automatic audible alert** using `playsound`
- Real-time interface with **OpenCV**

---

## 🧰 Requirements

### Python Libraries

Clone the repository:

```bash
git clone https://github.com/HyAgOsK/Drowsy_detection.git
cd Drowsy_detection/
```

Create and activate a virtual environment (recommended for Raspberry Pi 4/5):

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Active the default performace EdgeTPU Coral USB

```bash
silvatpu-linux-setup
```

Activate the `max` performace of EdgeTPU Coral USB

```bash
silvatpu-linux-setup --speed max

```

> ⚠️ **Note:** Installing `dlib` requires **CMake** and a compatible C++ compiler.
> 
### Other Prerequisites

- Python 3.8 or higher  
- Coral EdgeTPU (optional, for model acceleration)  
- Webcam or video file  
- Required files:
  - `shape_predictor_68_face_landmarks.dat` (Dlib facial landmark model)
  - `alarm.wav` (alert sound)
  - Trained YOLO model (`best.pt`) .pt .tflite
  - YOLO tracking configuration file: `botsort.yaml` (if using `track()`)

---

## 🗂️ Expected Folder Structure

```
project/
│
├── alarm.wav
├── shape_predictor_68_face_landmarks.dat
├── main_final.py                    # main script
├── botsort.yaml              # YOLO tracker configuration
├── runs_yolov5/
│   └── detect/
│       └── train/
│           └── weights/
│               ├── best.pt
│               ├── model_float32.tflite
│               ├── model_int8_edgetpu.tflite
│               └── ...
```

---

## ▶️ How to Run

Run the main script:

```bash
python main.py
```

A window will open showing the webcam feed and displaying visual and audible alerts in real time.

---

## 🧠 Alert Logic

- **Closed eyes** for 10 consecutive frames (Blink Ratio BR > 4.5) **and** YOLO detects `drowsy` ➜ Alert
- **Head leaning forward** (DSC > 2.9) for 10 frames **and** YOLO detects `drowsy` ➜ Alert
- YOLO detects `drowsy`, `phone`, or `smoking` for 10 frames ➜ Alert
- Any **combination of 2 or more active alerts at the same time** ➜ Triggers an audible alert once

---

## 📈 Performance

- **FPS** displayed in real-time in the terminal  
- Uses **GPU (CUDA)** when available  
- Compatible with **EdgeTPU** when a suitable `.tflite` model is loaded in `main.py`

---

## 📌 Final Notes

- Ensure your YOLO model is trained for the expected classes: `drowsy`, `phone`, `smoking`, `seatbelt`, `awake`
- By default, the script uses the webcam (`cv2.VideoCapture(0)`) — modify this to load a video file if needed from the root directory

---
