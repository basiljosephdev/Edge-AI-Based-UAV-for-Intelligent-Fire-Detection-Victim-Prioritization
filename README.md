# EdgeFireUAV: Autonomous UAV Edge Fire Detection & Victim Posture Classification System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8n-orange.svg)](https://github.com/ultralytics/ultralytics)
[![ONNX](https://img.shields.io/badge/ONNX-Opset%2012-brightgreen.svg)](https://onnx.ai/)
[![NVIDIA Jetson](https://img.shields.io/badge/NVIDIA-Jetson%20Nano-76B900.svg)](https://developer.nvidia.com/embedded/jetson-nano)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end edge-computing aerial search-and-rescue (SAR) system designed for low-power unmanned aerial vehicles (UAVs). EdgeFireUAV simultaneously detects **wildfire/smoke hazards** and classifies **human victim postures** (*Lying, Sitting, Standing*) in real time directly on an onboard NVIDIA Jetson Nano.

Victim geolocations are estimated via 3D ray-casting using UAV altitude and gimbal attitude telemetry (Pixhawk / MAVLink), and victims are prioritized using a mathematical triage score to accelerate search and rescue operations in disaster scenarios.

---

## 📌 Key Highlights

- **Dual Edge AI Inference:** Runs optimized lightweight models (YOLOv8n base, ONNX / TensorRT FP16) for simultaneous flame/smoke detection and human posture categorization.
- **Multi-Modal Posture Analysis:**
  - **Skeletal Joint Kinematics:** Computes torso inclination and knee flexion angles ($\theta_{\text{knee}} = \angle(\text{Hip}, \text{Knee}, \text{Ankle})$) using YOLOv8-Pose (17 COCO keypoints).
  - **Aerial Aspect-Ratio Heuristics:** Employs normalized bounding box geometry ($R = h/w$) to classify victims at extreme nadir/top-down viewing angles:
    - **Lying (Critical / Unconscious):** $R < 0.75$
    - **Sitting (Injured / Incapacitated):** $0.75 \le R < 1.35$
    - **Standing (Ambulatory):** $R \ge 1.35$
- **Onboard Ray-Casting Geolocation:** Projects 2D camera pixel coordinates into GPS coordinates $(\text{Lat}, \text{Lon})$ using UAV telemetry (altitude, roll, pitch, heading) and camera field-of-view (FOV).
- **Rescue Triage Prioritization:** Formulates a rescue urgency score:
  $$P_{\text{victim}} = w_{\text{posture}} S_{\text{posture}} + w_{\text{dist}} \left(1 - \frac{d_{\text{fire}}}{d_{\text{max}}}\right) + w_{\text{fire}} S_{\text{fire}}$$
- **Low-Latency Edge Performance:** Optimized with ONNX and TensorRT FP16 to achieve 15–20 FPS on 5–10W edge hardware.

---

## 📊 Quantitative Experimental Results

| Metric | Posture Classification (YOLOv8n) | Fire / Smoke Detector |
|---|---|---|
| **Precision ($P$)** | **99.3%** | 94.8% |
| **Recall ($R$)** | **99.9%** | 92.6% |
| **mAP@0.50** | **99.5%** | 96.1% |
| **mAP@0.50:0.95** | **80.6%** | 71.4% |
| **Base Parameters** | 3.2M | 3.2M |
| **Exported Model Size** | 12.2 MB (ONNX) | 12.2 MB (ONNX) |

---

## 🏗️ System Architecture

```
                    ┌─────────────────────────┐
                    │      UAV Sensors        │
                    │  - CSI Camera (IMX219)  │
                    │  - Pixhawk Flight Ctrl  │
                    └────────────┬────────────┘
                                 │ Frame & MAVLink Telemetry
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│               NVIDIA Jetson Nano Edge Computing                 │
│                                                                 │
│  ┌───────────────────────┐         ┌─────────────────────────┐  │
│  │   Fire/Smoke Engine   │         │ Custom Posture Detector │  │
│  │    (TensorRT FP16)    │         │     (TensorRT FP16)     │  │
│  └───────────┬───────────┘         └────────────┬────────────┘  │
│              │                                  │               │
│              └───────────────┬──────────────────┘               │
│                              ▼                                  │
│                 ┌─────────────────────────┐                     │
│                 │  Ray-Casting Geolocation│                     │
│                 │   & 3-Level Triage      │                     │
│                 └────────────┬────────────┘                     │
│                              │                                  │
│                 ┌────────────┴────────────┐                     │
│                 │ Local SQLite Persistence│                     │
│                 └─────────────────────────┘                     │
└────────────────────────────────┬────────────────────────────────┘
                                 │ WebSocket Stream (ws://:8765)
                                 ▼
                    ┌─────────────────────────┐
                    │  Ground Control Station │
                    │   Tactical Map & HUD    │
                    └─────────────────────────┘
```

---

## 📂 Project Structure

```
├── edge/                            # Onboard Jetson Nano deployment codebase
│   ├── config/settings.yaml         # Camera FOV, MAVLink baud, detection thresholds
│   ├── src/
│   │   ├── camera.py                # GStreamer CSI camera pipeline (IMX219)
│   │   ├── gps_projector.py         # 3D ray-casting pixel-to-GPS transformation
│   │   ├── inference.py             # Dual TensorRT FP16 engine wrapper & NMS
│   │   ├── mavlink_client.py        # MAVLink serial bridge (Pixhawk TELEM2)
│   │   ├── priority.py              # Mathematical rescue triage calculator
│   │   ├── streamer.py              # Real-time WebSocket telemetry server
│   │   └── logger.py                # Local SQLite mission logging
│   ├── main.py                      # Master edge orchestration loop
│   └── requirements.txt             # Edge runtime dependencies
├── models/
│   └── training/
│       ├── build_tensorrt_engine.sh # TensorRT engine compilation script (trtexec)
│       ├── export_onnx.py           # PyTorch .pt to ONNX Opset 12 exporter
│       ├── prepare_custom_posture_dataset.py  # Dataset sanitization & YOLO conversion
│       ├── train_custom_posture_detector.py   # YOLOv8 posture training script
│       ├── train_fire_detector.py             # YOLOv8 fire training script
│       ├── custom_posture_detector.onnx       # Exported ONNX posture model (12MB)
│       ├── fire_detector.onnx                 # Exported ONNX fire model (12MB)
│       └── weights/best.pt                    # Trained posture checkpoint
├── report_assets/                   # High-res confusion matrices, loss curves, and diagrams
├── test_media/                      # Test media repository (images and videos)
│   ├── test1.jpg                    # Sample aerial test scene 1
│   ├── test2.jpg                    # Sample aerial test scene 2
│   ├── test.jpeg                    # Sample test scene
│   └── test_video.mp4               # Sample aerial drone test video
├── test_and_evaluate.py             # Image testing and quantitative metrics/curves evaluation tool
├── test_video_inference.py          # Video demonstration with tactical HUD display
├── test_webcam.py                   # Live webcam testing with keypoint kinematics
├── yolov8n-pose.pt                  # YOLOv8n Pose estimation model
├── yolov8n.pt                       # YOLOv8n baseline model
├── Untitled.ipynb                   # Aerial dual-detection Jupyter notebook
├── Untitled1.ipynb                  # Extended evaluation and test notebook
└── PROJECT_STATUS_AND_MODIFICATIONS.txt # Engineering log & roadmap report
```

---

## 🚀 Getting Started

### 1. Installation

Clone this repository and install Python dependencies:

```bash
git clone https://github.com/basiljosephdev/fire_detection_human_classification_aerial.git
cd fire_detection_human_classification_aerial
pip install -r edge/requirements.txt
pip install ultralytics opencv-python onnx onnxruntime
```

### 2. Testing Models Against Images & Computing Metrics / Curves

Use `test_and_evaluate.py` to test models against images or compute exact metrics and curves:

```bash
# A. Test against all images in test_media/ (saves annotated images to test_results/)
python test_and_evaluate.py --mode test --source test_media/

# B. Test against a specific image
python test_and_evaluate.py --mode test --source test_media/test1.jpg --conf 0.30

# C. Compute quantitative metrics & generate PR curves, F1 curves, and Confusion Matrices
python test_and_evaluate.py --mode eval --data models/training/custom_posture_dataset/data.yaml

# D. Run both image testing and full metric curve generation
python test_and_evaluate.py --mode all
```

**Generated Metric Artifacts** (saved in `evaluation_results/`):
- `precision_recall_curve.png` (PR Curve, mAP@0.5 = 99.5%)
- `f1_score_curve.png` (Optimal decision confidence threshold curve)
- `confusion_matrix_normalized.png` & `confusion_matrix_counts.png`
- `metrics_summary.json` (Structured JSON with Precision, Recall, F1, mAP50, and mAP50-95 per class)

### 3. Video Inference & Tactical HUD Demo

Run the video inference demo with real-time tactical overlays, victim bounding boxes, posture classification, and fire detection:

```bash
python test_video_inference.py --video test_media/test_video.mp4 --conf 0.35 --use_custom
```

### 4. Interactive Webcam Testing (Kinematic Pose + Custom Drone Model)

Run the real-time webcam test tool:

```bash
python test_webcam.py --cam 0
```
- Press `k` to toggle skeletal joint connections.
- Press `m` to switch between **Pose Kinematics Mode** and **Trained Drone Detector Mode**.
- Press `q` to exit.

### 5. Running the Jupyter Notebooks

Launch Jupyter to explore dual-inference on sample aerial imagery:

```bash
jupyter notebook Untitled.ipynb
```

---

## ⚡ Edge Jetson Nano Deployment

1. **Compile TensorRT FP16 Engines:**
   Transfer the ONNX models to the Jetson Nano and execute:
   ```bash
   bash models/training/build_tensorrt_engine.sh
   ```
2. **Configure Hardware Interface:**
   Update `edge/config/settings.yaml` with your Pixhawk connection (e.g., `/dev/ttyTHS1`, 921600 baud) and camera calibration settings.
3. **Launch Edge Node:**
   ```bash
   python edge/main.py --config edge/config/settings.yaml
   ```

---

## 📜 Academic Report & Visual Evidence

Detailed diagrams, normalized confusion matrices, PR curves, and batch predictions are available in [`report_assets/`](report_assets/) with an accompanying citation and discussion guide in [`report_assets/README_REPORT_GUIDE.md`](report_assets/README_REPORT_GUIDE.md).

---

## 📄 License

This project is released under the [MIT License](LICENSE).
