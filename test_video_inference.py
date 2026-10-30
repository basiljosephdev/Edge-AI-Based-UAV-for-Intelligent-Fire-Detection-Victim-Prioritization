#!/usr/bin/env python3
"""
test_video_inference.py — Real-time Video Inference for EdgeFireUAV
Processes test_video.mp4 with:
  1. Fire & Smoke Detection (fire_detector.onnx)
  2. Victim Posture Classification (Lying / Sitting / Standing)
     - Combines aerial person detection (YOLOv8n) with body geometry ratio:
       * Standing: h/w >= 1.35
       * Sitting:  0.75 <= h/w < 1.35
       * Lying:    h/w < 0.75
     - Also supports evaluating the custom drone-trained 3-class model.
Outputs an annotated video with HUD dashboard and saves summary statistics.
"""

import argparse
import os
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Distinct color palette (BGR)
COLORS = {
    "Lying": (0, 0, 255),       # Red (Critical priority victim)
    "Sitting": (0, 215, 255),   # Yellow/Gold (Medium priority)
    "Standing": (0, 255, 0),    # Green (Low priority / ambulatory)
    "fire": (0, 69, 255),       # Orange-Red
    "smoke": (180, 180, 180),   # Grey-White
}

def draw_hud(frame, frame_idx, total_frames, fps, posture_counts, fire_count):
    h, w = frame.shape[:2]
    # Semi-transparent overlay bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Text items
    info_text = f"Frame: {frame_idx}/{total_frames} ({fps:.1f} FPS)"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Posture counts
    x_offset = 260
    cv2.putText(frame, f"Lying: {posture_counts['Lying']}", (x_offset, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["Lying"], 2)
    x_offset += 120
    cv2.putText(frame, f"Sitting: {posture_counts['Sitting']}", (x_offset, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["Sitting"], 2)
    x_offset += 130
    cv2.putText(frame, f"Standing: {posture_counts['Standing']}", (x_offset, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["Standing"], 2)

    # Fire status
    x_offset += 140
    fire_text = f"Fire/Smoke: {fire_count}" if fire_count > 0 else "Fire: None"
    fire_color = (0, 69, 255) if fire_count > 0 else (120, 255, 120)
    cv2.putText(frame, fire_text, (x_offset, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fire_color, 2)


def process_video(video_path, output_path, conf_thresh=0.25, use_custom=False):
    print("=" * 60)
    print(f"EdgeFireUAV — Processing Video: {video_path}")
    print(f"Output Destination: {output_path}")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video Properties: {width}x{height} @ {orig_fps:.2f} FPS ({total_frames} frames)")

    # Load models
    print("\nLoading Person / Posture Detector...")
    if use_custom:
        custom_weights = "models/training/weights/best.pt" if os.path.exists("models/training/weights/best.pt") else "runs/detect/runs/custom_posture_detector/train-3/weights/best.pt"
        print(f"Using Custom 3-Class Posture Model: {custom_weights}")
        posture_model = YOLO(custom_weights)
        is_custom_model = True
    else:
        print("Using Base Aerial Person Detector (yolov8n.pt) + Geometry Posture Classifier")
        posture_model = YOLO("yolov8n.pt")
        is_custom_model = False

    # Check for Fire Model
    fire_model = None
    fire_path = "models/training/fire_detector.onnx"
    if os.path.exists(fire_path):
        try:
            print(f"Loading Fire Detector: {fire_path}")
            fire_model = YOLO(fire_path, task="detect")
        except Exception as e:
            print(f"Could not load fire detector ({e}), skipping fire detection.")

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, orig_fps, (width, height))

    frame_idx = 0
    start_time = time.time()
    total_detections_history = []

    print("\nStarting frame-by-frame inference...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t_frame_start = time.time()

        posture_counts = {"Lying": 0, "Sitting": 0, "Standing": 0}
        fire_count = 0

        # 1. Fire / Smoke Inference (optional)
        if fire_model is not None and frame_idx % 2 == 1:
            try:
                fire_res = fire_model.predict(frame, conf=0.35, verbose=False)
                for b in fire_res[0].boxes:
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    name = "smoke" if cls_id == 0 else "fire"
                    fire_count += 1
                    xyxy = [int(v) for v in b.xyxy[0].tolist()]
                    color = COLORS.get(name, (0, 0, 255))
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                    label = f"{name.upper()} {conf:.2f}"
                    cv2.putText(frame, label, (xyxy[0], max(15, xyxy[1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            except Exception:
                pass

        # 2. Person Posture Inference
        if is_custom_model:
            results = posture_model.predict(frame, conf=conf_thresh, imgsz=864, verbose=False)
            for b in results[0].boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                name = posture_model.names.get(cls_id, f"Class {cls_id}")
                if name in posture_counts:
                    posture_counts[name] += 1
                xyxy = [int(v) for v in b.xyxy[0].tolist()]
                color = COLORS.get(name, (0, 255, 0))
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                label = f"{name} {conf:.2f}"
                cv2.putText(frame, label, (xyxy[0], max(15, xyxy[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        else:
            # Detect person (class 0 in COCO)
            results = posture_model.predict(frame, conf=conf_thresh, classes=[0], imgsz=864, verbose=False)
            for b in results[0].boxes:
                conf = float(b.conf[0])
                xyxy = [int(v) for v in b.xyxy[0].tolist()]
                w_box = max(1, xyxy[2] - xyxy[0])
                h_box = max(1, xyxy[3] - xyxy[1])
                ratio = h_box / float(w_box)

                if ratio >= 1.35:
                    posture = "Standing"
                elif ratio >= 0.75:
                    posture = "Sitting"
                else:
                    posture = "Lying"

                posture_counts[posture] += 1
                color = COLORS[posture]

                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                label = f"{posture} {conf:.2f}"
                cv2.putText(frame, label, (xyxy[0], max(15, xyxy[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        t_frame_end = time.time()
        inst_fps = 1.0 / max(1e-5, t_frame_end - t_frame_start)

        # Draw Dashboard HUD
        draw_hud(frame, frame_idx, total_frames, inst_fps, posture_counts, fire_count)

        out.write(frame)

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            avg_fps = frame_idx / max(1e-5, elapsed)
            print(f"  Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) | "
                  f"Avg FPS: {avg_fps:.1f} | Active: "
                  f"Standing={posture_counts['Standing']} Sitting={posture_counts['Sitting']} Lying={posture_counts['Lying']}")

    cap.release()
    out.release()

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Inference Completed Successfully in {total_time:.2f}s!")
    print(f"Overall Processing Speed: {frame_idx / total_time:.1f} FPS")
    print(f"Saved annotated video to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    default_video = "test_media/test_video.mp4" if os.path.exists("test_media/test_video.mp4") else "test_video.mp4"
    default_out = "test_media/test_video_annotated.mp4" if os.path.isdir("test_media") else "test_video_annotated.mp4"

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=default_video, help="Path to input video")
    parser.add_argument("--out", default=default_out, help="Path to output video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--use_custom", action="store_true", help="Use custom drone posture model directly")
    args = parser.parse_args()

    process_video(args.video, args.out, conf_thresh=args.conf, use_custom=args.use_custom)
