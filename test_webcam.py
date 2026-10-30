"""
test_webcam.py — Real-time Webcam Posture & Person Tracker
Accurately detects and tracks the complete person, showing posture:
  - SITTING
  - STANDING
  - LYING

Uses YOLOv8 Pose Estimation (yolov8n-pose.pt) to understand human body
biomechanics and bounding box geometry in real-time, working seamlessly
for close-up webcam views, desk seating, and full-body postures.
Also allows switching to the custom drone-trained model with the 'm' key.

Controls:
  'q' : Quit
  'k' : Toggle skeleton overlay on/off
  'm' : Toggle mode (Pose Posture Tracker vs Drone Custom Model)
  's' : Save snapshot
  '+' / '-' : Increase / decrease confidence threshold
"""

import argparse
import os
import time
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

# Distinct color palette (BGR)
COLOR_PALETTE = {
    "Lying": (0, 0, 255),       # Red (Critical priority)
    "Sitting": (0, 215, 255),    # Yellow/Gold (Medium priority)
    "Standing": (0, 255, 0),     # Green (Low priority / ambulatory)
}

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),             # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),    # Upper body
    (5, 11), (6, 12), (11, 12),                 # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),     # Legs
]


def calculate_joint_angle(p1, p2, p3):
    """Calculates angle (degrees) at p2 between p1-p2 and p3-p2."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=float)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-6:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))))


def classify_posture_from_pose(box, kpts_data):
    """
    Classifies posture (Lying, Sitting, Standing) using skeletal joint kinematics,
    leg extension, and spatial alignment.
    box: [x1, y1, x2, y2]
    kpts_data: (17, 3) -> [x, y, conf]
    """
    x1, y1, x2, y2 = box
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    aspect_ratio = h / w

    # 1. Check Lying: body is horizontal
    if aspect_ratio < 0.8:
        return "Lying", 0.95

    if kpts_data is None or len(kpts_data) < 17:
        if aspect_ratio >= 1.3:
            return "Standing", 0.80
        return "Sitting", 0.75

    conf = kpts_data[:, 2]

    # Torso keypoints: Shoulders (5, 6) and Hips (11, 12)
    s_conf = (conf[5] + conf[6]) / 2.0
    h_conf = (conf[11] + conf[12]) / 2.0
    k_conf = (conf[13] + conf[14]) / 2.0
    a_conf = (conf[15] + conf[16]) / 2.0

    # 2. Check Spine Angle for Lying
    if s_conf > 0.25 and h_conf > 0.25:
        s_mid = (kpts_data[5, :2] + kpts_data[6, :2]) / 2.0
        h_mid = (kpts_data[11, :2] + kpts_data[12, :2]) / 2.0
        torso_dx = abs(s_mid[0] - h_mid[0])
        torso_dy = abs(s_mid[1] - h_mid[1])

        # If torso is horizontal -> Lying
        if torso_dx > torso_dy * 1.25:
            return "Lying", 0.95

    # 3. Analyze Legs and Joint Angles (Standing vs Sitting)
    if h_conf > 0.2 and k_conf > 0.2:
        s_mid = (kpts_data[5, :2] + kpts_data[6, :2]) / 2.0 if s_conf > 0.2 else np.array([(x1 + x2) / 2.0, y1 + 0.2 * h])
        h_mid = (kpts_data[11, :2] + kpts_data[12, :2]) / 2.0
        k_mid = (kpts_data[13, :2] + kpts_data[14, :2]) / 2.0

        torso_dy = max(15.0, h_mid[1] - s_mid[1])
        thigh_dy = k_mid[1] - h_mid[1]

        # Case A: Ankles are also detected -> Compute exact joint angles
        if a_conf > 0.2:
            a_mid = (kpts_data[15, :2] + kpts_data[16, :2]) / 2.0
            knee_angle = calculate_joint_angle(h_mid, k_mid, a_mid)
            torso_leg_angle = calculate_joint_angle(s_mid, h_mid, k_mid)

            # Sitting on a chair: knees bent sharply (~90 deg) and torso bent relative to thighs
            if knee_angle < 125.0 and torso_leg_angle < 130.0:
                return "Sitting", 0.92

            # Straight legs or upright thigh -> Standing
            if knee_angle >= 135.0 or thigh_dy > 0.5 * torso_dy:
                return "Standing", 0.95

        # Case B: Ankles are occluded/cut off, but knees are visible
        # If thighs extend downward vertically below hips -> Standing
        if thigh_dy > 0.45 * torso_dy:
            return "Standing", 0.92
        else:
            # Thighs are horizontal (knees at same level as hips) -> Sitting
            return "Sitting", 0.90

    # 4. Upper-body only view (legs not visible in frame, e.g., sitting at a desk)
    if k_conf < 0.2 and a_conf < 0.2:
        # Seated close-up at desk
        if aspect_ratio < 1.25:
            return "Sitting", 0.85
        else:
            return "Standing", 0.85

    # Default to Standing for upright bodies
    if aspect_ratio >= 1.2:
        return "Standing", 0.88
    return "Sitting", 0.80


def main():
    parser = argparse.ArgumentParser(description="Real-Time Webcam Posture & Person Tracker")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    args = parser.parse_args()

    project_dir = Path(os.getcwd())

    # Load Pose Model (General high-accuracy human posture tracking)
    print("=" * 60)
    print("Loading YOLOv8 Pose Model (yolov8n-pose.pt)...")
    pose_model = YOLO("yolov8n-pose.pt")

    # Load custom drone-trained weights if available
    drone_weights = project_dir / "models/training/weights/best.pt"
    if not drone_weights.exists():
        drone_weights = project_dir / "runs/detect/runs/custom_posture_detector/train-3/weights/best.pt"
    custom_model = None
    if drone_weights.exists():
        try:
            custom_model = YOLO(str(drone_weights))
            print(f"Loaded Drone-trained Custom Model: {drone_weights}")
        except Exception as e:
            print(f"Could not load custom model: {e}")
    print("=" * 60)

    # Open Webcam
    print(f"\nOpening webcam (device index: {args.cam})...")
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        print(f"[Error] Could not open webcam index {args.cam}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mode = "pose"  # 'pose' or 'custom'
    show_skeleton = True
    conf_thresh = args.conf

    print("\n[Controls]")
    print("  'q' : Quit")
    print("  'k' : Toggle skeleton overlay ON/OFF")
    print("  'm' : Toggle mode (Smart Pose Tracker <-> Drone Custom Model)")
    print("  's' : Save snapshot")
    print("  '+' / '-' : Increase / Decrease confidence threshold")
    print("-" * 60)

    prev_time = time.time()
    fps_smooth = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Could not read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps

        detections = []

        if mode == "pose":
            results = pose_model.predict(source=frame, conf=conf_thresh, verbose=False, device=0)
            for r in results:
                boxes = r.boxes
                kpts = r.keypoints.data if r.keypoints is not None else None

                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    # Class 0 in pose model is person
                    if cls_id != 0:
                        continue
                    box_xyxy = box.xyxy[0].cpu().numpy()
                    kpts_i = kpts[i].cpu().numpy() if (kpts is not None and i < len(kpts)) else None

                    posture_label, posture_conf = classify_posture_from_pose(box_xyxy, kpts_i)
                    overall_conf = float(box.conf[0]) * posture_conf
                    detections.append((posture_label, overall_conf, box_xyxy.astype(int), kpts_i))

        else:
            # Custom Drone Model
            if custom_model is not None:
                results = custom_model.predict(source=frame, conf=conf_thresh, verbose=False, device=0)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = custom_model.names.get(cls_id, f"Class_{cls_id}")
                        conf = float(box.conf[0])
                        box_xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        detections.append((cls_name, conf, box_xyxy, None))

        # Render Bounding Boxes & Skeleton
        for posture_label, conf, (x1, y1, x2, y2), kpts_i in detections:
            color = COLOR_PALETTE.get(posture_label, (0, 255, 0))

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Header pill
            label_text = f"{posture_label.upper()} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            bg_y1 = max(0, y1 - th - 12)
            cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 10, y1), color, -1)
            cv2.putText(
                frame,
                label_text,
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # Draw Skeleton Keypoints
            if show_skeleton and kpts_i is not None:
                for idx, (kx, ky, kc) in enumerate(kpts_i):
                    if kc > 0.4:
                        cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 255), -1)

                for p1, p2 in SKELETON_EDGES:
                    if kpts_i[p1, 2] > 0.4 and kpts_i[p2, 2] > 0.4:
                        pt1 = (int(kpts_i[p1, 0]), int(kpts_i[p1, 1]))
                        pt2 = (int(kpts_i[p2, 0]), int(kpts_i[p2, 1]))
                        cv2.line(frame, pt1, pt2, (255, 200, 0), 2)

        # HUD Overlay
        hud = frame.copy()
        cv2.rectangle(hud, (10, 10), (370, 95), (20, 20, 20), -1)
        frame = cv2.addWeighted(hud, 0.7, frame, 0.3, 0)

        mode_name = "Smart Pose Tracker" if mode == "pose" else "Drone Custom Model"
        cv2.putText(
            frame,
            f"Mode: {mode_name} (Press 'm')",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FPS: {fps_smooth:.1f} | People: {len(detections)} | Conf: {conf_thresh:.2f}",
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Skeleton: {'ON' if show_skeleton else 'OFF'} (Press 'k') | 'q': Quit",
            (20, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("EdgeFireUAV — Live Posture & Person Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("k"):
            show_skeleton = not show_skeleton
        elif key == ord("m") and custom_model is not None:
            mode = "custom" if mode == "pose" else "pose"
            print(f"Switched mode to: {mode}")
        elif key in [ord("+"), ord("=")]:
            conf_thresh = min(0.9, conf_thresh + 0.05)
        elif key in [ord("-"), ord("_")]:
            conf_thresh = max(0.1, conf_thresh - 0.05)
        elif key == ord("s"):
            fn = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fn, frame)
            print(f"Saved snapshot to: {fn}")

    cap.release()
    cv2.destroyAllWindows()
    print("Camera stream closed.")


if __name__ == "__main__":
    main()
