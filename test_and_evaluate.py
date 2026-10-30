#!/usr/bin/env python3
"""
================================================================================
EdgeFireUAV - Model Testing, Inference & Quantitative Evaluation Pipeline
================================================================================
Capabilities:
1. Test Model Against Images (--mode test):
   - Runs posture and fire detection models against a single image or an entire folder.
   - Overlays tactical bounding boxes, aspect ratio calculations, and HUD triage summary.
   - Saves annotated visual outputs to an output folder (default: test_results/).

2. Compute Model Metrics & Generate Performance Curves (--mode eval):
   - Evaluates the trained YOLOv8 model against a validation dataset (data.yaml).
   - Computes: Precision, Recall, F1-Score, mAP@0.5, mAP@0.5:0.95 (overall & per class).
   - Generates & exports high-resolution visual curves:
     * Precision-Recall (PR) Curve
     * F1-Score vs. Confidence Threshold Curve
     * Precision vs. Confidence Curve
     * Recall vs. Confidence Curve
     * Normalized Confusion Matrix & Raw Counts Matrix
     * Complete metrics summary (JSON & terminal summary table).

Usage Examples:
  # 1. Test against images in test_media/
  python test_and_evaluate.py --mode test --source test_media/

  # 2. Test against a specific image
  python test_and_evaluate.py --mode test --source test_media/test1.jpg --conf 0.30

  # 3. Evaluate model metrics and generate PR / F1 / Confusion Matrix curves
  python test_and_evaluate.py --mode eval --data models/training/custom_posture_dataset/data.yaml

  # 4. Run both testing and full evaluation
  python test_and_evaluate.py --mode all
================================================================================
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics is required. Install via: pip install ultralytics")
    sys.exit(1)

# Default paths
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "models/training/weights/best.pt"
    if (PROJECT_ROOT / "models/training/weights/best.pt").exists()
    else PROJECT_ROOT / "runs/detect/runs/custom_posture_detector/train-3/weights/best.pt"
)
DEFAULT_FIRE_MODEL = PROJECT_ROOT / "models/training/fire_detector.onnx"
DEFAULT_DATA_YAML = PROJECT_ROOT / "models/training/custom_posture_dataset/data.yaml"
DEFAULT_TEST_DIR = PROJECT_ROOT / "test_media"

# Color scheme (BGR)
COLORS = {
    "Lying": (0, 0, 230),       # Bright Red (Critical)
    "Sitting": (0, 165, 255),   # Orange (Warning)
    "Standing": (40, 200, 40),  # Green (Ambulatory)
    "Fire": (0, 69, 255),       # Orange-Red
    "Smoke": (200, 200, 200),   # Gray
    "Person": (255, 180, 0),    # Cyan/Gold (Generic)
}

CLASS_NAMES_POSTURE = {0: "Lying", 1: "Sitting", 2: "Standing"}
CLASS_NAMES_FIRE = {0: "Smoke", 1: "Fire"}


# ==============================================================================
# 1. INFERENCE & IMAGE TESTING MODULE
# ==============================================================================

def draw_hud_header(img: np.ndarray, counts: Dict[str, int], fire_detected: bool) -> np.ndarray:
    """Draws a tactical HUD banner at the top of the image showing detection counts."""
    h, w = img.shape[:2]
    header_h = 55
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 24, 30), -1)
    img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)

    # Title
    cv2.putText(img, "EDGEFIRE-UAV TACTICAL INFERENCE", (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    # Status / Triage Banner
    if counts.get("Lying", 0) > 0:
        triage_text = f"TRIAGE: CRITICAL ({counts['Lying']} Lying)"
        triage_color = (0, 0, 255)
    elif counts.get("Sitting", 0) > 0:
        triage_text = f"TRIAGE: PRIORITY-2 ({counts['Sitting']} Sitting)"
        triage_color = (0, 165, 255)
    elif fire_detected:
        triage_text = "HAZARD: ACTIVE FIRE DETECTED"
        triage_color = (0, 69, 255)
    else:
        triage_text = "TRIAGE: STABLE / MONITORING"
        triage_color = (40, 200, 40)

    cv2.putText(img, triage_text, (15, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, triage_color, 2, cv2.LINE_AA)

    # Counts readout on right side
    stat_items = [
        f"Lying: {counts.get('Lying', 0)}",
        f"Sitting: {counts.get('Sitting', 0)}",
        f"Standing: {counts.get('Standing', 0)}",
        f"Fire: {counts.get('Fire', 0)}",
        f"Smoke: {counts.get('Smoke', 0)}"
    ]
    stat_str = " | ".join(stat_items)
    (tw, _), _ = cv2.getTextSize(stat_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(img, stat_str, (max(w - tw - 20, 400), 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    return img


def run_inference_on_image(
    image_path: Path,
    posture_model: YOLO,
    fire_model: Optional[YOLO] = None,
    conf_thresh: float = 0.25,
    save_path: Optional[Path] = None
) -> Dict:
    """Runs dual detection (fire + posture) on a single image and generates annotated output."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] Unable to load image: {image_path}")
        return {}

    orig_h, orig_w = img.shape[:2]
    annotated = img.copy()
    counts = {"Lying": 0, "Sitting": 0, "Standing": 0, "Fire": 0, "Smoke": 0}
    detections = []

    # 1. Fire / Smoke Detection
    if fire_model is not None:
        try:
            fire_results = fire_model(img, conf=conf_thresh, verbose=False)
            for res in fire_results:
                boxes = res.boxes
                for b in boxes:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

                    cls_name = CLASS_NAMES_FIRE.get(cls_id, f"Hazard_{cls_id}")
                    counts[cls_name] = counts.get(cls_name, 0) + 1
                    color = COLORS.get(cls_name, (0, 140, 255))

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf:.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, max(0, y1)), color, -1)
                    cv2.putText(annotated, label, (x1 + 3, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    detections.append({
                        "type": "hazard", "label": cls_name, "conf": round(conf, 3),
                        "box": [x1, y1, x2, y2]
                    })
        except Exception as e:
            print(f"[WARN] Fire detection failed on {image_path.name}: {e}")

    # 2. Posture Detection
    try:
        posture_results = posture_model(img, conf=conf_thresh, verbose=False)
        for res in posture_results:
            boxes = res.boxes
            for b in boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                bw, bh = max(1, x2 - x1), max(1, y2 - y1)
                aspect_ratio = bh / bw

                # Check if model has 3 posture classes (Lying, Sitting, Standing)
                # or base 80-class YOLO (Person = 0)
                if len(posture_model.names) == 3:
                    posture = CLASS_NAMES_POSTURE.get(cls_id, f"Class_{cls_id}")
                else:
                    # Fallback geometry heuristic if base person model
                    if cls_id != 0:
                        continue  # only person
                    if aspect_ratio >= 1.35:
                        posture = "Standing"
                    elif aspect_ratio >= 0.75:
                        posture = "Sitting"
                    else:
                        posture = "Lying"

                counts[posture] = counts.get(posture, 0) + 1
                color = COLORS.get(posture, (255, 255, 0))

                # Box & Label
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{posture} ({conf:.0%}, ar:{aspect_ratio:.2f})"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, max(0, y1)), color, -1)
                text_color = (255, 255, 255) if posture == "Lying" else (0, 0, 0)
                cv2.putText(annotated, label, (x1 + 3, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

                detections.append({
                    "type": "victim", "posture": posture, "conf": round(conf, 3),
                    "aspect_ratio": round(aspect_ratio, 2), "box": [x1, y1, x2, y2]
                })
    except Exception as e:
        print(f"[WARN] Posture detection failed on {image_path.name}: {e}")

    # Draw HUD
    annotated = draw_hud_header(annotated, counts, fire_detected=(counts.get("Fire", 0) > 0))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), annotated)

    return {
        "image": image_path.name,
        "resolution": f"{orig_w}x{orig_h}",
        "counts": counts,
        "detections": detections,
        "saved_to": str(save_path) if save_path else None
    }


def test_against_images(
    source_path: Path,
    weights_path: Path,
    fire_model_path: Optional[Path],
    conf_thresh: float,
    output_dir: Path
):
    """Batches inference across all specified images and displays summary."""
    print("=" * 70)
    print("EdgeFireUAV - Image Inference & Testing")
    print("=" * 70)
    print(f"Loading Posture Model : {weights_path}")
    posture_model = YOLO(str(weights_path))

    fire_model = None
    if fire_model_path and fire_model_path.exists():
        print(f"Loading Fire Model    : {fire_model_path}")
        try:
            fire_model = YOLO(str(fire_model_path), task="detect")
        except Exception as e:
            print(f"[WARN] Could not load fire model: {e}")

    # Discover images
    if source_path.is_file():
        image_files = [source_path]
    elif source_path.is_dir():
        image_files = sorted([
            p for p in source_path.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
            and not p.name.endswith(("_annotated.jpg", "_result.jpeg"))
        ])
    else:
        print(f"[ERROR] Source path does not exist: {source_path}")
        return

    if not image_files:
        print(f"[WARN] No suitable image files found in {source_path}")
        return

    print(f"\nProcessing {len(image_files)} test image(s)...")
    print(f"Output directory: {output_dir}\n")

    summary_list = []
    for idx, img_p in enumerate(image_files, 1):
        out_p = output_dir / f"{img_p.stem}_annotated.jpg"
        result = run_inference_on_image(
            img_p, posture_model, fire_model,
            conf_thresh=conf_thresh, save_path=out_p
        )
        if result:
            summary_list.append(result)
            c = result["counts"]
            print(f"[{idx}/{len(image_files)}] {img_p.name:<25} -> "
                  f"Lying: {c.get('Lying', 0)} | Sitting: {c.get('Sitting', 0)} | "
                  f"Standing: {c.get('Standing', 0)} | Fire: {c.get('Fire', 0)} | "
                  f"Smoke: {c.get('Smoke', 0)}")

    print("\n" + "=" * 70)
    print(f"Inference complete! Annotated images saved in: {output_dir}")
    print("=" * 70)


# ==============================================================================
# 2. MODEL METRICS & EVALUATION CURVES MODULE
# ==============================================================================

def evaluate_model_metrics(
    weights_path: Path,
    data_yaml_path: Path,
    output_dir: Path,
    imgsz: int = 640
) -> Dict:
    """
    Evaluates the YOLOv8 posture model on the validation dataset.
    Extracts Precision, Recall, mAP50, mAP50-95, and copies all generated
    metric curves (PR curve, F1-confidence, confusion matrix) to output_dir.
    """
    print("\n" + "=" * 70)
    print("EdgeFireUAV - Model Quantitative Metric Evaluation")
    print("=" * 70)
    print(f"Model Weights : {weights_path}")
    print(f"Dataset YAML  : {data_yaml_path}")
    print(f"Evaluation Dir: {output_dir}")

    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        return {}

    if not data_yaml_path.exists():
        print(f"[ERROR] Dataset configuration not found: {data_yaml_path}")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    val_run_dir = output_dir / "ultralytics_val_run"

    print("\nRunning Ultralytics validation pipeline...")
    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_yaml_path),
        split="val",
        imgsz=imgsz,
        save_json=True,
        plots=True,
        project=str(output_dir),
        name="val_run",
        exist_ok=True,
        verbose=True
    )

    # Harvest quantitative metrics
    mp = float(metrics.box.mp)        # Mean Precision
    mr = float(metrics.box.mr)        # Mean Recall
    map50 = float(metrics.box.map50)  # mAP@0.50
    map = float(metrics.box.map)      # mAP@0.50:0.95
    f1 = 2 * (mp * mr) / (mp + mr + 1e-9)

    class_metrics = {}
    names = metrics.names
    for i, name in names.items():
        if i < len(metrics.box.p):
            p = float(metrics.box.p[i])
            r = float(metrics.box.r[i])
            ap50 = float(metrics.box.ap50[i])
            ap = float(metrics.box.ap[i])
            c_f1 = 2 * (p * r) / (p + r + 1e-9)
            class_metrics[name] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1_score": round(c_f1, 4),
                "mAP50": round(ap50, 4),
                "mAP50_95": round(ap, 4)
            }

    summary = {
        "model_weights": str(weights_path.name),
        "dataset": str(data_yaml_path.name),
        "overall_metrics": {
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "f1_score": round(f1, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map, 4)
        },
        "per_class_metrics": class_metrics
    }

    # Save metrics JSON
    json_path = output_dir / "metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    # Locate and copy visual curve plots to top of output_dir
    run_dir = Path(metrics.save_dir) if hasattr(metrics, "save_dir") else output_dir / "val_run"
    plot_mappings = {
        "BoxPR_curve.png": "precision_recall_curve.png",
        "PR_curve.png": "precision_recall_curve.png",
        "BoxF1_curve.png": "f1_score_curve.png",
        "F1_curve.png": "f1_score_curve.png",
        "BoxP_curve.png": "precision_confidence_curve.png",
        "P_curve.png": "precision_confidence_curve.png",
        "BoxR_curve.png": "recall_confidence_curve.png",
        "R_curve.png": "recall_confidence_curve.png",
        "confusion_matrix_normalized.png": "confusion_matrix_normalized.png",
        "confusion_matrix.png": "confusion_matrix_counts.png",
        "val_batch0_labels.jpg": "val_sample_ground_truth.jpg",
        "val_batch0_pred.jpg": "val_sample_predictions.jpg"
    }

    copied_plots = []
    if run_dir.exists():
        for src_name, dst_name in plot_mappings.items():
            src_f = run_dir / src_name
            dst_f = output_dir / dst_name
            if src_f.exists() and not dst_f.exists():
                shutil.copy2(src_f, dst_f)
                copied_plots.append(dst_name)

    # Print clean results table
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS & MODEL PERFORMANCE METRICS")
    print("=" * 70)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'mAP@0.50':<12} {'mAP@0.5:0.95':<12}")
    print("-" * 75)
    for cls_name, m in class_metrics.items():
        print(f"{cls_name:<15} {m['precision']:<12.1%} {m['recall']:<12.1%} "
              f"{m['f1_score']:<12.1%} {m['mAP50']:<12.1%} {m['mAP50_95']:<12.1%}")
    print("-" * 75)
    print(f"{'OVERALL (ALL)':<15} {mp:<12.1%} {mr:<12.1%} {f1:<12.1%} {map50:<12.1%} {map:<12.1%}")
    print("=" * 70)

    print(f"\n[+] Metrics Summary Saved : {json_path}")
    print("[+] Generated Metric Curves & Plots:")
    for plot in copied_plots:
        print(f"    - {output_dir / plot}")

    return summary


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EdgeFireUAV - Model Testing & Quantitative Metric Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode", choices=["test", "eval", "all"], default="all",
        help="Execution mode: 'test' (run on images), 'eval' (compute metrics & curves), 'all' (both)"
    )
    parser.add_argument(
        "--source", default=str(DEFAULT_TEST_DIR),
        help=f"Path to input image or directory of images (default: {DEFAULT_TEST_DIR})"
    )
    parser.add_argument(
        "--weights", default=str(DEFAULT_WEIGHTS),
        help=f"Path to posture detector weights (default: {DEFAULT_WEIGHTS})"
    )
    parser.add_argument(
        "--fire_model", default=str(DEFAULT_FIRE_MODEL),
        help=f"Path to fire/smoke model (default: {DEFAULT_FIRE_MODEL})"
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA_YAML),
        help=f"Path to dataset data.yaml for validation evaluation (default: {DEFAULT_DATA_YAML})"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for testing inference (default: 0.25)"
    )
    parser.add_argument(
        "--test_out", default="test_results",
        help="Output directory for annotated test images (default: test_results)"
    )
    parser.add_argument(
        "--eval_out", default="evaluation_results",
        help="Output directory for quantitative metrics and curve plots (default: evaluation_results)"
    )

    args = parser.parse_args()

    weights_p = Path(args.weights)
    fire_p = Path(args.fire_model) if args.fire_model else None
    source_p = Path(args.source)
    data_p = Path(args.data)
    test_out_p = Path(args.test_out)
    eval_out_p = Path(args.eval_out)

    # Mode 1: Image Testing / Inference
    if args.mode in ["test", "all"]:
        test_against_images(
            source_path=source_p,
            weights_path=weights_p,
            fire_model_path=fire_p,
            conf_thresh=args.conf,
            output_dir=test_out_p
        )

    # Mode 2: Model Quantitative Metrics & Curves
    if args.mode in ["eval", "all"]:
        if not data_p.exists():
            print(f"\n[INFO] Dataset YAML ({data_p}) not found. Skipping evaluation curves.")
            print("To run metric curves, ensure models/training/custom_posture_dataset/data.yaml is present.")
        else:
            evaluate_model_metrics(
                weights_path=weights_p,
                data_yaml_path=data_p,
                output_dir=eval_out_p
            )


if __name__ == "__main__":
    main()

