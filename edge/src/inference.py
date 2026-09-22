"""
inference.py — Runs the two TensorRT-accelerated models (fire/smoke detector
and victim posture detector) against a camera frame and returns decoded
detections.

Swapped from TFLite to TensorRT: same YOLO-style output decoding as before
(box + per-class scores in one tensor), just backed by trt_engine.TRTModel
instead of the TFLite interpreter. If your exported engine's output layout
differs (e.g. NMS baked into the ONNX export via `model.export(..., nms=True)`),
skip `_decode_yolo_output` entirely and read boxes/scores/classes straight
from the output tensor instead.
"""

import numpy as np
import cv2

from .trt_engine import TRTModel


def _nms(boxes, scores, iou_threshold):
    idxs = cv2.dnn.NMSBoxes(
        bboxes=[list(map(float, b)) for b in boxes],
        scores=[float(s) for s in scores],
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )
    if len(idxs) == 0:
        return []
    return [i[0] if isinstance(i, (list, np.ndarray)) else i for i in idxs]


def _decode_yolo_output(raw, class_names, frame_w, frame_h, score_threshold, iou_threshold):
    """Decode a (1, 4+num_classes, num_boxes) or (1, num_boxes, 4+num_classes)
    YOLO-style tensor into a list of detections."""
    arr = raw[0]
    if arr.shape[0] == 4 + len(class_names):
        arr = arr.T  # -> (num_boxes, 4+num_classes)

    boxes_xywh = arr[:, :4]
    class_scores = arr[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]

    keep = confidences > score_threshold
    boxes_xywh, class_ids, confidences = boxes_xywh[keep], class_ids[keep], confidences[keep]
    if len(confidences) == 0:
        return []

    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x = (cx - w / 2) * frame_w
    y = (cy - h / 2) * frame_h
    w_px = w * frame_w
    h_px = h * frame_h
    boxes_px = np.stack([x, y, w_px, h_px], axis=1)

    keep_idx = _nms(boxes_px, confidences, iou_threshold)

    detections = []
    for i in keep_idx:
        bx, by, bw, bh = boxes_px[i]
        detections.append(
            {
                "class": class_names[class_ids[i]],
                "confidence": float(confidences[i]),
                "bbox_px": (float(bx), float(by), float(bw), float(bh)),
                "center_px": (float(bx + bw / 2), float(by + bh / 2)),
            }
        )
    return detections


class InferenceEngine:
    def __init__(self, cfg):
        self.fire_model = TRTModel(cfg["fire_model_path"])
        self.posture_model = TRTModel(cfg["posture_model_path"])
        self.fire_classes = cfg["fire_classes"]
        self.posture_classes = cfg["posture_classes"]
        self.score_threshold = cfg.get("score_threshold", 0.45)
        self.iou_threshold = cfg.get("nms_iou_threshold", 0.45)

    def _preprocess(self, frame_bgr, model: TRTModel):
        img = cv2.resize(frame_bgr, (model.in_w, model.in_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(model.in_dtype) / np.array(255.0, dtype=model.in_dtype)
        return np.expand_dims(img, axis=0)

    def run(self, frame_bgr):
        h, w = frame_bgr.shape[:2]

        fire_input = self._preprocess(frame_bgr, self.fire_model)
        fire_raw = self.fire_model.infer(fire_input)
        fire_dets = _decode_yolo_output(
            fire_raw, self.fire_classes, w, h, self.score_threshold, self.iou_threshold
        )
        for d in fire_dets:
            d["severity_normalized"] = d["confidence"]
            d["severity_label"] = (
                "high" if d["confidence"] > 0.75 else "moderate" if d["confidence"] > 0.5 else "low"
            )

        posture_input = self._preprocess(frame_bgr, self.posture_model)
        posture_raw = self.posture_model.infer(posture_input)
        victim_dets = _decode_yolo_output(
            posture_raw, self.posture_classes, w, h, self.score_threshold, self.iou_threshold
        )

        return fire_dets, victim_dets
