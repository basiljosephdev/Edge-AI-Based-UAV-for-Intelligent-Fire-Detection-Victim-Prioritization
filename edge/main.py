#!/usr/bin/env python3
"""
main.py — Entry point for EdgeFireUAV on Jetson Nano.

Pipeline each loop iteration:
  1. Grab the latest camera frame (non-blocking; camera runs in its own thread)
  2. Run fire/smoke + victim posture inference (TFLite)
  3. Pull the latest Pixhawk telemetry (GPS/heading/altitude)
  4. Project each detection's pixel location to an estimated GPS coordinate
  5. Score and rank victims by rescue priority
  6. Log new detections to SQLite
  7. Push a JSON update to any connected GCS dashboards over WebSocket
  8. (optional, --debug) draw overlays and show a local preview window

On shutdown (Ctrl+C), writes a final JSON mission report to reports/.

Usage:
    python3 main.py            # normal headless run (as a systemd service)
    python3 main.py --debug    # also show an annotated preview window
"""

import argparse
import os
import signal
import sys
import time

import cv2
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from src.camera import JetsonCamera
from src.inference import InferenceEngine
from src.mavlink_client import MavlinkClient
from src.gps_projector import project_detection_to_gps
from src.priority import compute_priorities
from src.logger import MissionLogger
from src.streamer import Streamer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(path=os.path.join(BASE_DIR, "config", "settings.yaml")):
    with open(path) as f:
        return yaml.safe_load(f)


def annotate_frame(frame, fires):
    for f in fires:
        x, y, w, h = f["bbox_px"]
        label_text = f"{f['class'].upper()} {f['confidence']:.2f}"
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        cv2.putText(frame, label_text, (int(x), int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="show annotated preview window")
    args = parser.parse_args()

    cfg = load_config()

    print(f"[EdgeFireUAV] Starting on Jetson Nano — camera sensor {cfg['camera']['sensor_id']}")

    camera = JetsonCamera(cfg["camera"]).start()
    print("[EdgeFireUAV] Camera thread started.")

    engine = InferenceEngine(cfg["models"])
    print("[EdgeFireUAV] Fire Detection Model loaded.")

    mav = MavlinkClient(cfg["serial"]["port"], cfg["serial"]["baud"]).start()
    print(f"[EdgeFireUAV] MAVLink connected on {cfg['serial']['port']}.")

    logger = MissionLogger(
        os.path.join(BASE_DIR, cfg["database"]["path"]),
        cfg["reporting"]["drone_id"],
    )
    mission_id = logger.start_mission()
    print(f"[EdgeFireUAV] Mission started: {mission_id}")

    streamer = Streamer(cfg["streaming"]["host"], cfg["streaming"]["port"]).start()
    print(f"[EdgeFireUAV] WebSocket streaming on ws://{cfg['streaming']['host']}:{cfg['streaming']['port']}")

    running = {"flag": True}

    def _handle_sigint(signum, frame):
        running["flag"] = False

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    loop_period = 1.0 / cfg["loop"].get("target_hz", 10)
    frame_w = cfg["camera"]["display_width"]
    frame_h = cfg["camera"]["display_height"]
    cam_cfg = cfg["camera"]

    try:
        while running["flag"]:
            t0 = time.time()

            frame, frame_ts = camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            fire_dets = engine.run(frame)
            telemetry = mav.get_telemetry()

            for f in fire_dets:
                f["gps"] = project_detection_to_gps(f["center_px"], frame_w, frame_h, telemetry, cam_cfg)

            if fire_dets:
                logger.log_fire_zones(fire_dets)

            streamer.push({
                "type": "telemetry_update",
                "timestamp": frame_ts,
                "mission_id": mission_id,
                "telemetry": telemetry.as_dict(),
                "fire_zones": [
                    {"gps": f["gps"], "severity_label": f["severity_label"],
                     "severity_normalized": f["severity_normalized"]}
                    for f in fire_dets if f["gps"] is not None
                ],
            })

            if args.debug:
                annotated = annotate_frame(frame, fire_dets)
                cv2.imshow("EdgeFireUAV (debug)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed = time.time() - t0
            if elapsed < loop_period:
                time.sleep(loop_period - elapsed)

    finally:
        print("\n[EdgeFireUAV] Shutting down — writing mission report...")
        report_path = logger.end_mission(
            area_covered_m2=None,  # wire up real coverage-area estimation if you track flight path
            output_dir=os.path.join(BASE_DIR, cfg["reporting"]["output_dir"]),
        )
        print(f"[EdgeFireUAV] Mission report written: {report_path}")
        camera.stop()
        mav.stop()
        streamer.stop()
        if args.debug:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
