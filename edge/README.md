# EdgeFireUAV — edge/ (Jetson Nano + TensorRT build)

Adapted from the original project's Raspberry Pi 4 / TFLite design to run on
a **Jetson Nano** with a **Raspberry Pi Camera Module V2** (CSI) and
**TensorRT** inference, with the FLIR thermal camera path omitted.

## Building the two models — this is a 3-machine pipeline

TensorRT engines are tied to the exact GPU + TensorRT version they're built
on, so this can't be a single "run this script" step:

| Step | Where it runs | Script |
|---|---|---|
| 1. Train | PC / cloud GPU | `models/training/train_fire_detector.py`, `train_posture_detector.py` |
| 2. Export to ONNX | Same PC/GPU | `models/training/export_onnx.py` |
| 3. Copy the `.onnx` file(s) to the Nano | — | `scp` / USB drive |
| 4. Build the `.engine` | **On the Jetson Nano itself** | `models/training/build_tensorrt_engine.sh` |
| 5. Move `.engine` into `edge/models/`, run `main.py` | Jetson Nano | — |

**Datasets**: fire detection can use DFire out of the box (already in YOLO
format — see the original repo's `DATASET.md`). Posture detection has no
ready-made aerial dataset for the 5 classes — read the header comment in
`train_posture_detector.py` before you start; it explains what you'll need
to assemble yourself and roughly how.

**Model size**: use `yolov8n` (nano) as the base model for both — anything
bigger won't hit usable FPS on a Nano even with TensorRT/FP16.

## Before running edge/main.py

1. Both `.engine` files must exist in `edge/models/` (see pipeline above).
2. Confirm `tensorrt` and `pycuda` are importable — they come from JetPack,
   not pip: `python3 -c "import tensorrt, pycuda.autoinit"`.
3. Flight controller wiring: Jetson Nano 40-pin UART -> Pixhawk TELEM2
   (TX->RX, RX->TX, GND->GND), baud rate matching `config/settings.yaml`.
4. `pip3 install -r requirements.txt` for the remaining (non-Jetson-specific)
   dependencies.

## Running it

```bash
cd edge/
python3 main.py --debug     # shows a live annotated preview window
python3 main.py             # headless, for the systemd service
```

No hardware yet? Point `serial.port` in `config/settings.yaml` at
`udp:127.0.0.1:14550` and run the original repo's ArduPilot SITL simulator
alongside it.

## What the program actually outputs

1. **Console log** — startup messages, then a final line on shutdown
   pointing to the mission report file.
2. **`db/mission_log.sqlite`** — three tables (`missions`, `victims`,
   `fire_zones`) with every detection logged as it happens, timestamped and
   GPS-tagged.
3. **A live WebSocket feed** on `ws://<jetson-ip>:8765` (needs *some*
   network reachability to the Nano — see below) — every loop iteration
   (10 Hz by default) broadcasts a JSON message like:

   ```json
   {
     "type": "telemetry_update",
     "timestamp": 1737000000.12,
     "mission_id": "MISSION-20260913-101500",
     "telemetry": {"lat": 9.9312, "lon": 76.2673, "alt_m": 42.1,
                    "relative_alt_m": 15.0, "heading_deg": 87.3,
                    "battery_pct": 76, "gps_fix_type": 3},
     "fire_zones": [
       {"gps": {"lat": 9.93125, "lon": 76.26745, "alt_m": 15.0},
        "severity_label": "high", "severity_normalized": 0.85}
     ],
     "victims": [
       {"gps": {"lat": 9.93122, "lon": 76.26733, "alt_m": 15.2},
        "class": "Lying", "priority_score": 0.92,
        "severity_class": "CRITICAL", "rescue_rank": 1,
        "distance_to_nearest_fire_m": 3.8}
     ]
   }
   ```

4. **`reports/<mission_id>.json`** — written once, on shutdown (Ctrl+C or
   service stop), matching the schema in the original README's "Mission
   Reports" section: full victim list with GPS/posture/priority, plus a
   `recommended_rescue_order` array of victim IDs sorted by priority.

5. **(with `--debug`)** a live OpenCV preview window with green boxes around
   victims (labeled with posture, priority score, and rescue rank) and red
   boxes around fire/smoke detections.

## No WiFi on the Nano

Everything above except item 3 (the WebSocket feed) works with zero
network — camera is CSI, flight controller is UART/serial, and logging/
reports are local disk. If you want a laptop to see the live feed, options
are USB Ethernet direct to a laptop, a USB WiFi dongle, or just skip it —
the script runs fine with no clients connected.

There is no PDF generation in this edge module — the original design puts
that on the GCS side (`gcs/server/report_generator.py`, not yet built). The
JSON report above has everything needed to build that later.
