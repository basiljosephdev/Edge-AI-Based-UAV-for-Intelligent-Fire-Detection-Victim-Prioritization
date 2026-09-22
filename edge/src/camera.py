"""
camera.py — Threaded CSI camera capture for Jetson Nano.

Unlike a Raspberry Pi (which uses picamera2/libcamera), the Jetson Nano reads
its CSI camera (Raspberry Pi Camera Module V2 / IMX219) through the NVIDIA
Argus camera stack, exposed to OpenCV via a GStreamer pipeline using the
`nvarguscamerasrc` element. This requires an OpenCV build with GStreamer
support enabled — this is included by default in NVIDIA's JetPack OS image,
so no extra install is normally needed on a stock JetPack flash.

The capture runs in its own thread and always keeps only the latest frame,
so slow downstream inference never processes a backlog of stale frames.
"""

import threading
import time

import cv2


def _gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=21,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false"
    )


class JetsonCamera:
    """Background thread that continuously grabs frames from the CSI camera."""

    def __init__(self, cfg):
        self._pipeline = _gstreamer_pipeline(
            sensor_id=cfg.get("sensor_id", 0),
            capture_width=cfg.get("capture_width", 1280),
            capture_height=cfg.get("capture_height", 720),
            display_width=cfg.get("display_width", 640),
            display_height=cfg.get("display_height", 480),
            framerate=cfg.get("fps", 21),
            flip_method=cfg.get("flip_method", 0),
        )
        self._cap = None
        self._frame = None
        self._frame_ts = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._cap = cv2.VideoCapture(self._pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Could not open CSI camera via GStreamer/nvargus. "
                "Check that the ribbon cable is seated on CAM0, that no other "
                "process holds the camera, and that OpenCV was built with "
                "GStreamer support (`python3 -c \"import cv2; print(cv2.getBuildInformation())\"` "
                "should list GStreamer: YES)."
            )
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
                self._frame_ts = time.time()

    def read(self):
        """Return (frame, timestamp) for the most recently captured frame, or (None, 0)."""
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._frame_ts

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
