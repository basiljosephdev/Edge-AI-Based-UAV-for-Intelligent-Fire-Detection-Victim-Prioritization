"""
mavlink_client.py — Background thread that reads GPS position, attitude and
battery telemetry from the Pixhawk over MAVLink and exposes the latest values
thread-safely to the rest of the application.

Wiring: Jetson Nano 40-pin header UART (TXD/RXD, e.g. /dev/ttyTHS1) to the
Pixhawk's TELEM2 port (cross TX->RX, RX->TX, plus GND). Set TELEM2 baud to
match `serial.baud` in settings.yaml (SERIAL2_BAUD parameter in ArduPilot,
value 921 for 921600).
"""

import threading
import time

from pymavlink import mavutil


class Telemetry:
    """Plain snapshot of the most recent flight-controller state."""

    def __init__(self):
        self.lat = None
        self.lon = None
        self.alt_m = None          # AMSL altitude
        self.relative_alt_m = None  # altitude above home/takeoff point
        self.heading_deg = None
        self.roll_deg = None
        self.pitch_deg = None
        self.battery_pct = None
        self.gps_fix_type = None
        self.last_update = 0.0

    def is_valid(self, max_age_sec=3.0):
        return self.lat is not None and (time.time() - self.last_update) < max_age_sec

    def as_dict(self):
        return {
            "lat": self.lat,
            "lon": self.lon,
            "alt_m": self.alt_m,
            "relative_alt_m": self.relative_alt_m,
            "heading_deg": self.heading_deg,
            "roll_deg": self.roll_deg,
            "pitch_deg": self.pitch_deg,
            "battery_pct": self.battery_pct,
            "gps_fix_type": self.gps_fix_type,
        }


class MavlinkClient:
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.telemetry = Telemetry()
        self._conn = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        # mavutil auto-detects udp:/tcp: prefixes for SITL; otherwise treats
        # `port` as a serial device and uses `baud`.
        if self.port.startswith("udp:") or self.port.startswith("tcp:"):
            self._conn = mavutil.mavlink_connection(self.port)
        else:
            self._conn = mavutil.mavlink_connection(self.port, baud=self.baud)
        self._conn.wait_heartbeat(timeout=10)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            msg = self._conn.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                continue
            msg_type = msg.get_type()
            with self._lock:
                if msg_type == "GLOBAL_POSITION_INT":
                    self.telemetry.lat = msg.lat / 1e7
                    self.telemetry.lon = msg.lon / 1e7
                    self.telemetry.alt_m = msg.alt / 1000.0
                    self.telemetry.relative_alt_m = msg.relative_alt / 1000.0
                    self.telemetry.heading_deg = msg.hdg / 100.0 if msg.hdg != 65535 else None
                    self.telemetry.last_update = time.time()
                elif msg_type == "ATTITUDE":
                    import math

                    self.telemetry.roll_deg = math.degrees(msg.roll)
                    self.telemetry.pitch_deg = math.degrees(msg.pitch)
                    self.telemetry.last_update = time.time()
                elif msg_type == "SYS_STATUS":
                    self.telemetry.battery_pct = msg.battery_remaining
                elif msg_type == "GPS_RAW_INT":
                    self.telemetry.gps_fix_type = msg.fix_type

    def get_telemetry(self):
        with self._lock:
            snap = Telemetry()
            snap.__dict__.update(self.telemetry.__dict__)
            return snap

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._conn is not None:
            self._conn.close()
