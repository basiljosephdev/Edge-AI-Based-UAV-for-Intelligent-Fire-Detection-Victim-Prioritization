"""
logger.py — Local SQLite mission logging + JSON mission report generation,
matching the schema shown in the README's "Mission Reports" section.
"""

import json
import os
import sqlite3
import time
from datetime import datetime, timezone


SCHEMA = """
CREATE TABLE IF NOT EXISTS missions (
    mission_id TEXT PRIMARY KEY,
    drone_id TEXT,
    start_time TEXT,
    end_time TEXT,
    area_covered_m2 REAL
);

CREATE TABLE IF NOT EXISTS victims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id TEXT,
    victim_id INTEGER,
    timestamp TEXT,
    lat REAL, lon REAL, alt_m REAL,
    posture TEXT,
    distance_to_nearest_fire_m REAL,
    fire_severity_normalized REAL,
    priority_score REAL,
    severity_class TEXT,
    rescue_rank INTEGER
);

CREATE TABLE IF NOT EXISTS fire_zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id TEXT,
    timestamp TEXT,
    lat REAL, lon REAL,
    severity_label TEXT,
    severity_normalized REAL
);
"""


class MissionLogger:
    def __init__(self, db_path, drone_id):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self.drone_id = drone_id
        self.mission_id = None
        self._start_time = None
        self._seen_victim_gps = []  # crude de-dupe of already-logged victims
        self._next_victim_id = 1

    def start_mission(self):
        self._start_time = datetime.now(timezone.utc)
        self.mission_id = "MISSION-" + self._start_time.strftime("%Y%m%d-%H%M%S")
        self.conn.execute(
            "INSERT INTO missions (mission_id, drone_id, start_time) VALUES (?, ?, ?)",
            (self.mission_id, self.drone_id, self._start_time.isoformat()),
        )
        self.conn.commit()
        return self.mission_id

    def _is_new_victim(self, gps, threshold_m=5.0):
        from .gps_projector import haversine_distance_m

        for lat, lon in self._seen_victim_gps:
            if haversine_distance_m(lat, lon, gps["lat"], gps["lon"]) < threshold_m:
                return False
        return True

    def log_victims(self, victims):
        ts = datetime.now(timezone.utc).isoformat()
        for v in victims:
            if v.get("gps") is None:
                continue
            is_new = self._is_new_victim(v["gps"])
            victim_id = self._next_victim_id if is_new else None
            if is_new:
                self._seen_victim_gps.append((v["gps"]["lat"], v["gps"]["lon"]))
                self._next_victim_id += 1
            self.conn.execute(
                """INSERT INTO victims
                   (mission_id, victim_id, timestamp, lat, lon, alt_m, posture,
                    distance_to_nearest_fire_m, fire_severity_normalized,
                    priority_score, severity_class, rescue_rank)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    self.mission_id,
                    victim_id if victim_id is not None else -1,
                    ts,
                    v["gps"]["lat"], v["gps"]["lon"], v["gps"]["alt_m"],
                    v["class"],
                    v.get("distance_to_nearest_fire_m"),
                    v.get("fire_severity_normalized"),
                    v.get("priority_score"),
                    v.get("severity_class"),
                    v.get("rescue_rank"),
                ),
            )
        self.conn.commit()

    def log_fire_zones(self, fires):
        ts = datetime.now(timezone.utc).isoformat()
        for f in fires:
            if f.get("gps") is None:
                continue
            self.conn.execute(
                """INSERT INTO fire_zones
                   (mission_id, timestamp, lat, lon, severity_label, severity_normalized)
                   VALUES (?,?,?,?,?,?)""",
                (
                    self.mission_id, ts, f["gps"]["lat"], f["gps"]["lon"],
                    f.get("severity_label"), f.get("severity_normalized"),
                ),
            )
        self.conn.commit()

    def end_mission(self, area_covered_m2, output_dir):
        end_time = datetime.now(timezone.utc)
        self.conn.execute(
            "UPDATE missions SET end_time=?, area_covered_m2=? WHERE mission_id=?",
            (end_time.isoformat(), area_covered_m2, self.mission_id),
        )
        self.conn.commit()
        return self._write_json_report(end_time, area_covered_m2, output_dir)

    def _write_json_report(self, end_time, area_covered_m2, output_dir):
        cur = self.conn.execute(
            """SELECT victim_id, timestamp, lat, lon, alt_m, posture,
                      distance_to_nearest_fire_m, fire_severity_normalized,
                      priority_score, severity_class, rescue_rank
               FROM victims WHERE mission_id=? AND victim_id != -1
               ORDER BY rescue_rank ASC""",
            (self.mission_id,),
        )
        victims = []
        for row in cur.fetchall():
            victims.append(
                {
                    "victim_id": row[0],
                    "timestamp": row[1],
                    "gps": {"lat": row[2], "lon": row[3], "alt_m": row[4]},
                    "posture": row[5],
                    "distance_to_nearest_fire_m": row[6],
                    "fire_severity_normalized": row[7],
                    "priority_score": row[8],
                    "severity_class": row[9],
                    "rescue_rank": row[10],
                }
            )

        fire_count = self.conn.execute(
            "SELECT COUNT(DISTINCT lat || ',' || lon) FROM fire_zones WHERE mission_id=?",
            (self.mission_id,),
        ).fetchone()[0]

        report = {
            "mission_id": self.mission_id,
            "drone_id": self.drone_id,
            "flight_duration_sec": int((end_time - self._start_time).total_seconds()),
            "start_time": self._start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "area_covered_m2": area_covered_m2,
            "fire_zones_detected": fire_count,
            "total_victims_detected": len(victims),
            "victims": victims,
            "recommended_rescue_order": [v["victim_id"] for v in victims],
        }

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.mission_id}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        return path
