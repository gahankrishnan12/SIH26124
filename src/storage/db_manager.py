"""
SIH26124: SQLite Storage Manager Module
Provides structured persistence, querying, statistics, and CSV/JSON exports for UrbanEvent records.
"""
import sqlite3
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.events.schema import UrbanEvent
from config import settings

class DatabaseManager:
    """
    Manages local SQLite persistence for urban events.
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else settings.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_database()

    def get_connection(self) -> sqlite3.Connection:
        """Create a connection with Row factory enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_database(self) -> None:
        """Create the events table and performance indexes if not exists."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    severity TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    detection_mode TEXT NOT NULL,
                    gps_mode TEXT NOT NULL,
                    frame_index INTEGER NOT NULL,
                    bbox TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_class ON events(class_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
            conn.commit()
        finally:
            conn.close()

    def insert_event(self, event: UrbanEvent) -> bool:
        """Insert a single UrbanEvent into the database."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO events (
                    event_id, event_type, class_name, confidence, severity,
                    latitude, longitude, timestamp, source_id,
                    detection_mode, gps_mode, frame_index, bbox
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.event_type,
                event.class_name,
                event.confidence,
                event.severity,
                event.latitude,
                event.longitude,
                event.timestamp,
                event.source_id,
                event.detection_mode,
                event.gps_mode,
                event.frame_index,
                json.dumps(event.bbox)
            ))
            conn.commit()
            return True
        finally:
            conn.close()

    def insert_events(self, events: List[UrbanEvent]) -> int:
        """Batch insert multiple UrbanEvents inside a single transaction."""
        if not events:
            return 0
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            rows = [
                (
                    e.event_id,
                    e.event_type,
                    e.class_name,
                    e.confidence,
                    e.severity,
                    e.latitude,
                    e.longitude,
                    e.timestamp,
                    e.source_id,
                    e.detection_mode,
                    e.gps_mode,
                    e.frame_index,
                    json.dumps(e.bbox)
                )
                for e in events
            ]
            cursor.executemany("""
                INSERT OR REPLACE INTO events (
                    event_id, event_type, class_name, confidence, severity,
                    latitude, longitude, timestamp, source_id,
                    detection_mode, gps_mode, frame_index, bbox
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            return len(rows)
        finally:
            conn.close()

    def get_events(self, limit: int = 500, offset: int = 0) -> List[UrbanEvent]:
        """Fetch all events ordered chronologically."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ? OFFSET ?", (limit, offset))
            rows = cursor.fetchall()
            return [UrbanEvent.from_dict(dict(row)) for row in rows]
        finally:
            conn.close()

    def filter_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        detection_mode: Optional[str] = None,
        class_name: Optional[str] = None,
        limit: int = 500
    ) -> List[UrbanEvent]:
        """Filter events by optional parameters."""
        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.upper())
        if severity:
            query += " AND severity = ?"
            params.append(severity.lower())
        if detection_mode:
            query += " AND detection_mode = ?"
            params.append(detection_mode)
        if class_name:
            query += " AND class_name = ?"
            params.append(class_name.lower())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [UrbanEvent.from_dict(dict(row)) for row in rows]
        finally:
            conn.close()

    def get_event_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics directly in SQLite."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM events")
            total_count = cursor.fetchone()[0]

            cursor.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type")
            by_type = dict(cursor.fetchall())

            cursor.execute("SELECT class_name, COUNT(*) FROM events GROUP BY class_name")
            by_class = dict(cursor.fetchall())

            cursor.execute("SELECT severity, COUNT(*) FROM events GROUP BY severity")
            by_severity = dict(cursor.fetchall())

            cursor.execute("SELECT detection_mode, COUNT(*) FROM events GROUP BY detection_mode")
            by_mode = dict(cursor.fetchall())

            return {
                "total_events": total_count,
                "by_event_type": by_type,
                "by_class": by_class,
                "by_severity": by_severity,
                "by_detection_mode": by_mode,
                "gps_mode": "SIMULATED",
                "db_path": str(self.db_path)
            }
        finally:
            conn.close()

    def export_events_csv(self, file_path: Optional[str] = None) -> str:
        """Export all events to a CSV file."""
        out_path = Path(file_path) if file_path else settings.EVENTS_DATA_DIR / "events_export.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        events = self.get_events(limit=100000)
        fieldnames = [
            "event_id", "timestamp", "event_type", "class_name", "confidence",
            "severity", "latitude", "longitude", "source_id", "detection_mode",
            "gps_mode", "frame_index", "bbox"
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for evt in events:
                d = evt.to_dict()
                d["bbox"] = json.dumps(d["bbox"])
                writer.writerow(d)

        return str(out_path)

    def export_events_json(self, file_path: Optional[str] = None) -> str:
        """Export all events to a JSON file."""
        out_path = Path(file_path) if file_path else settings.EVENTS_DATA_DIR / "events_export.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        events = self.get_events(limit=100000)
        data = [e.to_dict() for e in events]

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return str(out_path)

    def clear_events(self) -> None:
        """Clear all records from events table."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM events")
            conn.commit()
        finally:
            conn.close()
