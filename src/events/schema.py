"""
SIH26124: Event Schema Module
Defines standard structured event data structures for urban transit intelligence.
"""
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import json
import uuid
from datetime import datetime, timezone

@dataclass
class UrbanEvent:
    """
    Standard data schema for all urban mobility and road condition events.
    """
    event_id: str
    event_type: str            # "VEHICLE" or "ROAD_DAMAGE"
    class_name: str            # e.g., "pothole", "car", "bus", "truck", "motorcycle", "person"
    confidence: float          # Model detection confidence [0.0 - 1.0]
    severity: str              # "low", "medium", "high", "none"
    latitude: float            # Latitude coordinate
    longitude: float           # Longitude coordinate
    timestamp: str             # ISO 8601 string (e.g. "2026-08-28T07:00:00Z")
    source_id: str = "BUS_DEMO_01"  # Simulated bus vehicle identifier
    detection_mode: str = "REAL_AI" # "REAL_AI" or "DEMO_SIMULATION"
    gps_mode: str = "SIMULATED"     # Explicitly "SIMULATED" for this MVP
    frame_index: int = 0
    bbox: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0]) # [x1, y1, x2, y2]

    @classmethod
    def create(
        cls,
        event_type: str,
        class_name: str,
        confidence: float,
        latitude: float,
        longitude: float,
        frame_index: int,
        bbox: List[float],
        severity: str = "none",
        timestamp: Optional[str] = None,
        source_id: str = "BUS_DEMO_01",
        detection_mode: str = "REAL_AI",
        gps_mode: str = "SIMULATED",
        event_id: Optional[str] = None
    ) -> "UrbanEvent":
        """
        Factory method to create a properly formatted UrbanEvent with auto-generated ID and timestamp.
        """
        if event_id is None:
            # Generate clean deterministic timestamped ID: EVT-YYYYMMDD-HHMMSS-XXXX
            now_utc = datetime.now(timezone.utc)
            random_suffix = uuid.uuid4().hex[:6].upper()
            event_id = f"EVT-{now_utc.strftime('%Y%m%d-%H%M%S')}-{random_suffix}"

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        return cls(
            event_id=event_id,
            event_type=event_type.upper(),
            class_name=class_name.lower(),
            confidence=round(float(confidence), 4),
            severity=severity.lower(),
            latitude=round(float(latitude), 6),
            longitude=round(float(longitude), 6),
            timestamp=timestamp,
            source_id=source_id,
            detection_mode=detection_mode,
            gps_mode=gps_mode,
            frame_index=int(frame_index),
            bbox=[round(float(x), 2) for x in bbox]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert event object to dictionary."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UrbanEvent":
        """Reconstruct event object from dictionary."""
        bbox_raw = data.get("bbox", [0.0, 0.0, 0.0, 0.0])
        if isinstance(bbox_raw, str):
            try:
                bbox_list = json.loads(bbox_raw)
            except Exception:
                bbox_list = [0.0, 0.0, 0.0, 0.0]
        else:
            bbox_list = list(bbox_raw)

        return cls(
            event_id=str(data["event_id"]),
            event_type=str(data.get("event_type", "UNKNOWN")).upper(),
            class_name=str(data.get("class_name", "unknown")).lower(),
            confidence=float(data.get("confidence", 0.0)),
            severity=str(data.get("severity", "none")).lower(),
            latitude=float(data.get("latitude", 0.0)),
            longitude=float(data.get("longitude", 0.0)),
            timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            source_id=str(data.get("source_id", "BUS_DEMO_01")),
            detection_mode=str(data.get("detection_mode", "REAL_AI")),
            gps_mode=str(data.get("gps_mode", "SIMULATED")),
            frame_index=int(data.get("frame_index", 0)),
            bbox=bbox_list
        )
