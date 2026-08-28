"""
SIH26124: Event Generator & Deduplication Module
Converts raw AI detections into structured timestamped events and applies prototype
spatial-temporal deduplication to prevent flooding the database with identical detections across consecutive frames.
"""
import math
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone, timedelta
from src.events.schema import UrbanEvent
from src.events.geo_tagger import GeoTagger

class EventGenerator:
    """
    Transforms raw frame-level AI detections into persistent UrbanEvent objects.
    Includes prototype spatial-temporal deduplication for consecutive frames.
    """
    def __init__(
        self,
        geo_tagger: Optional[GeoTagger] = None,
        source_id: str = "BUS_DEMO_01",
        dedup_window_frames: int = 15,
        dedup_spatial_threshold: float = 80.0 # Pixel distance threshold for centers
    ):
        self.geo_tagger = geo_tagger or GeoTagger()
        self.source_id = source_id
        self.dedup_window_frames = dedup_window_frames
        self.dedup_spatial_threshold = dedup_spatial_threshold
        
        # Tracking history for deduplication
        # List of active event records: [{"event": UrbanEvent, "last_seen_frame": int, "center": (cx, cy)}]
        self.active_tracks: List[Dict[str, Any]] = []
        self.all_generated_events: List[UrbanEvent] = []
        self.total_raw_detections_count: int = 0
        self.total_duplicates_filtered: int = 0
        self.base_timestamp = datetime.now(timezone.utc)

    @staticmethod
    def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
        """Compute (x_center, y_center) from [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        """Calculate euclidean distance between two pixel coordinates."""
        return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    def process_frame_detections(
        self,
        frame_index: int,
        total_frames: int,
        vehicle_detections: List[Dict[str, Any]],
        damage_detections: List[Dict[str, Any]],
        video_fps: float = 25.0
    ) -> List[UrbanEvent]:
        """
        Process detections for a single frame, interpolate simulated GPS,
        apply deduplication, and return newly created canonical events.
        """
        # Calculate simulated timestamp based on frame index & FPS
        seconds_offset = frame_index / max(video_fps, 1.0)
        current_time_iso = (self.base_timestamp + timedelta(seconds=seconds_offset)).isoformat()

        # Get simulated GPS for current frame
        gps_info = self.geo_tagger.get_coordinate_for_frame(frame_index, total_frames)
        lat = gps_info["latitude"]
        lon = gps_info["longitude"]

        # Clean old tracks outside sliding window
        self.active_tracks = [
            t for t in self.active_tracks 
            if (frame_index - t["last_seen_frame"]) <= self.dedup_window_frames
        ]

        new_events_this_frame: List[UrbanEvent] = []

        # 1. Process Vehicle Detections
        for det in vehicle_detections:
            self.total_raw_detections_count += 1
            evt = self._process_single_detection(
                event_type="VEHICLE",
                det=det,
                frame_index=frame_index,
                lat=lat,
                lon=lon,
                timestamp=current_time_iso,
                detection_mode="REAL_AI"
            )
            if evt is not None:
                new_events_this_frame.append(evt)

        # 2. Process Road Damage Detections
        for det in damage_detections:
            self.total_raw_detections_count += 1
            det_mode = det.get("detection_mode", "REAL_AI")
            evt = self._process_single_detection(
                event_type="ROAD_DAMAGE",
                det=det,
                frame_index=frame_index,
                lat=lat,
                lon=lon,
                timestamp=current_time_iso,
                detection_mode=det_mode
            )
            if evt is not None:
                new_events_this_frame.append(evt)

        return new_events_this_frame

    def _process_single_detection(
        self,
        event_type: str,
        det: Dict[str, Any],
        frame_index: int,
        lat: float,
        lon: float,
        timestamp: str,
        detection_mode: str
    ) -> Optional[UrbanEvent]:
        """
        Check if detection matches an existing active track.
        If matched, merge and update confidence.
        If new, create UrbanEvent and add to active tracks.
        """
        bbox = det["bbox"]
        cname = det["class_name"].lower()
        conf = float(det["confidence"])
        severity = det.get("severity", "none").lower()
        current_center = self.get_bbox_center(bbox)

        # Look for matching track of same type and class within spatial proximity
        matched_track = None
        min_dist = float("inf")

        for track in self.active_tracks:
            t_evt = track["event"]
            if t_evt.event_type == event_type and t_evt.class_name == cname:
                dist = self.euclidean_distance(current_center, track["center"])
                if dist < self.dedup_spatial_threshold and dist < min_dist:
                    min_dist = dist
                    matched_track = track

        if matched_track is not None:
            # Duplicate detection in consecutive frames -> Merge & update
            self.total_duplicates_filtered += 1
            matched_track["last_seen_frame"] = frame_index
            matched_track["center"] = current_center
            # Update to highest observed confidence
            if conf > matched_track["event"].confidence:
                matched_track["event"].confidence = round(conf, 4)
                matched_track["event"].bbox = bbox
            return None
        else:
            # Distinct new event -> Create and track
            new_event = UrbanEvent.create(
                event_type=event_type,
                class_name=cname,
                confidence=conf,
                latitude=lat,
                longitude=lon,
                frame_index=frame_index,
                bbox=bbox,
                severity=severity,
                timestamp=timestamp,
                source_id=self.source_id,
                detection_mode=detection_mode,
                gps_mode="SIMULATED"
            )
            self.active_tracks.append({
                "event": new_event,
                "last_seen_frame": frame_index,
                "center": current_center
            })
            self.all_generated_events.append(new_event)
            return new_event
