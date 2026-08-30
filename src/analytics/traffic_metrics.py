"""
SIH26124: Traffic Analytics Module (Checkpoint 5)
Provides transparent metrics, aggregation, temporal analysis, density classification heuristics,
and source/route statistics using edge-detected UrbanEvent records.

NOTE & DISCLAIMER:
These analytics are derived from mobile transit camera detections and deduplicated event records.
They serve as observational proxy indicators and do not claim to be scientifically calibrated
or certified induction-loop/radar traffic measurements.
"""
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from collections import Counter
import math

from src.events.schema import UrbanEvent
from src.storage.db_manager import DatabaseManager


# Whitelist of valid vehicle classes for traffic analytics
VALID_VEHICLE_CLASSES: Set[str] = {
    "car",
    "motorcycle",
    "bus",
    "truck"
}
VEHICLE_CLASSES = VALID_VEHICLE_CLASSES

# Standard traffic density classification thresholds (events observed per minute)
DEFAULT_DENSITY_THRESHOLDS = {
    "EMPTY": 0.0,
    "LOW": 5.0,        # < 5 events/min
    "MODERATE": 15.0,  # 5 - 15 events/min
    "HIGH": 30.0,      # 15 - 30 events/min
    "CONGESTED": float("inf") # >= 30 events/min
}

TRANSPARENT_METHODOLOGY_NOTE = (
    "Heuristic density classification based on mobile camera vehicle detection event rates "
    "(events per minute). Deduplication is applied at frame level by EventGenerator. "
    "Not a scientifically calibrated induction-loop or radar volume measurement."
)

DATA_DISCLAIMER = (
    "Metrics represent vehicle detection events captured by mobile transit edge sensors "
    "and deduplicated via prototype spatial-temporal tracking. Observational proxy data only."
)


def parse_iso_timestamp(ts_str: str) -> datetime:
    """
    Parse ISO 8601 timestamp string into a timezone-aware UTC datetime.
    Supports standard ISO formats with 'Z' or offset strings.
    """
    if not ts_str:
        return datetime.now(timezone.utc)
    
    clean_str = ts_str.strip()
    if clean_str.endswith("Z"):
        clean_str = clean_str[:-1] + "+00:00"
    
    try:
        dt = datetime.fromisoformat(clean_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        # Fallback if format is non-standard
        try:
            dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)


class TrafficAnalytics:
    """
    Traffic analytics engine for computing transparent metrics from UrbanEvent datasets.
    Supports input as a list of UrbanEvent objects, dict representations, or direct DatabaseManager queries.
    """

    def __init__(
        self,
        events: Optional[List[Union[UrbanEvent, Dict[str, Any]]]] = None,
        db_manager: Optional[DatabaseManager] = None,
        density_thresholds: Optional[Dict[str, float]] = None,
        valid_vehicle_classes: Optional[Set[str]] = None
    ):
        self.db_manager = db_manager
        self.density_thresholds = density_thresholds or DEFAULT_DENSITY_THRESHOLDS.copy()
        self.valid_vehicle_classes = valid_vehicle_classes or VALID_VEHICLE_CLASSES.copy()
        self._raw_events: List[UrbanEvent] = []
        
        if events is not None:
            self.set_events(events)
        elif db_manager is not None:
            self.load_from_db(db_manager)

    def set_events(self, events: List[Union[UrbanEvent, Dict[str, Any]]]) -> None:
        """Load and normalize an in-memory list of events."""
        normalized: List[UrbanEvent] = []
        for e in events:
            if isinstance(e, UrbanEvent):
                normalized.append(e)
            elif isinstance(e, dict):
                normalized.append(UrbanEvent.from_dict(e))
        self._raw_events = normalized

    def load_from_db(self, db_manager: Optional[DatabaseManager] = None, limit: int = 100000) -> None:
        """Fetch events directly from SQLite database."""
        mgr = db_manager or self.db_manager
        if mgr is None:
            raise ValueError("No DatabaseManager provided to load events from.")
        self._raw_events = mgr.get_events(limit=limit)

    @property
    def all_events(self) -> List[UrbanEvent]:
        """Return all loaded raw events."""
        return self._raw_events

    def get_vehicle_events(
        self,
        source_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
        detection_mode: Optional[str] = None,
        class_name: Optional[str] = None
    ) -> List[UrbanEvent]:
        """
        Extract only valid VEHICLE events matching optional filter criteria.
        Filters out non-vehicle records (such as ROAD_DAMAGE or pedestrian/person records).
        """
        filtered = []
        for e in self._raw_events:
            if e.event_type.upper() != "VEHICLE":
                continue
            if e.class_name.lower() not in self.valid_vehicle_classes:
                continue
            if source_id is not None and e.source_id != source_id:
                continue
            if min_confidence is not None and e.confidence < min_confidence:
                continue
            if detection_mode is not None and e.detection_mode != detection_mode:
                continue
            if class_name is not None and e.class_name.lower() != class_name.lower():
                continue
            filtered.append(e)
        return filtered

    # -------------------------------------------------------------------------
    # 1. Vehicle Counts
    # -------------------------------------------------------------------------
    def get_total_vehicle_count(
        self,
        source_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
        detection_mode: Optional[str] = None
    ) -> int:
        """Compute the total number of vehicle detection events."""
        veh_events = self.get_vehicle_events(
            source_id=source_id,
            min_confidence=min_confidence,
            detection_mode=detection_mode
        )
        return len(veh_events)

    def get_unique_detection_frames_count(
        self,
        source_id: Optional[str] = None
    ) -> int:
        """Count distinct video frame indices where vehicles were detected."""
        veh_events = self.get_vehicle_events(source_id=source_id)
        unique_frames = {e.frame_index for e in veh_events}
        return len(unique_frames)

    # -------------------------------------------------------------------------
    # 2. Counts by Vehicle Class
    # -------------------------------------------------------------------------
    def get_vehicle_counts_by_class(
        self,
        source_id: Optional[str] = None,
        min_confidence: Optional[float] = None
    ) -> Dict[str, int]:
        """
        Compute frequency counts partitioned by vehicle class (e.g. car, bus, truck, motorcycle).
        Returns a sorted dictionary mapping class name to count.
        """
        veh_events = self.get_vehicle_events(source_id=source_id, min_confidence=min_confidence)
        counts = Counter(e.class_name.lower() for e in veh_events)
        # Return sorted by count descending
        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

    def get_vehicle_class_distribution(
        self,
        source_id: Optional[str] = None,
        min_confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compute full class breakdown including absolute counts, percentage share, and dominant class.
        """
        counts = self.get_vehicle_counts_by_class(source_id=source_id, min_confidence=min_confidence)
        total = sum(counts.values())

        percentages = {}
        for cname, count in counts.items():
            pct = (count / total * 100.0) if total > 0 else 0.0
            percentages[cname] = round(pct, 2)

        dominant_class = max(counts, key=counts.get) if counts else None

        return {
            "total_vehicle_events": total,
            "counts_by_class": counts,
            "percentages_by_class": percentages,
            "dominant_class": dominant_class
        }

    # -------------------------------------------------------------------------
    # 3. Vehicle-Event Counts Over Time
    # -------------------------------------------------------------------------
    def get_vehicle_counts_over_time(
        self,
        interval_seconds: int = 60,
        source_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Aggregate vehicle events into discrete chronological time buckets.
        Calculates time duration, event rates (per sec, per min, per hr), and peak intervals.
        """
        veh_events = self.get_vehicle_events(source_id=source_id)
        if not veh_events:
            return {
                "total_events": 0,
                "interval_seconds": interval_seconds,
                "start_time": None,
                "end_time": None,
                "duration_seconds": 0.0,
                "events_per_second": 0.0,
                "events_per_minute": 0.0,
                "events_per_hour": 0.0,
                "time_buckets": [],
                "peak_bucket": None,
                "peak_count": 0
            }

        if interval_seconds <= 0:
            interval_seconds = 60

        # Sort chronologically
        sorted_events = sorted(veh_events, key=lambda e: parse_iso_timestamp(e.timestamp))
        dts = [parse_iso_timestamp(e.timestamp) for e in sorted_events]
        start_dt = dts[0]
        end_dt = dts[-1]
        
        duration_seconds = max((end_dt - start_dt).total_seconds(), 0.0)
        total_count = len(sorted_events)

        # Rate calculations
        if duration_seconds > 0:
            events_per_sec = total_count / duration_seconds
            events_per_min = events_per_sec * 60.0
            events_per_hr = events_per_min * 60.0
        else:
            events_per_sec = 0.0
            events_per_min = 0.0
            events_per_hr = 0.0

        # Bucket events
        buckets: Dict[str, Dict[str, Any]] = {}
        for evt, dt in zip(sorted_events, dts):
            offset_sec = (dt - start_dt).total_seconds()
            bucket_idx = int(offset_sec // interval_seconds)
            bucket_start_ts = datetime.fromtimestamp(
                start_dt.timestamp() + (bucket_idx * interval_seconds),
                tz=timezone.utc
            ).isoformat()

            if bucket_start_ts not in buckets:
                buckets[bucket_start_ts] = {
                    "timestamp": bucket_start_ts,
                    "count": 0,
                    "by_class": Counter(),
                    "frame_indices": []
                }

            buckets[bucket_start_ts]["count"] += 1
            buckets[bucket_start_ts]["by_class"][evt.class_name.lower()] += 1
            buckets[bucket_start_ts]["frame_indices"].append(evt.frame_index)

        # Format bucket list
        time_buckets_list = []
        peak_bucket_ts = None
        peak_count = 0

        for b_ts, b_data in buckets.items():
            b_item = {
                "timestamp": b_ts,
                "count": b_data["count"],
                "by_class": dict(b_data["by_class"]),
                "min_frame": min(b_data["frame_indices"]),
                "max_frame": max(b_data["frame_indices"])
            }
            time_buckets_list.append(b_item)
            if b_data["count"] > peak_count:
                peak_count = b_data["count"]
                peak_bucket_ts = b_ts

        return {
            "total_events": total_count,
            "interval_seconds": interval_seconds,
            "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
            "duration_seconds": round(duration_seconds, 2),
            "events_per_second": round(events_per_sec, 4),
            "events_per_minute": round(events_per_min, 2),
            "events_per_hour": round(events_per_hr, 2),
            "time_buckets": time_buckets_list,
            "peak_bucket": peak_bucket_ts,
            "peak_count": peak_count
        }

    # -------------------------------------------------------------------------
    # 4. Basic Traffic Density Classification (Transparent Heuristic)
    # -------------------------------------------------------------------------
    def classify_traffic_density(
        self,
        time_window_seconds: Optional[float] = None,
        source_id: Optional[str] = None,
        thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Classify traffic density based on transparent observed detection rate (events per minute).
        
        Classification levels:
        - "EMPTY": 0 valid vehicle events observed
        - "LOW": 0 < rate < 5.0 events/min (or instantaneous single detections)
        - "MODERATE": 5.0 <= rate < 15.0 events/min
        - "HIGH": 15.0 <= rate < 30.0 events/min
        - "CONGESTED": rate >= 30.0 events/min
        """
        thresh = thresholds or self.density_thresholds
        veh_events = self.get_vehicle_events(source_id=source_id)
        total_events = len(veh_events)

        if total_events == 0:
            return {
                "density_level": "EMPTY",
                "observed_events": 0,
                "observed_duration_seconds": 0.0,
                "observed_rate_per_minute": 0.0,
                "thresholds": thresh,
                "methodology": TRANSPARENT_METHODOLOGY_NOTE,
                "scientific_validation": False,
                "disclaimer": DATA_DISCLAIMER
            }

        # Calculate observation duration
        if time_window_seconds is not None and time_window_seconds > 0:
            duration_sec = time_window_seconds
            rate_per_minute = (total_events / duration_sec) * 60.0
        else:
            dts = [parse_iso_timestamp(e.timestamp) for e in veh_events]
            raw_duration = (max(dts) - min(dts)).total_seconds()
            duration_sec = raw_duration
            if duration_sec > 0:
                rate_per_minute = (total_events / duration_sec) * 60.0
            else:
                rate_per_minute = 0.0

        # Density classification
        low_limit = thresh.get("LOW", 5.0)
        mod_limit = thresh.get("MODERATE", 15.0)
        high_limit = thresh.get("HIGH", 30.0)

        if rate_per_minute < low_limit:
            density_level = "LOW"
        elif rate_per_minute < mod_limit:
            density_level = "MODERATE"
        elif rate_per_minute < high_limit:
            density_level = "HIGH"
        else:
            density_level = "CONGESTED"

        return {
            "density_level": density_level,
            "observed_events": total_events,
            "observed_duration_seconds": round(duration_sec, 2),
            "observed_rate_per_minute": round(rate_per_minute, 2),
            "thresholds": {
                "LOW": f"< {low_limit} ev/min",
                "MODERATE": f"{low_limit} - {mod_limit} ev/min",
                "HIGH": f"{mod_limit} - {high_limit} ev/min",
                "CONGESTED": f">= {high_limit} ev/min"
            },
            "methodology": TRANSPARENT_METHODOLOGY_NOTE,
            "scientific_validation": False,
            "disclaimer": DATA_DISCLAIMER
        }

    # -------------------------------------------------------------------------
    # 5. Route / Source-Level Statistics
    # -------------------------------------------------------------------------
    def get_source_level_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Group and compute statistics for each unique source_id (e.g. transit bus / sensor feed).
        Includes vehicle count, class distribution, time bounds, geographic bounding box,
        and detection mode breakdown.
        """
        sources = {e.source_id for e in self._raw_events}
        if not sources and not self._raw_events:
            return {}

        result: Dict[str, Dict[str, Any]] = {}

        for src in sorted(sources):
            src_all_events = [e for e in self._raw_events if e.source_id == src]
            src_veh_events = self.get_vehicle_events(source_id=src)
            
            # Class counts (only valid vehicle classes)
            class_counts = Counter(e.class_name.lower() for e in src_veh_events)
            
            # Time range
            if src_all_events:
                dts = [parse_iso_timestamp(e.timestamp) for e in src_all_events]
                first_seen = min(dts).isoformat()
                last_seen = max(dts).isoformat()
                duration_sec = max((max(dts) - min(dts)).total_seconds(), 0.0)
            else:
                first_seen = None
                last_seen = None
                duration_sec = 0.0

            # Geo bounding box
            lats = [e.latitude for e in src_all_events if not math.isnan(e.latitude)]
            lons = [e.longitude for e in src_all_events if not math.isnan(e.longitude)]
            
            if lats and lons:
                geo_bounds = {
                    "min_latitude": round(min(lats), 6),
                    "max_latitude": round(max(lats), 6),
                    "min_longitude": round(min(lons), 6),
                    "max_longitude": round(max(lons), 6),
                    "center_latitude": round(sum(lats) / len(lats), 6),
                    "center_longitude": round(sum(lons) / len(lons), 6)
                }
            else:
                geo_bounds = None

            # Modes breakdown
            modes = Counter(e.detection_mode for e in src_all_events)
            gps_modes = Counter(e.gps_mode for e in src_all_events)

            # Average confidence of valid vehicles
            confidences = [e.confidence for e in src_veh_events]
            avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

            # Density classification for this source
            source_density = self.classify_traffic_density(source_id=src)

            result[src] = {
                "source_id": src,
                "total_events_recorded": len(src_all_events),
                "total_vehicle_events": len(src_veh_events),
                "vehicle_counts_by_class": dict(class_counts),
                "first_seen_timestamp": first_seen,
                "last_seen_timestamp": last_seen,
                "duration_seconds": round(duration_sec, 2),
                "geographic_bounds": geo_bounds,
                "average_vehicle_confidence": avg_conf,
                "detection_modes": dict(modes),
                "gps_modes": dict(gps_modes),
                "traffic_density": source_density["density_level"],
                "observed_rate_per_minute": source_density["observed_rate_per_minute"]
            }

        return result

    # -------------------------------------------------------------------------
    # 6. Comprehensive Summary
    # -------------------------------------------------------------------------
    def get_summary(self, temporal_interval_sec: int = 60) -> Dict[str, Any]:
        """
        Generate a complete, structured traffic analytics report containing all metrics,
        breakdowns, density classification, source metrics, and transparency disclosures.
        """
        total_raw = len(self._raw_events)
        veh_events = self.get_vehicle_events()
        road_damage_events = [e for e in self._raw_events if e.event_type.upper() == "ROAD_DAMAGE"]
        non_vehicle_events_count = total_raw - len(veh_events)

        class_dist = self.get_vehicle_class_distribution()
        temporal = self.get_vehicle_counts_over_time(interval_seconds=temporal_interval_sec)
        density = self.classify_traffic_density()
        sources = self.get_source_level_statistics()

        modes_breakdown = Counter(e.detection_mode for e in veh_events)
        
        return {
            "analytics_generated_at": datetime.now(timezone.utc).isoformat(),
            "total_events_in_dataset": total_raw,
            "vehicle_events_analyzed": len(veh_events),
            "road_damage_events_excluded": len(road_damage_events),
            "non_vehicle_events_excluded": non_vehicle_events_count,
            "vehicle_counts": {
                "total_vehicle_count": len(veh_events),
                "unique_detection_frames": self.get_unique_detection_frames_count(),
                "by_class": class_dist["counts_by_class"],
                "class_percentage_shares": class_dist["percentages_by_class"],
                "dominant_vehicle_class": class_dist["dominant_class"]
            },
            "temporal_distribution": temporal,
            "traffic_density_classification": density,
            "source_level_statistics": sources,
            "data_transparency": {
                "detection_modes": dict(modes_breakdown),
                "gps_mode": "SIMULATED",
                "valid_vehicle_classes": sorted(list(self.valid_vehicle_classes)),
                "is_scientifically_calibrated": False,
                "methodology_note": TRANSPARENT_METHODOLOGY_NOTE,
                "disclaimer": DATA_DISCLAIMER
            }
        }


# -------------------------------------------------------------------------
# Standalone Functional API for Convenience
# -------------------------------------------------------------------------

def compute_traffic_metrics(
    events: Optional[List[Union[UrbanEvent, Dict[str, Any]]]] = None,
    db_manager: Optional[DatabaseManager] = None,
    interval_seconds: int = 60
) -> Dict[str, Any]:
    """
    Convenience wrapper to compute full traffic analytics from an event list or DatabaseManager.
    """
    analytics = TrafficAnalytics(events=events, db_manager=db_manager)
    return analytics.get_summary(temporal_interval_sec=interval_seconds)


def compute_vehicle_counts(
    events: List[Union[UrbanEvent, Dict[str, Any]]]
) -> Dict[str, Any]:
    """Compute total vehicle count and class distribution directly."""
    analytics = TrafficAnalytics(events=events)
    return analytics.get_vehicle_class_distribution()


def compute_temporal_counts(
    events: List[Union[UrbanEvent, Dict[str, Any]]],
    interval_seconds: int = 60
) -> Dict[str, Any]:
    """Compute time-series event aggregations directly."""
    analytics = TrafficAnalytics(events=events)
    return analytics.get_vehicle_counts_over_time(interval_seconds=interval_seconds)


def classify_traffic_density(
    events: List[Union[UrbanEvent, Dict[str, Any]]],
    time_window_seconds: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Classify traffic density level for given events."""
    analytics = TrafficAnalytics(events=events, density_thresholds=thresholds)
    return analytics.classify_traffic_density(time_window_seconds=time_window_seconds)


def compute_source_statistics(
    events: List[Union[UrbanEvent, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """Compute per-source/route statistics directly."""
    analytics = TrafficAnalytics(events=events)
    return analytics.get_source_level_statistics()


# Alias for backward compatibility / nomenclature flexibility
TrafficMetricsCalculator = TrafficAnalytics
