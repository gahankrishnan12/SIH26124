"""
SIH26124: Road Health & Maintenance Priority Analytics Module
Provides segment-level aggregation, transparent deterministic scoring,
and priority heuristics for municipal road maintenance decision support.

DISCLAIMER:
"This is a prototype decision-support heuristic and is not an official government road-maintenance standard."
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import math
import json
import csv
from pathlib import Path
from datetime import datetime, timezone

from src.events.schema import UrbanEvent
from src.events.geo_tagger import GeoTagger
from src.storage.db_manager import DatabaseManager
from config import settings

# Explicit Mandated Prototype Disclaimer
PROTOTYPE_DISCLAIMER = "This is a prototype decision-support heuristic and is not an official government road-maintenance standard."


@dataclass
class RoadSegment:
    """
    Represents a discrete physical road segment between two transit waypoints.
    """
    segment_id: str
    segment_name: str
    start_latitude: float
    start_longitude: float
    end_latitude: float
    end_longitude: float
    length_meters: float
    speed_limit_kmh: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrafficExposure:
    """
    Represents real measured vehicle traffic observed on a road segment.
    """
    vehicle_count: int
    vehicle_breakdown: Dict[str, int]
    heavy_vehicle_count: int  # Buses, trucks
    exposure_level: str       # "NONE", "LOW", "MEDIUM", "HIGH"
    is_measured: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentHealthSummary:
    """
    Structured segment-level aggregation output containing road damage metrics,
    traffic exposure, prototype health score (0-100), and maintenance priority.
    """
    segment_id: str
    segment_name: str
    damage_count: int
    severity_breakdown: Dict[str, int]
    dominant_severity: str
    severity_score: float
    recurrence_count: int
    traffic_exposure: Optional[Dict[str, Any]]
    health_score: float
    maintenance_priority: str
    priority_score: float
    start_coord: Tuple[float, float]
    end_coord: Tuple[float, float]
    length_meters: float
    event_ids: List[str] = field(default_factory=list)
    disclaimer: str = PROTOTYPE_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["start_coord"] = list(self.start_coord)
        d["end_coord"] = list(self.end_coord)
        return d


@dataclass
class RoadHealthReport:
    """
    Aggregated network-level analytics report across all route segments.
    """
    overall_network_health: float
    total_segments: int
    critical_segments_count: int
    high_priority_segments_count: int
    medium_priority_segments_count: int
    low_priority_segments_count: int
    normal_segments_count: int
    total_damage_events: int
    total_vehicle_events: int
    segments: List[SegmentHealthSummary]
    scoring_parameters: Dict[str, Any]
    timestamp: str
    disclaimer: str = PROTOTYPE_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_network_health": self.overall_network_health,
            "total_segments": self.total_segments,
            "critical_segments_count": self.critical_segments_count,
            "high_priority_segments_count": self.high_priority_segments_count,
            "medium_priority_segments_count": self.medium_priority_segments_count,
            "low_priority_segments_count": self.low_priority_segments_count,
            "normal_segments_count": self.normal_segments_count,
            "total_damage_events": self.total_damage_events,
            "total_vehicle_events": self.total_vehicle_events,
            "scoring_parameters": self.scoring_parameters,
            "timestamp": self.timestamp,
            "disclaimer": self.disclaimer,
            "segments": [s.to_dict() for s in self.segments]
        }


class RoadHealthAnalyzer:
    """
    Deterministic decision-support heuristic engine for computing road health scores
    and maintenance priority rankings based on physical damage events, spatial recurrence,
    and observed traffic exposure.
    
    Formula & Scoring Principles:
    -----------------------------
    1. Base Damage Severity Weighting:
       - 'low': 1.0
       - 'medium': 2.5
       - 'high': 5.0
    2. Recurrence Factor:
       - Damage events within close proximity (<= 30 meters) indicate recurring / concentrated failure.
       - Recurrence factor = 1.0 + (recurrence_clusters * 0.5), capped at 2.0.
    3. Health Score (0–100):
       - Baseline pristine road = 100.0
       - Damage Penalty = sum(severity_weights) * recurrence_factor * scale_factor
       - Health Score = max(0.0, min(100.0, 100.0 - Damage Penalty))
    4. Traffic Exposure Factor:
       - Observed vehicle count adds maintenance urgency without fabricating synthetic traffic.
       - Traffic Multiplier = 1.0 + min(vehicle_count / 20.0, 1.0) * 0.5 (if vehicles observed, else 1.0)
    5. Maintenance Priority (0–100 score & category):
       - Priority Score = min(100.0, (100.0 - Health Score) * Traffic Multiplier)
       - Category: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NORMAL'
    """

    # Transparent Constant Parameters
    SEVERITY_WEIGHTS = {
        "low": 1.0,
        "medium": 2.5,
        "high": 5.0,
        "none": 0.0
    }
    DAMAGE_SCALE_FACTOR = 8.0     # 1 High damage (5.0) -> 40 pt penalty; 2 High -> 80 pt penalty
    RECURRENCE_CLUSTER_DIST_M = 30.0  # Meters threshold for spatial recurrence clustering
    RECURRENCE_BOOST_PER_CLUSTER = 0.5
    MAX_RECURRENCE_MULTIPLIER = 2.0
    TRAFFIC_IMPACT_MAX_BOOST = 0.5    # Up to +50% priority score multiplier under heavy traffic

    def __init__(
        self,
        route_file_path: Optional[str] = None,
        custom_waypoints: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize the analyzer with route waypoints.
        """
        self.geo_tagger = GeoTagger(route_file_path=route_file_path)
        if custom_waypoints and len(custom_waypoints) >= 2:
            self.waypoints = custom_waypoints
        else:
            self.waypoints = self.geo_tagger.waypoints

        self.segments: List[RoadSegment] = self._build_road_segments(self.waypoints)

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great-circle distance between two GPS coordinates in meters.
        """
        R = 6371000.0  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return R * c

    def _build_road_segments(self, waypoints: List[Dict[str, Any]]) -> List[RoadSegment]:
        """
        Construct sequential RoadSegment objects from route waypoints.
        """
        segments: List[RoadSegment] = []
        if len(waypoints) < 2:
            return segments

        for i in range(len(waypoints) - 1):
            wp_start = waypoints[i]
            wp_end = waypoints[i + 1]

            seg_id = f"SEG-{i:02d}"
            name_start = wp_start.get("segment_name", f"Point {i}")
            name_end = wp_end.get("segment_name", f"Point {i+1}")
            seg_name = f"{name_start} to {name_end}"

            lat1 = float(wp_start["latitude"])
            lon1 = float(wp_start["longitude"])
            lat2 = float(wp_end["latitude"])
            lon2 = float(wp_end["longitude"])

            length_m = round(self.haversine_distance(lat1, lon1, lat2, lon2), 2)
            speed = wp_start.get("speed_kmh", None)

            segments.append(
                RoadSegment(
                    segment_id=seg_id,
                    segment_name=seg_name,
                    start_latitude=lat1,
                    start_longitude=lon1,
                    end_latitude=lat2,
                    end_longitude=lon2,
                    length_meters=length_m,
                    speed_limit_kmh=speed
                )
            )

        return segments

    def find_nearest_segment(self, lat: float, lon: float) -> RoadSegment:
        """
        Associate any (lat, lon) coordinate to its geometrically nearest road segment
        using orthogonal line-segment projection.
        """
        if not self.segments:
            raise ValueError("No road segments defined in analyzer.")

        best_segment = self.segments[0]
        min_distance = float("inf")

        for seg in self.segments:
            # Segment line vector in lat/lon space
            dlat = seg.end_latitude - seg.start_latitude
            dlon = seg.end_longitude - seg.start_longitude
            seg_len_sq = dlat * dlat + dlon * dlon

            if seg_len_sq == 0.0:
                proj_lat, proj_lon = seg.start_latitude, seg.start_longitude
            else:
                # Orthogonal projection factor t clamped to [0, 1]
                t = max(0.0, min(1.0, ((lat - seg.start_latitude) * dlat + (lon - seg.start_longitude) * dlon) / seg_len_sq))
                proj_lat = seg.start_latitude + t * dlat
                proj_lon = seg.start_longitude + t * dlon

            dist = self.haversine_distance(lat, lon, proj_lat, proj_lon)
            if dist < min_distance:
                min_distance = dist
                best_segment = seg

        return best_segment

    def detect_recurrence_clusters(self, damage_events: List[UrbanEvent]) -> int:
        """
        Compute the number of spatial recurrence clusters among damage events.
        Multiple damage events within RECURRENCE_CLUSTER_DIST_M of each other indicate
        concentrated recurring distress.
        """
        if len(damage_events) <= 1:
            return 0

        # Simple single-linkage spatial clustering
        clusters: List[List[UrbanEvent]] = []
        for evt in damage_events:
            added = False
            for cluster in clusters:
                # Check distance to any event in existing cluster
                if any(
                    self.haversine_distance(evt.latitude, evt.longitude, c_evt.latitude, c_evt.longitude) <= self.RECURRENCE_CLUSTER_DIST_M
                    for c_evt in cluster
                ):
                    cluster.append(evt)
                    added = True
                    break
            if not added:
                clusters.append([evt])

        # Recurrence count is the number of distinct events belonging to clusters of size >= 2
        recurrence_count = sum(len(c) - 1 for c in clusters if len(c) >= 2)
        return recurrence_count

    def calculate_health_score(
        self,
        damage_events: List[UrbanEvent],
        recurrence_clusters: int = 0
    ) -> Tuple[float, float, Dict[str, int], str]:
        """
        Compute transparent 0–100 Road Health Score.
        100.0 = pristine road (no damage)
        0.0 = critically degraded road

        Returns:
            (health_score, total_weighted_severity, severity_breakdown, dominant_severity)
        """
        severity_breakdown = {"low": 0, "medium": 0, "high": 0}
        total_weighted_severity = 0.0

        for evt in damage_events:
            sev = evt.severity.lower()
            if sev not in severity_breakdown:
                sev = "low"  # default fallback
            severity_breakdown[sev] += 1
            total_weighted_severity += self.SEVERITY_WEIGHTS.get(sev, 1.0)

        if not damage_events:
            return 100.0, 0.0, severity_breakdown, "none"

        # Dominant severity
        if severity_breakdown["high"] > 0:
            dominant_severity = "high"
        elif severity_breakdown["medium"] > 0:
            dominant_severity = "medium"
        elif severity_breakdown["low"] > 0:
            dominant_severity = "low"
        else:
            dominant_severity = "none"

        # Recurrence multiplier
        recurrence_mult = min(
            self.MAX_RECURRENCE_MULTIPLIER,
            1.0 + (recurrence_clusters * self.RECURRENCE_BOOST_PER_CLUSTER)
        )

        # Damage penalty
        penalty = (total_weighted_severity * recurrence_mult) * self.DAMAGE_SCALE_FACTOR
        health_score = max(0.0, min(100.0, round(100.0 - penalty, 2)))

        return health_score, round(total_weighted_severity, 2), severity_breakdown, dominant_severity

    def calculate_maintenance_priority(
        self,
        health_score: float,
        damage_count: int,
        severity_breakdown: Dict[str, int],
        vehicle_count: Optional[int] = None
    ) -> Tuple[float, str]:
        """
        Determine maintenance urgency score (0-100) and discrete priority tier:
        - "CRITICAL": Urgent hazard requiring immediate dispatch
        - "HIGH": Significant degradation requiring near-term maintenance
        - "MEDIUM": Moderate wear requiring scheduled maintenance
        - "LOW": Minor wear requiring routine monitoring
        - "NORMAL": Good condition, no action required
        """
        if damage_count == 0:
            return 0.0, "NORMAL"

        base_urgency = 100.0 - health_score

        # Traffic Exposure Impact (only applied if traffic was measured)
        if vehicle_count is not None and vehicle_count > 0:
            # Scale traffic boost: up to +50% if >= 20 vehicles observed
            traffic_ratio = min(1.0, vehicle_count / 20.0)
            traffic_multiplier = 1.0 + (traffic_ratio * self.TRAFFIC_IMPACT_MAX_BOOST)
        else:
            traffic_multiplier = 1.0

        priority_score = max(0.0, min(100.0, round(base_urgency * traffic_multiplier, 2)))

        # Discrete Priority Tier Logic
        high_sev = severity_breakdown.get("high", 0)
        med_sev = severity_breakdown.get("medium", 0)

        if high_sev >= 2 or priority_score >= 75.0 or health_score <= 25.0:
            tier = "CRITICAL"
        elif high_sev >= 1 or priority_score >= 50.0 or health_score <= 50.0:
            tier = "HIGH"
        elif med_sev >= 1 or priority_score >= 25.0 or health_score <= 75.0:
            tier = "MEDIUM"
        elif priority_score > 0.0 or health_score < 98.0:
            tier = "LOW"
        else:
            tier = "NORMAL"

        return priority_score, tier

    def analyze_events(self, events: List[UrbanEvent]) -> RoadHealthReport:
        """
        Perform complete segment-level aggregation and heuristic health scoring
        across all events (both ROAD_DAMAGE and VEHICLE events).
        """
        # Bucket events by nearest segment
        segment_damage_map: Dict[str, List[UrbanEvent]] = {s.segment_id: [] for s in self.segments}
        segment_vehicle_map: Dict[str, List[UrbanEvent]] = {s.segment_id: [] for s in self.segments}

        total_damage_count = 0
        total_vehicle_count = 0

        for evt in events:
            nearest_seg = self.find_nearest_segment(evt.latitude, evt.longitude)
            if evt.event_type == "ROAD_DAMAGE":
                segment_damage_map[nearest_seg.segment_id].append(evt)
                total_damage_count += 1
            elif evt.event_type == "VEHICLE":
                segment_vehicle_map[nearest_seg.segment_id].append(evt)
                total_vehicle_count += 1

        summaries: List[SegmentHealthSummary] = []
        network_health_sum = 0.0

        crit_count = 0
        high_count = 0
        med_count = 0
        low_count = 0
        norm_count = 0

        has_any_vehicle_data = (total_vehicle_count > 0)

        for seg in self.segments:
            dam_events = segment_damage_map[seg.segment_id]
            veh_events = segment_vehicle_map[seg.segment_id]

            # 1. Recurrence
            recurrence_count = self.detect_recurrence_clusters(dam_events)

            # 2. Health Score
            health_score, sev_score, sev_breakdown, dominant_sev = self.calculate_health_score(
                damage_events=dam_events,
                recurrence_clusters=recurrence_count
            )
            network_health_sum += health_score

            # 3. Traffic Exposure (only report if measured)
            if has_any_vehicle_data or len(veh_events) > 0:
                v_count = len(veh_events)
                v_breakdown: Dict[str, int] = {}
                heavy_count = 0
                for v in veh_events:
                    cname = v.class_name.lower()
                    v_breakdown[cname] = v_breakdown.get(cname, 0) + 1
                    if cname in ["bus", "truck"]:
                        heavy_count += 1

                if v_count >= 15:
                    exp_lvl = "HIGH"
                elif v_count >= 5:
                    exp_lvl = "MEDIUM"
                elif v_count > 0:
                    exp_lvl = "LOW"
                else:
                    exp_lvl = "NONE"

                traffic_exp = TrafficExposure(
                    vehicle_count=v_count,
                    vehicle_breakdown=v_breakdown,
                    heavy_vehicle_count=heavy_count,
                    exposure_level=exp_lvl,
                    is_measured=True
                ).to_dict()
                veh_count_for_prio = v_count
            else:
                traffic_exp = None
                veh_count_for_prio = None

            # 4. Maintenance Priority
            prio_score, prio_tier = self.calculate_maintenance_priority(
                health_score=health_score,
                damage_count=len(dam_events),
                severity_breakdown=sev_breakdown,
                vehicle_count=veh_count_for_prio
            )

            # Tally categories
            if prio_tier == "CRITICAL":
                crit_count += 1
            elif prio_tier == "HIGH":
                high_count += 1
            elif prio_tier == "MEDIUM":
                med_count += 1
            elif prio_tier == "LOW":
                low_count += 1
            else:
                norm_count += 1

            summary = SegmentHealthSummary(
                segment_id=seg.segment_id,
                segment_name=seg.segment_name,
                damage_count=len(dam_events),
                severity_breakdown=sev_breakdown,
                dominant_severity=dominant_sev,
                severity_score=sev_score,
                recurrence_count=recurrence_count,
                traffic_exposure=traffic_exp,
                health_score=health_score,
                maintenance_priority=prio_tier,
                priority_score=prio_score,
                start_coord=(seg.start_latitude, seg.start_longitude),
                end_coord=(seg.end_latitude, seg.end_longitude),
                length_meters=seg.length_meters,
                event_ids=[e.event_id for e in dam_events + veh_events],
                disclaimer=PROTOTYPE_DISCLAIMER
            )
            summaries.append(summary)

        total_segs = len(self.segments)
        avg_health = round(network_health_sum / max(total_segs, 1), 2) if total_segs > 0 else 100.0

        report = RoadHealthReport(
            overall_network_health=avg_health,
            total_segments=total_segs,
            critical_segments_count=crit_count,
            high_priority_segments_count=high_count,
            medium_priority_segments_count=med_count,
            low_priority_segments_count=low_count,
            normal_segments_count=norm_count,
            total_damage_events=total_damage_count,
            total_vehicle_events=total_vehicle_count,
            segments=summaries,
            scoring_parameters={
                "severity_weights": self.SEVERITY_WEIGHTS,
                "damage_scale_factor": self.DAMAGE_SCALE_FACTOR,
                "recurrence_cluster_distance_meters": self.RECURRENCE_CLUSTER_DIST_M,
                "recurrence_boost_per_cluster": self.RECURRENCE_BOOST_PER_CLUSTER,
                "traffic_impact_max_boost": self.TRAFFIC_IMPACT_MAX_BOOST
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
            disclaimer=PROTOTYPE_DISCLAIMER
        )
        return report

    def analyze_database(self, db_manager: Optional[DatabaseManager] = None) -> RoadHealthReport:
        """
        Fetch all events directly from SQLite database and execute road health analysis.
        """
        db = db_manager or DatabaseManager()
        events = db.get_events(limit=100000)
        return self.analyze_events(events)

    def export_report_json(self, report: RoadHealthReport, file_path: Optional[str] = None) -> str:
        """
        Export complete road health report to a structured JSON file.
        """
        out_path = Path(file_path) if file_path else settings.EVENTS_DATA_DIR / "road_health_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        return str(out_path)

    def export_report_csv(self, report: RoadHealthReport, file_path: Optional[str] = None) -> str:
        """
        Export segment-level health summaries to a tabular CSV file.
        """
        out_path = Path(file_path) if file_path else settings.EVENTS_DATA_DIR / "road_health_segments.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "segment_id", "segment_name", "health_score", "maintenance_priority",
            "priority_score", "damage_count", "dominant_severity", "severity_score",
            "low_severity_count", "med_severity_count", "high_severity_count",
            "recurrence_count", "traffic_vehicle_count", "traffic_exposure_level",
            "length_meters", "disclaimer"
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for seg in report.segments:
                traf_cnt = seg.traffic_exposure["vehicle_count"] if seg.traffic_exposure else "N/A"
                traf_lvl = seg.traffic_exposure["exposure_level"] if seg.traffic_exposure else "N/A"
                writer.writerow({
                    "segment_id": seg.segment_id,
                    "segment_name": seg.segment_name,
                    "health_score": seg.health_score,
                    "maintenance_priority": seg.maintenance_priority,
                    "priority_score": seg.priority_score,
                    "damage_count": seg.damage_count,
                    "dominant_severity": seg.dominant_severity,
                    "severity_score": seg.severity_score,
                    "low_severity_count": seg.severity_breakdown.get("low", 0),
                    "med_severity_count": seg.severity_breakdown.get("medium", 0),
                    "high_severity_count": seg.severity_breakdown.get("high", 0),
                    "recurrence_count": seg.recurrence_count,
                    "traffic_vehicle_count": traf_cnt,
                    "traffic_exposure_level": traf_lvl,
                    "length_meters": seg.length_meters,
                    "disclaimer": seg.disclaimer
                })

        return str(out_path)
