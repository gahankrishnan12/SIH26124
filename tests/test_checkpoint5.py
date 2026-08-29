"""
Checkpoint 5 Test Suite: Road Health & Maintenance Priority Decision-Support Heuristic
"""
import unittest
import os
import json
import csv
import gc
from pathlib import Path
from typing import List

from src.events.schema import UrbanEvent
from src.events.geo_tagger import GeoTagger
from src.storage.db_manager import DatabaseManager
from src.analytics.road_health import (
    RoadHealthAnalyzer,
    RoadSegment,
    SegmentHealthSummary,
    RoadHealthReport,
    PROTOTYPE_DISCLAIMER
)
from config import settings


class TestCheckpoint5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_path = settings.DATA_DIR / "events" / "test_road_health.db"
        cls.db = DatabaseManager(db_path=str(cls.test_db_path))
        cls.analyzer = RoadHealthAnalyzer()

    @classmethod
    def tearDownClass(cls):
        del cls.db
        gc.collect()
        try:
            if cls.test_db_path.exists():
                cls.test_db_path.unlink()
        except PermissionError:
            pass

    def setUp(self):
        self.db.clear_events()

    def test_01_mandated_disclaimer_presence(self):
        """Verify the exact required prototype disclaimer string is defined and exported."""
        expected_disclaimer = "This is a prototype decision-support heuristic and is not an official government road-maintenance standard."
        self.assertEqual(PROTOTYPE_DISCLAIMER, expected_disclaimer)
        
        # Also verify report and segment summaries embed it
        report = self.analyzer.analyze_events([])
        self.assertEqual(report.disclaimer, expected_disclaimer)
        self.assertGreater(len(report.segments), 0)
        self.assertEqual(report.segments[0].disclaimer, expected_disclaimer)

    def test_02_road_segments_initialization(self):
        """Verify road segments are correctly constructed from simulated route waypoints."""
        segments = self.analyzer.segments
        self.assertEqual(len(segments), len(self.analyzer.waypoints) - 1)
        
        # Verify first segment structure
        seg0 = segments[0]
        self.assertEqual(seg0.segment_id, "SEG-00")
        self.assertIn("Sector 4 Junction", seg0.segment_name)
        self.assertAlmostEqual(seg0.start_latitude, 28.6139, places=3)
        self.assertAlmostEqual(seg0.start_longitude, 77.2090, places=3)
        self.assertGreater(seg0.length_meters, 0.0)

    def test_03_nearest_segment_spatial_projection(self):
        """Verify events at arbitrary coordinates map to the geometrically closest road segment."""
        # Point near Sector 4 Junction (start of route)
        seg = self.analyzer.find_nearest_segment(28.6140, 77.2092)
        self.assertEqual(seg.segment_id, "SEG-00")

        # Point near Terminal (end of route)
        last_seg = self.analyzer.segments[-1]
        seg_end = self.analyzer.find_nearest_segment(28.6370, 77.2390)
        self.assertEqual(seg_end.segment_id, last_seg.segment_id)

    def test_04_health_score_bounds_and_zero_damage(self):
        """Verify health score is 100.0 with 0 damages and stays within [0.0, 100.0]."""
        # Empty damage list
        score, sev_score, sev_breakdown, dominant_sev = self.analyzer.calculate_health_score([])
        self.assertEqual(score, 100.0)
        self.assertEqual(sev_score, 0.0)
        self.assertEqual(dominant_sev, "none")
        self.assertEqual(sev_breakdown["low"], 0)
        self.assertEqual(sev_breakdown["medium"], 0)
        self.assertEqual(sev_breakdown["high"], 0)

        # Extreme damage list should not drop below 0.0
        many_damages = [
            UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.9, 28.614, 77.209, i, [0, 0, 10, 10], severity="high")
            for i in range(20)
        ]
        score_extreme, _, _, _ = self.analyzer.calculate_health_score(many_damages, recurrence_clusters=5)
        self.assertEqual(score_extreme, 0.0)

    def test_05_severity_weighting_impact(self):
        """Verify higher severity damages cause proportionally greater health score reductions."""
        low_evt = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.8, 28.614, 77.209, 1, [0, 0, 10, 10], severity="low")
        med_evt = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.8, 28.614, 77.209, 2, [0, 0, 10, 10], severity="medium")
        high_evt = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.8, 28.614, 77.209, 3, [0, 0, 10, 10], severity="high")

        score_low, _, _, dom_low = self.analyzer.calculate_health_score([low_evt])
        score_med, _, _, dom_med = self.analyzer.calculate_health_score([med_evt])
        score_high, _, _, dom_high = self.analyzer.calculate_health_score([high_evt])

        self.assertEqual(dom_low, "low")
        self.assertEqual(dom_med, "medium")
        self.assertEqual(dom_high, "high")

        self.assertGreater(score_low, score_med)
        self.assertGreater(score_med, score_high)

    def test_06_recurrence_clustering(self):
        """Verify damage events clustered within proximity threshold increase recurrence and penalty."""
        # Two events 5 meters apart
        evt1 = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.8, 28.61400, 77.20900, 1, [0, 0, 10, 10], severity="medium")
        evt2 = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.8, 28.61403, 77.20903, 2, [0, 0, 10, 10], severity="medium")

        # Two events 500 meters apart (separate clusters)
        evt3 = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.8, 28.62000, 77.21500, 3, [0, 0, 10, 10], severity="medium")

        rec_clustered = self.analyzer.detect_recurrence_clusters([evt1, evt2])
        rec_separated = self.analyzer.detect_recurrence_clusters([evt1, evt3])

        self.assertEqual(rec_clustered, 1)
        self.assertEqual(rec_separated, 0)

        # Verify recurring cluster penalizes health score more
        score_single, _, _, _ = self.analyzer.calculate_health_score([evt1], recurrence_clusters=0)
        score_recurrent, _, _, _ = self.analyzer.calculate_health_score([evt1, evt2], recurrence_clusters=1)
        self.assertGreater(score_single, score_recurrent)

    def test_07_traffic_exposure_handling(self):
        """Verify traffic exposure reflects actual vehicle counts and does not fabricate data when missing."""
        dam_evt = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.8, 28.614, 77.209, 1, [0, 0, 10, 10], severity="medium")
        veh_bus = UrbanEvent.create("VEHICLE", "bus", 0.9, 28.614, 77.209, 2, [0, 0, 10, 10])
        veh_car = UrbanEvent.create("VEHICLE", "car", 0.9, 28.614, 77.209, 3, [0, 0, 10, 10])

        # Case A: No vehicle events present at all
        report_no_traffic = self.analyzer.analyze_events([dam_evt])
        seg0_no_traffic = report_no_traffic.segments[0]
        self.assertIsNone(seg0_no_traffic.traffic_exposure)

        # Case B: Vehicle events present
        report_with_traffic = self.analyzer.analyze_events([dam_evt, veh_bus, veh_car])
        seg0_traffic = report_with_traffic.segments[0]
        self.assertIsNotNone(seg0_traffic.traffic_exposure)
        self.assertEqual(seg0_traffic.traffic_exposure["vehicle_count"], 2)
        self.assertEqual(seg0_traffic.traffic_exposure["heavy_vehicle_count"], 1)
        self.assertEqual(seg0_traffic.traffic_exposure["vehicle_breakdown"]["bus"], 1)
        self.assertEqual(seg0_traffic.traffic_exposure["vehicle_breakdown"]["car"], 1)
        self.assertTrue(seg0_traffic.traffic_exposure["is_measured"])

    def test_08_maintenance_priority_tiers(self):
        """Verify maintenance priority categorization based on health score and severity."""
        # 1. Normal: 0 damage
        prio_norm, tier_norm = self.analyzer.calculate_maintenance_priority(100.0, 0, {"low": 0, "medium": 0, "high": 0})
        self.assertEqual(prio_norm, 0.0)
        self.assertEqual(tier_norm, "NORMAL")

        # 2. Low: minor wear
        prio_low, tier_low = self.analyzer.calculate_maintenance_priority(90.0, 1, {"low": 1, "medium": 0, "high": 0})
        self.assertEqual(tier_low, "LOW")
        self.assertGreater(prio_low, 0.0)

        # 3. Medium: moderate damage
        prio_med, tier_med = self.analyzer.calculate_maintenance_priority(70.0, 1, {"low": 0, "medium": 1, "high": 0})
        self.assertEqual(tier_med, "MEDIUM")

        # 4. High: single high severity
        prio_high, tier_high = self.analyzer.calculate_maintenance_priority(50.0, 1, {"low": 0, "medium": 0, "high": 1})
        self.assertEqual(tier_high, "HIGH")

        # 5. Critical: multiple high severity or very low health
        prio_crit, tier_crit = self.analyzer.calculate_maintenance_priority(20.0, 2, {"low": 0, "medium": 0, "high": 2})
        self.assertEqual(tier_crit, "CRITICAL")

    def test_09_all_required_segment_fields_provided(self):
        """Verify each segment summary contains all six required fields."""
        dam_evt = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.85, 28.614, 77.209, 1, [0, 0, 10, 10], severity="high")
        veh_evt = UrbanEvent.create("VEHICLE", "car", 0.90, 28.614, 77.209, 2, [0, 0, 10, 10])

        report = self.analyzer.analyze_events([dam_evt, veh_evt])
        for seg in report.segments:
            # 1. segment identifier
            self.assertTrue(hasattr(seg, "segment_id"))
            self.assertTrue(isinstance(seg.segment_id, str))
            # 2. damage count
            self.assertTrue(hasattr(seg, "damage_count"))
            self.assertTrue(isinstance(seg.damage_count, int))
            # 3. severity
            self.assertTrue(hasattr(seg, "dominant_severity"))
            self.assertIn(seg.dominant_severity, ["high", "medium", "low", "none"])
            # 4. traffic exposure if available
            self.assertTrue(hasattr(seg, "traffic_exposure"))
            # 5. health score (0-100)
            self.assertTrue(hasattr(seg, "health_score"))
            self.assertGreaterEqual(seg.health_score, 0.0)
            self.assertLessEqual(seg.health_score, 100.0)
            # 6. maintenance priority
            self.assertTrue(hasattr(seg, "maintenance_priority"))
            self.assertIn(seg.maintenance_priority, ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL"])

    def test_10_database_aggregation_integration(self):
        """Verify seamless end-to-end integration with SQLite DatabaseManager."""
        evt1 = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.9, 28.614, 77.209, 1, [0, 0, 10, 10], severity="high")
        evt2 = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.7, 28.623, 77.220, 2, [0, 0, 10, 10], severity="medium")
        evt3 = UrbanEvent.create("VEHICLE", "truck", 0.88, 28.614, 77.209, 3, [0, 0, 10, 10])

        self.db.insert_events([evt1, evt2, evt3])

        report = self.analyzer.analyze_database(db_manager=self.db)
        self.assertEqual(report.total_damage_events, 2)
        self.assertEqual(report.total_vehicle_events, 1)
        self.assertLess(report.overall_network_health, 100.0)
        self.assertGreater(report.critical_segments_count + report.high_priority_segments_count + report.medium_priority_segments_count, 0)

    def test_11_report_exports(self):
        """Verify JSON and CSV export capabilities."""
        evt = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.9, 28.614, 77.209, 1, [0, 0, 10, 10], severity="high")
        report = self.analyzer.analyze_events([evt])

        json_file = settings.EVENTS_DATA_DIR / "test_road_health_report.json"
        csv_file = settings.EVENTS_DATA_DIR / "test_road_health_segments.csv"

        out_json = self.analyzer.export_report_json(report, str(json_file))
        out_csv = self.analyzer.export_report_csv(report, str(csv_file))

        self.assertTrue(Path(out_json).exists())
        self.assertTrue(Path(out_csv).exists())

        # Verify JSON content
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("overall_network_health", data)
            self.assertEqual(data["disclaimer"], PROTOTYPE_DISCLAIMER)

        # Verify CSV content
        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), len(self.analyzer.segments))
            self.assertEqual(rows[0]["disclaimer"], PROTOTYPE_DISCLAIMER)

        # Cleanup
        if json_file.exists():
            try: json_file.unlink()
            except Exception: pass
        if csv_file.exists():
            try: csv_file.unlink()
            except Exception: pass


if __name__ == "__main__":
    unittest.main()
