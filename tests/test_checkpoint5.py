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

"""
Checkpoint 5 Test Suite: GIS Component & Folium Map Module
Validates interactive map generation, marker styling, simulated GPS disclosures,
and comprehensive metadata popups for urban transit intelligence.
"""
from src.maps.folium_map import (
    get_marker_styling,
    create_event_popup_html,
    create_event_tooltip,
    add_simulated_gps_watermark,
    add_route_corridor,
    create_event_map,
    render_folium_map
)
import folium

class TestRoadHealthCheckpoint5(unittest.TestCase):
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



class TestGISCheckpoint5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent.parent
        cls.geo_tagger = GeoTagger()

        # Sample test events
        cls.road_damage_high = UrbanEvent.create(
            event_type="ROAD_DAMAGE",
            class_name="pothole",
            confidence=0.925,
            latitude=28.6155,
            longitude=77.2115,
            frame_index=15,
            bbox=[100.0, 150.0, 300.0, 350.0],
            severity="high",
            timestamp="2026-08-28T07:15:30Z",
            source_id="BUS_DEMO_01",
            detection_mode="REAL_AI",
            gps_mode="SIMULATED",
            event_id="EVT-20260828-071530-POT001"
        )

        cls.road_damage_med = UrbanEvent.create(
            event_type="ROAD_DAMAGE",
            class_name="crack",
            confidence=0.745,
            latitude=28.6202,
            longitude=77.2172,
            frame_index=30,
            bbox=[50.0, 50.0, 150.0, 150.0],
            severity="medium",
            timestamp="2026-08-28T07:15:45Z",
            source_id="BUS_DEMO_01",
            detection_mode="DEMO_SIMULATION",
            gps_mode="SIMULATED",
            event_id="EVT-20260828-071545-CRK002"
        )

        cls.vehicle_bus = UrbanEvent.create(
            event_type="VEHICLE",
            class_name="bus",
            confidence=0.880,
            latitude=28.6258,
            longitude=77.2241,
            frame_index=45,
            bbox=[200.0, 100.0, 500.0, 400.0],
            severity="none",
            timestamp="2026-08-28T07:16:00Z",
            source_id="BUS_DEMO_01",
            detection_mode="REAL_AI",
            gps_mode="SIMULATED",
            event_id="EVT-20260828-071600-BUS003"
        )

    def test_01_gis_packages_importable(self):
        """Verify Folium and streamlit-folium import properly."""
        import folium
        import streamlit_folium
        self.assertIsNotNone(folium.__version__)
        self.assertTrue(hasattr(streamlit_folium, "st_folium"))

    def test_02_marker_styling_road_damage(self):
        """Verify marker styling varies correctly with road damage severity."""
        high_style = get_marker_styling(self.road_damage_high)
        self.assertEqual(high_style["color"], "red")
        self.assertEqual(high_style["icon"], "exclamation-triangle")

        med_style = get_marker_styling(self.road_damage_med)
        self.assertEqual(med_style["color"], "orange")

        low_evt = UrbanEvent.create("ROAD_DAMAGE", "surface_wear", 0.60, 28.63, 77.23, 10, [0, 0, 10, 10], severity="low")
        low_style = get_marker_styling(low_evt)
        self.assertEqual(low_style["color"], "beige")

    def test_03_marker_styling_vehicles(self):
        """Verify vehicle classes receive distinct colors/icons."""
        bus_style = get_marker_styling(self.vehicle_bus)
        self.assertEqual(bus_style["color"], "purple")
        self.assertEqual(bus_style["icon"], "bus")

        car_evt = UrbanEvent.create("VEHICLE", "car", 0.9, 28.6, 77.2, 1, [0, 0, 10, 10])
        car_style = get_marker_styling(car_evt)
        self.assertEqual(car_style["color"], "blue")
        self.assertEqual(car_style["icon"], "car")

        moto_evt = UrbanEvent.create("VEHICLE", "motorcycle", 0.85, 28.6, 77.2, 1, [0, 0, 10, 10])
        moto_style = get_marker_styling(moto_evt)
        self.assertEqual(moto_style["color"], "cadetblue")

    def test_04_popup_html_contains_all_9_required_fields(self):
        """
        Verify that popup HTML contains all 9 required specifications:
        1. Event marker context / ID
        2. Road-damage / vehicle marker distinction & class
        3. Severity information
        4. Latitude / Longitude coordinates
        5. Event type
        6. Confidence
        7. Timestamp
        8. Detection mode
        9. GPS mode (SIMULATED)
        """
        popup = create_event_popup_html(self.road_damage_high)

        # 1. Event marker / ID
        self.assertIn("EVT-20260828-071530-POT001", popup)
        # 2. Road damage class
        self.assertIn("POTHOLE", popup)
        # 3. Severity info
        self.assertIn("HIGH", popup)
        # 4. Latitude and longitude
        self.assertIn("28.615500", popup)
        self.assertIn("77.211500", popup)
        # 5. Event type
        self.assertIn("ROAD_DAMAGE", popup)
        # 6. Confidence
        self.assertIn("92.5%", popup)
        # 7. Timestamp
        self.assertIn("2026-08-28T07:15:30Z", popup)
        # 8. Detection mode
        self.assertIn("REAL_AI", popup)
        # 9. GPS mode explicitly SIMULATED
        self.assertIn("SIMULATED", popup)

    def test_05_popup_html_from_dict_and_demo_mode(self):
        """Verify popup generation works seamlessly from raw dictionaries with DEMO_SIMULATION mode."""
        d = self.road_damage_med.to_dict()
        popup = create_event_popup_html(d)
        self.assertIn("CRACK", popup)
        self.assertIn("MEDIUM", popup)
        self.assertIn("DEMO_SIMULATION", popup)
        self.assertIn("SIMULATED", popup)

    def test_06_tooltip_content(self):
        """Verify hover tooltip formats correctly with type, severity, and simulated flag."""
        tt_damage = create_event_tooltip(self.road_damage_high)
        self.assertIn("ROAD_DAMAGE", tt_damage)
        self.assertIn("pothole", tt_damage)
        self.assertIn("HIGH", tt_damage)
        self.assertIn("SIMULATED", tt_damage)

        tt_veh = create_event_tooltip(self.vehicle_bus)
        self.assertIn("VEHICLE", tt_veh)
        self.assertIn("bus", tt_veh)
        self.assertIn("SIMULATED", tt_veh)

    def test_07_map_generation_with_events(self):
        """Verify complete Folium map generation with multiple event markers and layer controls."""
        events = [self.road_damage_high, self.road_damage_med, self.vehicle_bus]
        m = create_event_map(events=events, show_route=True, enable_clustering=False)

        self.assertIsInstance(m, folium.Map)
        # Render map HTML representation to verify components
        html_str = m.get_root().render()
        self.assertIn("ROAD_DAMAGE", html_str)
        self.assertIn("VEHICLE", html_str)
        self.assertIn("SIMULATED", html_str)

    def test_08_map_generation_empty_events(self):
        """Verify map creation handles empty event lists without crashing."""
        m = create_event_map(events=[])
        self.assertIsInstance(m, folium.Map)
        html_str = m.get_root().render()
        self.assertIn("SIMULATED", html_str)

    def test_09_simulated_route_corridor(self):
        """Verify simulated transit route polyline and terminus waypoints are added."""
        m = folium.Map(location=[28.6139, 77.2090], zoom_start=14)
        fg = add_route_corridor(m, self.geo_tagger.waypoints, route_name="Test Corridor")
        self.assertIsInstance(fg, folium.FeatureGroup)
        m.add_child(fg)
        html_str = m.get_root().render()
        self.assertIn("Simulated Bus Route", html_str)
        self.assertIn("Route Origin", html_str)
        self.assertIn("Route Terminal", html_str)

    def test_10_simulated_gps_watermark(self):
        """Verify persistent simulated GPS disclosure banner is attached to map root."""
        m = folium.Map(location=[28.6139, 77.2090], zoom_start=14)
        add_simulated_gps_watermark(m, route_name="Corridor 7B")
        html_str = m.get_root().render()
        self.assertIn("GPS MODE: SIMULATED COORDINATES", html_str)
        self.assertIn("Corridor 7B", html_str)

    def test_11_db_events_integration(self):
        """Verify map generation using DatabaseManager records."""
        test_db_path = settings.DATA_DIR / "events" / "test_gis_events.db"
        db = DatabaseManager(db_path=str(test_db_path))
        try:
            db.clear_events()
            db.insert_events([self.road_damage_high, self.vehicle_bus])
            db_events = db.get_events()
            self.assertEqual(len(db_events), 2)

            m = create_event_map(events=db_events)
            html_str = m.get_root().render()
            self.assertIn("EVT-20260828-071530-POT001", html_str)
            self.assertIn("EVT-20260828-071600-BUS003", html_str)
        finally:
            del db
            if test_db_path.exists():
                try: test_db_path.unlink()
                except Exception: pass

    def test_12_app_syntax_and_compilation(self):
        """Verify app.py compiles cleanly with GIS module imported."""
        app_file = self.base_dir / "app.py"
        self.assertTrue(app_file.exists())
        with open(app_file, "r", encoding="utf-8") as f:
            code = f.read()
        compiled = compile(code, str(app_file), "exec")
        self.assertIsNotNone(compiled)

