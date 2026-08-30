"""
Checkpoint 5 Test Suite: GIS Component & Folium Map Module
Validates interactive map generation, marker styling, simulated GPS disclosures,
and comprehensive metadata popups for urban transit intelligence.
"""
import unittest
import json
from pathlib import Path
from src.events.schema import UrbanEvent
from src.events.geo_tagger import GeoTagger
from src.storage.db_manager import DatabaseManager
from src.maps.folium_map import (
    get_marker_styling,
    create_event_popup_html,
    create_event_tooltip,
    add_simulated_gps_watermark,
    add_route_corridor,
    create_event_map,
    render_folium_map
)
from config import settings
import folium


class TestCheckpoint5(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
