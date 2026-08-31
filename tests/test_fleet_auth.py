"""
SIH26124: Fleet Authentication & Bus Identity Test Suite
Validates prototype bus authentication, session state management, end-to-end bus_id
data flow, database schema migration, multi-bus filtering, legacy event preservation,
and traffic analytics compatibility.
"""
import unittest
import os
import json
import csv
import gc
from pathlib import Path

from config import settings
from config.buses import (
    get_available_bus_ids,
    get_available_buses,
    get_bus_info,
    format_bus_display,
    authenticate_bus,
    UNKNOWN_BUS_LABEL
)
from src.events.schema import UrbanEvent
from src.events.geo_tagger import GeoTagger
from src.events.generator import EventGenerator
from src.storage.db_manager import DatabaseManager
from src.detection.vehicle_detector import VehicleDetector
from src.detection.road_damage_detector import RoadDamageDetector
from src.video.processor import VideoProcessor
from src.analytics.traffic_metrics import TrafficAnalytics, compute_traffic_metrics


class TestFleetAuthenticationAndBusIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_path = settings.DATA_DIR / "events" / "test_fleet_auth.db"
        cls.db = DatabaseManager(db_path=str(cls.test_db_path))
        cls.geo_tagger = GeoTagger()
        cls.sample_video = settings.SAMPLE_DATA_DIR / "real_road_sample.mp4"

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

    # -------------------------------------------------------------------------
    # 1. Prototype Authentication & Config Tests
    # -------------------------------------------------------------------------
    def test_01_valid_bus_login_succeeds(self):
        """Verify that correct PIN authenticates configured demo buses."""
        self.assertTrue(authenticate_bus("BUS-001", "1001"))
        self.assertTrue(authenticate_bus("BUS-002", "1002"))
        self.assertTrue(authenticate_bus("BUS-003", "1003"))
        self.assertTrue(authenticate_bus("BUS-004", "1004"))
        self.assertTrue(authenticate_bus("BUS-005", "1005"))

    def test_02_invalid_pin_fails(self):
        """Verify that incorrect PIN or non-existent bus fails authentication."""
        self.assertFalse(authenticate_bus("BUS-001", "wrong_pin"))
        self.assertFalse(authenticate_bus("BUS-001", ""))
        self.assertFalse(authenticate_bus("NON_EXISTENT_BUS", "1001"))
        self.assertFalse(authenticate_bus("", "1001"))

    def test_03_available_buses_registry_and_info(self):
        """Verify configured bus list and public info retrieval without exposing PINs."""
        bus_ids = get_available_bus_ids()
        self.assertEqual(len(bus_ids), 5)
        self.assertIn("BUS-001", bus_ids)
        self.assertIn("BUS-005", bus_ids)

        info = get_bus_info("BUS-001")
        self.assertIsNotNone(info)
        self.assertEqual(info["bus_id"], "BUS-001")
        self.assertIn("route", info)
        self.assertNotIn("pin", info, "Security PIN must not be present in public bus profile")

        # Unknown bus info returns None
        self.assertIsNone(get_bus_info("UNKNOWN_BUS"))

    def test_04_format_bus_display(self):
        """Verify bus label formatting handles registered, unregistered, and legacy buses."""
        label_001 = format_bus_display("BUS-001")
        self.assertIn("BUS-001", label_001)

        label_none = format_bus_display(None)
        self.assertEqual(label_none, UNKNOWN_BUS_LABEL)

        label_empty = format_bus_display("")
        self.assertEqual(label_empty, UNKNOWN_BUS_LABEL)

    # -------------------------------------------------------------------------
    # 2. Schema & Bus Identity Data Flow Tests
    # -------------------------------------------------------------------------
    def test_05_urban_event_schema_with_bus_id(self):
        """Verify UrbanEvent creation, serialization, and deserialization with bus_id."""
        evt = UrbanEvent.create(
            event_type="VEHICLE",
            class_name="car",
            confidence=0.91,
            latitude=28.6139,
            longitude=77.2090,
            frame_index=5,
            bbox=[10.0, 20.0, 50.0, 60.0],
            bus_id="BUS-001"
        )
        self.assertEqual(evt.bus_id, "BUS-001")
        d = evt.to_dict()
        self.assertEqual(d["bus_id"], "BUS-001")

        # Reconstruct from dict
        reconstructed = UrbanEvent.from_dict(d)
        self.assertEqual(reconstructed.bus_id, "BUS-001")

    def test_06_urban_event_legacy_compatibility(self):
        """Verify historical events without bus_id default safely to None."""
        legacy_dict = {
            "event_id": "EVT-LEGACY-001",
            "event_type": "ROAD_DAMAGE",
            "class_name": "pothole",
            "confidence": 0.85,
            "severity": "high",
            "latitude": 28.61,
            "longitude": 77.20,
            "timestamp": "2026-08-28T07:00:00Z",
            "source_id": "BUS_DEMO_01",
            "detection_mode": "REAL_AI",
            "gps_mode": "SIMULATED",
            "frame_index": 0,
            "bbox": [0, 0, 10, 10]
            # No bus_id in legacy record
        }
        evt = UrbanEvent.from_dict(legacy_dict)
        self.assertIsNone(evt.bus_id)

    def test_07_event_generator_propagates_bus_id(self):
        """Verify EventGenerator attaches bus_id to all newly created events."""
        gen_002 = EventGenerator(geo_tagger=self.geo_tagger, source_id="CAM_UNIT_02", bus_id="BUS-002")
        veh_dets = [{"class_name": "car", "confidence": 0.88, "bbox": [10.0, 10.0, 50.0, 50.0]}]
        dam_dets = [{"class_name": "pothole", "confidence": 0.75, "bbox": [100.0, 100.0, 150.0, 150.0], "severity": "medium"}]

        events = gen_002.process_frame_detections(0, 100, veh_dets, dam_dets)
        self.assertEqual(len(events), 2)
        for e in events:
            self.assertEqual(e.bus_id, "BUS-002")

    # -------------------------------------------------------------------------
    # 3. Database Persistence, Migration, and Filtering Tests
    # -------------------------------------------------------------------------
    def test_08_database_stores_and_retrieves_bus_identity(self):
        """Verify SQLite database stores bus_id and retrieves it intact."""
        evt1 = UrbanEvent.create("VEHICLE", "car", 0.9, 28.61, 77.21, 1, [0, 0, 10, 10], bus_id="BUS-001")
        evt2 = UrbanEvent.create("VEHICLE", "bus", 0.85, 28.62, 77.22, 2, [0, 0, 10, 10], bus_id="BUS-002")
        evt_legacy = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.7, 28.63, 77.23, 3, [0, 0, 10, 10], bus_id=None)

        self.db.insert_events([evt1, evt2, evt_legacy])

        all_events = self.db.get_events()
        self.assertEqual(len(all_events), 3)

        bus_map = {e.event_id: e.bus_id for e in all_events}
        self.assertEqual(bus_map[evt1.event_id], "BUS-001")
        self.assertEqual(bus_map[evt2.event_id], "BUS-002")
        self.assertIsNone(bus_map[evt_legacy.event_id])

    def test_09_current_bus_data_isolation(self):
        """Verify database filtering isolates data for the active current bus."""
        evt_b1_1 = UrbanEvent.create("VEHICLE", "car", 0.9, 28.61, 77.21, 1, [0, 0, 10, 10], bus_id="BUS-001")
        evt_b1_2 = UrbanEvent.create("VEHICLE", "truck", 0.8, 28.61, 77.21, 2, [0, 0, 10, 10], bus_id="BUS-001")
        evt_b2_1 = UrbanEvent.create("VEHICLE", "bus", 0.85, 28.62, 77.22, 3, [0, 0, 10, 10], bus_id="BUS-002")
        evt_b3_1 = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.7, 28.63, 77.23, 4, [0, 0, 10, 10], bus_id="BUS-003")

        self.db.insert_events([evt_b1_1, evt_b1_2, evt_b2_1, evt_b3_1])

        # Current Bus 001 query
        b1_events = self.db.filter_events(bus_id="BUS-001")
        self.assertEqual(len(b1_events), 2)
        for e in b1_events:
            self.assertEqual(e.bus_id, "BUS-001")

        # Current Bus 002 query
        b2_events = self.db.filter_events(bus_id="BUS-002")
        self.assertEqual(len(b2_events), 1)
        self.assertEqual(b2_events[0].bus_id, "BUS-002")

    def test_10_fleet_view_aggregation(self):
        """Verify fleet view retrieves events across multiple buses simultaneously."""
        evt1 = UrbanEvent.create("VEHICLE", "car", 0.9, 28.61, 77.21, 1, [0, 0, 10, 10], bus_id="BUS-001")
        evt2 = UrbanEvent.create("VEHICLE", "car", 0.85, 28.62, 77.22, 2, [0, 0, 10, 10], bus_id="BUS-002")
        evt3 = UrbanEvent.create("VEHICLE", "car", 0.80, 28.63, 77.23, 3, [0, 0, 10, 10], bus_id="BUS-003")

        self.db.insert_events([evt1, evt2, evt3])

        # Fleet view query (bus_id=None)
        fleet_events = self.db.filter_events(bus_id=None)
        self.assertEqual(len(fleet_events), 3)

        stats = self.db.get_event_statistics()
        self.assertIn("by_bus_id", stats)
        self.assertEqual(stats["by_bus_id"].get("BUS-001"), 1)
        self.assertEqual(stats["by_bus_id"].get("BUS-002"), 1)
        self.assertEqual(stats["by_bus_id"].get("BUS-003"), 1)

    def test_11_legacy_events_remain_unknown(self):
        """Verify historical events without bus_id are distinguishable as UNKNOWN."""
        evt_legacy = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.7, 28.61, 77.21, 1, [0, 0, 10, 10], bus_id=None)
        evt_bus = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.8, 28.62, 77.22, 2, [0, 0, 10, 10], bus_id="BUS-001")

        self.db.insert_events([evt_legacy, evt_bus])

        legacy_results = self.db.filter_events(bus_id="UNKNOWN")
        self.assertEqual(len(legacy_results), 1)
        self.assertEqual(legacy_results[0].event_id, evt_legacy.event_id)
        self.assertIsNone(legacy_results[0].bus_id)

    def test_12_csv_and_json_export_preserves_bus_id(self):
        """Verify CSV and JSON file exports include bus_id."""
        evt = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.95, 28.614, 77.209, 12, [50, 50, 120, 120], severity="high", bus_id="BUS-004")
        self.db.insert_event(evt)

        csv_file = settings.DATA_DIR / "events" / "test_bus_export.csv"
        json_file = settings.DATA_DIR / "events" / "test_bus_export.json"

        self.db.export_events_csv(str(csv_file))
        self.db.export_events_json(str(json_file))

        self.assertTrue(csv_file.exists())
        self.assertTrue(json_file.exists())

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["bus_id"], "BUS-004")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["bus_id"], "BUS-004")

        if csv_file.exists(): 
            try: csv_file.unlink() 
            except Exception: pass
        if json_file.exists(): 
            try: json_file.unlink()
            except Exception: pass

    # -------------------------------------------------------------------------
    # 4. End-to-End Video Processor Bus ID Pipeline
    # -------------------------------------------------------------------------
    def test_13_video_pipeline_preserves_bus_identity(self):
        """Verify VideoProcessor generates events tagged with specified bus_id."""
        veh_det = VehicleDetector(model_name="yolov8n.pt")
        dam_det = RoadDamageDetector()
        proc = VideoProcessor(
            vehicle_detector=veh_det,
            road_damage_detector=dam_det,
            db_manager=self.db,
            geo_tagger=self.geo_tagger
        )

        res = proc.process_video(
            input_path=str(self.sample_video),
            frame_skip=2,
            max_frames=10,
            save_to_db=True,
            bus_id="BUS-005"
        )

        self.assertEqual(res["bus_id"], "BUS-005")
        self.assertGreater(res["total_generated_events"], 0)

        # Verify in DB
        bus_events = self.db.filter_events(bus_id="BUS-005")
        self.assertEqual(len(bus_events), res["total_generated_events"])
        for e in bus_events:
            self.assertEqual(e.bus_id, "BUS-005")

    # -------------------------------------------------------------------------
    # 5. Session State & Multi-Bus Isolation Simulation
    # -------------------------------------------------------------------------
    def test_14_logout_and_switch_bus_session_isolation(self):
        """Verify simulated session lifecycle: login -> process -> logout -> switch bus -> process."""
        # Simulated Session 1: BUS-001 Login & Event Generation
        session_state = {"authenticated": True, "current_bus_id": "BUS-001", "data_scope": "CURRENT BUS"}
        evt_bus1 = UrbanEvent.create("VEHICLE", "car", 0.9, 28.61, 77.21, 1, [0, 0, 10, 10], bus_id=session_state["current_bus_id"])
        self.db.insert_event(evt_bus1)

        # Verify Session 1 queries see only BUS-001
        events_s1 = self.db.filter_events(bus_id=session_state["current_bus_id"])
        self.assertEqual(len(events_s1), 1)
        self.assertEqual(events_s1[0].bus_id, "BUS-001")

        # Simulated Logout
        session_state["authenticated"] = False
        session_state["current_bus_id"] = None
        self.assertFalse(session_state["authenticated"])
        self.assertIsNone(session_state["current_bus_id"])

        # Simulated Session 2: Login as BUS-002
        self.assertTrue(authenticate_bus("BUS-002", "1002"))
        session_state["authenticated"] = True
        session_state["current_bus_id"] = "BUS-002"

        evt_bus2 = UrbanEvent.create("VEHICLE", "truck", 0.85, 28.62, 77.22, 2, [0, 0, 10, 10], bus_id=session_state["current_bus_id"])
        self.db.insert_event(evt_bus2)

        # Verify Session 2 Current Bus sees only BUS-002
        events_s2 = self.db.filter_events(bus_id=session_state["current_bus_id"])
        self.assertEqual(len(events_s2), 1)
        self.assertEqual(events_s2[0].bus_id, "BUS-002")

        # Fleet View sees both buses
        events_fleet = self.db.get_events()
        self.assertEqual(len(events_fleet), 2)
        bus_set = {e.bus_id for e in events_fleet}
        self.assertEqual(bus_set, {"BUS-001", "BUS-002"})

    # -------------------------------------------------------------------------
    # 6. Traffic Analytics Compatibility & Semantics Preservation
    # -------------------------------------------------------------------------
    def test_15_traffic_analytics_bus_filtering_and_class_semantics(self):
        """Verify traffic analytics can filter by bus_id while strictly preserving valid vehicle semantics."""
        evt_b1_car = UrbanEvent.create("VEHICLE", "car", 0.9, 28.61, 77.21, 1, [0, 0, 10, 10], bus_id="BUS-001")
        evt_b1_person = UrbanEvent.create("VEHICLE", "person", 0.9, 28.61, 77.21, 2, [0, 0, 10, 10], bus_id="BUS-001")
        evt_b2_bus = UrbanEvent.create("VEHICLE", "bus", 0.85, 28.62, 77.22, 3, [0, 0, 10, 10], bus_id="BUS-002")

        analytics = TrafficAnalytics(events=[evt_b1_car, evt_b1_person, evt_b2_bus])

        # Current Bus 001: should only count the 1 car (person excluded!)
        b1_vehs = analytics.get_vehicle_events(bus_id="BUS-001")
        self.assertEqual(len(b1_vehs), 1)
        self.assertEqual(b1_vehs[0].class_name, "car")

        # Current Bus 002: should count 1 bus
        b2_vehs = analytics.get_vehicle_events(bus_id="BUS-002")
        self.assertEqual(len(b2_vehs), 1)
        self.assertEqual(b2_vehs[0].class_name, "bus")

        # Fleet View: should count 2 vehicles (car + bus), strictly excluding person
        fleet_vehs = analytics.get_vehicle_events()
        self.assertEqual(len(fleet_vehs), 2)


if __name__ == "__main__":
    unittest.main()
