"""
Checkpoint 4 Test Suite: Event Generation, Geotagging, Deduplication & SQLite Persistence
"""
import unittest
import os
import json
import csv
import gc
from pathlib import Path
from src.events.schema import UrbanEvent
from src.events.geo_tagger import GeoTagger
from src.events.generator import EventGenerator
from src.storage.db_manager import DatabaseManager
from src.detection.vehicle_detector import VehicleDetector
from src.detection.road_damage_detector import RoadDamageDetector
from src.video.processor import VideoProcessor
from config import settings

class TestCheckpoint4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_path = settings.DATA_DIR / "events" / "test_urban_events.db"
        cls.db = DatabaseManager(db_path=str(cls.test_db_path))
        cls.geo_tagger = GeoTagger()
        cls.generator = EventGenerator(geo_tagger=cls.geo_tagger)
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

    def test_01_event_schema_creation(self):
        """Verify UrbanEvent creation and field structure."""
        evt = UrbanEvent.create(
            event_type="ROAD_DAMAGE",
            class_name="pothole",
            confidence=0.88,
            latitude=28.6139,
            longitude=77.2090,
            frame_index=10,
            bbox=[100.0, 150.0, 200.0, 250.0],
            severity="high",
            detection_mode="REAL_AI",
            gps_mode="SIMULATED"
        )
        self.assertEqual(evt.event_type, "ROAD_DAMAGE")
        self.assertEqual(evt.class_name, "pothole")
        self.assertEqual(evt.severity, "high")
        self.assertEqual(evt.detection_mode, "REAL_AI")
        self.assertEqual(evt.gps_mode, "SIMULATED")

    def test_02_event_id_generation(self):
        """Verify unique deterministic timestamped ID generation."""
        evt1 = UrbanEvent.create("VEHICLE", "car", 0.9, 28.0, 77.0, 1, [0, 0, 10, 10])
        evt2 = UrbanEvent.create("VEHICLE", "car", 0.9, 28.0, 77.0, 2, [0, 0, 10, 10])
        self.assertTrue(evt1.event_id.startswith("EVT-"))
        self.assertNotEqual(evt1.event_id, evt2.event_id)

    def test_03_gps_interpolation(self):
        """Verify simulated GPS interpolation along route waypoints."""
        coord_mid = self.geo_tagger.get_coordinate_for_frame(50, 100)
        self.assertEqual(coord_mid["gps_mode"], "SIMULATED")
        self.assertIsInstance(coord_mid["latitude"], float)
        self.assertIsInstance(coord_mid["longitude"], float)

    def test_04_gps_boundary_conditions(self):
        """Verify start, end, and single-frame boundary handling in GeoTagger."""
        coord_start = self.geo_tagger.get_coordinate_for_frame(0, 100)
        first_wp = self.geo_tagger.waypoints[0]
        self.assertAlmostEqual(coord_start["latitude"], first_wp["latitude"], places=4)

        coord_end = self.geo_tagger.get_coordinate_for_frame(99, 100)
        last_wp = self.geo_tagger.waypoints[-1]
        self.assertAlmostEqual(coord_end["latitude"], last_wp["latitude"], places=4)

        coord_single = self.geo_tagger.get_coordinate_for_frame(0, 1)
        self.assertAlmostEqual(coord_single["latitude"], first_wp["latitude"], places=4)

    def test_05_event_generation_from_ai_detections(self):
        """Verify EventGenerator produces UrbanEvent from raw detection dict."""
        gen = EventGenerator(geo_tagger=self.geo_tagger)
        veh_dets = [{"class_name": "car", "confidence": 0.85, "bbox": [100.0, 100.0, 200.0, 200.0]}]
        dam_dets = [{"class_name": "pothole", "confidence": 0.78, "bbox": [300.0, 300.0, 400.0, 400.0], "severity": "medium", "detection_mode": "REAL_AI"}]

        events = gen.process_frame_detections(0, 100, veh_dets, dam_dets)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, "VEHICLE")
        self.assertEqual(events[1].event_type, "ROAD_DAMAGE")

    def test_06_detection_mode_preservation(self):
        """Verify REAL_AI and DEMO_SIMULATION modes are preserved on events."""
        gen = EventGenerator(geo_tagger=self.geo_tagger)
        sim_dam_det = [{"class_name": "pothole", "confidence": 0.50, "bbox": [50.0, 50.0, 100.0, 100.0], "severity": "low", "detection_mode": "DEMO_SIMULATION"}]
        events = gen.process_frame_detections(0, 100, [], sim_dam_det)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].detection_mode, "DEMO_SIMULATION")

    def test_07_sqlite_initialization(self):
        """Verify SQLite database and events table exist."""
        self.assertTrue(self.test_db_path.exists())
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'")
            self.assertIsNotNone(cursor.fetchone())
        finally:
            conn.close()

    def test_08_event_insertion_and_retrieval(self):
        """Verify single and batch event insertion and retrieval."""
        evt1 = UrbanEvent.create("VEHICLE", "bus", 0.92, 28.61, 77.21, 5, [10, 10, 50, 50])
        evt2 = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.81, 28.62, 77.22, 10, [20, 20, 80, 80], severity="high")

        self.db.insert_event(evt1)
        self.db.insert_events([evt2])

        events = self.db.get_events()
        self.assertEqual(len(events), 2)
        ids = [e.event_id for e in events]
        self.assertIn(evt1.event_id, ids)
        self.assertIn(evt2.event_id, ids)

    def test_09_event_filtering(self):
        """Verify database queries filter by type, severity, and mode."""
        evt1 = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.8, 28.6, 77.2, 1, [0, 0, 10, 10], severity="high", detection_mode="REAL_AI")
        evt2 = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.6, 28.6, 77.2, 2, [0, 0, 10, 10], severity="low", detection_mode="REAL_AI")
        evt3 = UrbanEvent.create("VEHICLE", "car", 0.9, 28.6, 77.2, 3, [0, 0, 10, 10])

        self.db.insert_events([evt1, evt2, evt3])

        high_sev = self.db.filter_events(severity="high")
        self.assertEqual(len(high_sev), 1)
        self.assertEqual(high_sev[0].event_id, evt1.event_id)

        vehicles = self.db.filter_events(event_type="VEHICLE")
        self.assertEqual(len(vehicles), 1)

    def test_10_deduplication_mechanism(self):
        """Verify consecutive frame detections of the same pothole are deduplicated."""
        gen = EventGenerator(geo_tagger=self.geo_tagger, dedup_window_frames=10, dedup_spatial_threshold=50.0)
        
        det_frame1 = [{"class_name": "pothole", "confidence": 0.70, "bbox": [100.0, 100.0, 150.0, 150.0], "severity": "medium"}]
        new1 = gen.process_frame_detections(1, 100, [], det_frame1)
        self.assertEqual(len(new1), 1, "First occurrence should be a new event")

        det_frame2 = [{"class_name": "pothole", "confidence": 0.85, "bbox": [102.0, 102.0, 152.0, 152.0], "severity": "medium"}]
        new2 = gen.process_frame_detections(2, 100, [], det_frame2)
        self.assertEqual(len(new2), 0, "Consecutive occurrence should be deduplicated")
        self.assertEqual(gen.total_duplicates_filtered, 1)
        self.assertEqual(gen.all_generated_events[0].confidence, 0.85)

    def test_11_csv_and_json_export(self):
        """Verify CSV and JSON file exports."""
        evt = UrbanEvent.create("ROAD_DAMAGE", "pothole", 0.95, 28.614, 77.209, 12, [50, 50, 120, 120], severity="high")
        self.db.insert_event(evt)

        csv_file = settings.DATA_DIR / "events" / "test_export.csv"
        json_file = settings.DATA_DIR / "events" / "test_export.json"

        self.db.export_events_csv(str(csv_file))
        self.db.export_events_json(str(json_file))

        self.assertTrue(csv_file.exists())
        self.assertTrue(json_file.exists())

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["class_name"], "pothole")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["event_id"], evt.event_id)

        if csv_file.exists(): 
            try: csv_file.unlink() 
            except Exception: pass
        if json_file.exists(): 
            try: json_file.unlink()
            except Exception: pass

    def test_12_end_to_end_pipeline(self):
        """Verify complete Video -> AI -> Event Generator -> SQLite pipeline on 15 frames."""
        self.assertTrue(self.sample_video.exists())
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
            frame_skip=1,
            max_frames=15,
            save_to_db=True
        )

        self.assertGreater(res["total_raw_detections"], 0)
        self.assertGreater(res["total_generated_events"], 0)
        self.assertIn("database_statistics", res)
        self.assertEqual(res["gps_mode"], "SIMULATED")

        stored = self.db.get_events()
        self.assertEqual(len(stored), res["total_generated_events"])

if __name__ == "__main__":
    unittest.main()
