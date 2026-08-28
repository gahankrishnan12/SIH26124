import unittest
import json
import os
import cv2
from pathlib import Path

class TestCheckpoint1(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(__file__).resolve().parent.parent

    def test_01_required_packages_importable(self):
        """Verify core minimal packages import without errors."""
        import streamlit
        import cv2
        import pandas
        import numpy
        import ultralytics
        import torch
        import folium
        import streamlit_folium

        self.assertIsNotNone(streamlit.__version__)
        self.assertIsNotNone(ultralytics.__version__)
        self.assertIsNotNone(torch.__version__)
        self.assertIsNotNone(folium.__version__)

    def test_02_directory_structure_exists(self):
        """Verify standard project directories are created."""
        expected_dirs = [
            "config",
            "models",
            "data/sample",
            "data/events",
            "data/gps",
            "src/video",
            "src/detection",
            "src/events",
            "src/storage",
            "src/analytics",
            "src/maps",
            "tests",
            "docs"
        ]
        for rel_dir in expected_dirs:
            full_path = self.base_dir / rel_dir
            self.assertTrue(full_path.exists(), f"Missing directory: {rel_dir}")
            self.assertTrue(full_path.is_dir(), f"Path is not a directory: {rel_dir}")

    def test_03_settings_configuration(self):
        """Verify config/settings.py loads paths and mappings."""
        from config import settings
        self.assertTrue(settings.DATA_DIR.exists())
        self.assertIn(2, settings.VEHICLE_CLASS_MAP)
        self.assertEqual(settings.VEHICLE_CLASS_MAP[2], "car")

    def test_04_gps_sample_route_validity(self):
        """Verify sample GPS route JSON parses properly and contains simulated flags."""
        route_file = self.base_dir / "data" / "gps" / "sample_bus_route.json"
        self.assertTrue(route_file.exists(), "sample_bus_route.json missing")
        with open(route_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(data.get("is_simulated", False))
        self.assertGreater(len(data.get("waypoints", [])), 0)

    def test_05_backup_video_readable(self):
        """Verify sample backup video exists and is openable via OpenCV."""
        video_file = self.base_dir / "data" / "sample" / "backup_road_demo.mp4"
        self.assertTrue(video_file.exists(), "backup_road_demo.mp4 missing")
        cap = cv2.VideoCapture(str(video_file))
        self.assertTrue(cap.isOpened(), "Could not open video file")
        ret, frame = cap.read()
        self.assertTrue(ret, "Could not read first frame from video")
        self.assertEqual(frame.shape, (480, 640, 3))
        cap.release()

    def test_06_app_compiles(self):
        """Verify app.py syntax compiles cleanly."""
        app_file = self.base_dir / "app.py"
        self.assertTrue(app_file.exists(), "app.py missing")
        with open(app_file, "r", encoding="utf-8") as f:
            source = f.read()
        compiled = compile(source, str(app_file), "exec")
        self.assertIsNotNone(compiled)

if __name__ == "__main__":
    unittest.main()
