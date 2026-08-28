"""
Checkpoint 3 Test Suite: Road Damage Detection & Dual Pipeline
"""
import unittest
import os
from pathlib import Path
import cv2
import numpy as np
from src.detection.road_damage_detector import RoadDamageDetector
from src.detection.vehicle_detector import VehicleDetector
from src.video.processor import VideoProcessor
from config import settings

class TestCheckpoint3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.damage_detector = RoadDamageDetector()
        cls.vehicle_detector = VehicleDetector(model_name="yolov8n.pt")
        cls.processor = VideoProcessor(
            vehicle_detector=cls.vehicle_detector,
            road_damage_detector=cls.damage_detector
        )
        cls.test_image_path = settings.SAMPLE_DATA_DIR / "test_pothole_big.jpg"
        cls.test_video_path = settings.SAMPLE_DATA_DIR / "real_road_sample.mp4"

    def test_01_damage_detector_mode_and_loading(self):
        """Verify RoadDamageDetector correctly loads model and sets REAL_AI mode."""
        self.assertEqual(self.damage_detector.detection_mode, "REAL_AI")
        self.assertIsNotNone(self.damage_detector.model)
        self.assertIn("REAL AI MODE", self.damage_detector.mode_disclosure_text)

    def test_02_simulation_mode_fallback(self):
        """Verify DEMO_SIMULATION mode when force_demo_mode=True."""
        sim_detector = RoadDamageDetector(force_demo_mode=True)
        self.assertEqual(sim_detector.detection_mode, "DEMO_SIMULATION")
        self.assertIn("DEMO/SIMULATION MODE", sim_detector.mode_disclosure_text)
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = sim_detector.detect(test_frame)
        self.assertEqual(res["detection_mode"], "DEMO_SIMULATION")

    def test_03_severity_heuristic_calculation(self):
        """Verify severity classification based on bbox relative area."""
        # Frame: 1000 x 1000 = 1,000,000 px^2
        # Low: 100 x 100 = 10,000 px^2 (1.0% area)
        sev_low = self.damage_detector.calculate_severity([0, 0, 100, 100], 1000, 1000)
        self.assertEqual(sev_low, "low")

        # Medium: 150 x 150 = 22,500 px^2 (2.25% area)
        sev_med = self.damage_detector.calculate_severity([0, 0, 150, 150], 1000, 1000)
        self.assertEqual(sev_med, "medium")

        # High: 250 x 200 = 50,000 px^2 (5.0% area)
        sev_high = self.damage_detector.calculate_severity([0, 0, 250, 200], 1000, 1000)
        self.assertEqual(sev_high, "high")

    def test_04_single_image_inference_on_known_pothole(self):
        """Verify actual model detection on test_pothole_big.jpg."""
        self.assertTrue(self.test_image_path.exists(), "test_pothole_big.jpg missing")
        img = cv2.imread(str(self.test_image_path))
        self.assertIsNotNone(img)

        res = self.damage_detector.detect(img)
        self.assertEqual(res["detection_mode"], "REAL_AI")
        self.assertIsInstance(res["detections"], list)
        self.assertGreater(len(res["detections"]), 0, "Model should detect pothole in test image")

        det = res["detections"][0]
        self.assertIn("class_name", det)
        self.assertIn("confidence", det)
        self.assertIn("bbox", det)
        self.assertIn("severity", det)
        self.assertIn(det["severity"], ["low", "medium", "high"])

    def test_05_dual_video_processing(self):
        """Verify dual vehicle + road damage video pipeline on 15 frames."""
        self.assertTrue(self.test_video_path.exists())
        output_file = settings.SAMPLE_DATA_DIR / "test_dual_annotated.mp4"

        benchmarks = self.processor.process_video(
            input_path=str(self.test_video_path),
            output_path=str(output_file),
            frame_skip=1,
            max_frames=15,
            enable_vehicle_detection=True,
            enable_damage_detection=True
        )

        self.assertEqual(benchmarks["processed_frames_count"], 15)
        self.assertEqual(benchmarks["road_damage_mode"], "REAL_AI")
        self.assertGreater(benchmarks["avg_combined_inference_ms"], 0.0)
        self.assertGreater(benchmarks["dual_model_inference_fps"], 0.0)
        self.assertIn("damage_severity_counts", benchmarks)
        self.assertTrue(output_file.exists())

        if output_file.exists():
            output_file.unlink()

if __name__ == "__main__":
    unittest.main()
