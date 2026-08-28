"""
Checkpoint 2 Test Suite: Vehicle Detection & CPU Benchmarking
"""
import unittest
import os
from pathlib import Path
import cv2
import numpy as np
from src.detection.vehicle_detector import VehicleDetector
from src.video.processor import VideoProcessor
from config import settings

class TestCheckpoint2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = VehicleDetector(model_name="yolov8n.pt", conf_threshold=0.35)
        cls.processor = VideoProcessor(detector=cls.detector)
        cls.test_video_path = settings.SAMPLE_DATA_DIR / "real_road_sample.mp4"
        if not cls.test_video_path.exists():
            cls.test_video_path = settings.SAMPLE_DATA_DIR / "backup_road_demo.mp4"

    def test_01_model_loading(self):
        """Verify YOLOv8n loads and records positive load time."""
        self.assertIsNotNone(self.detector.model)
        self.assertGreater(self.detector.load_time_sec, 0.0)
        self.assertEqual(self.detector.model_name, "yolov8n.pt")

    def test_02_single_frame_inference_schema(self):
        """Verify structured detection schema on a synthetic/test frame."""
        # Create test frame with arbitrary dimensions
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.detector.detect(test_frame)
        
        self.assertIn("detections", res)
        self.assertIn("inference_time_ms", res)
        self.assertIsInstance(res["detections"], list)
        self.assertIsInstance(res["inference_time_ms"], float)
        self.assertGreater(res["inference_time_ms"], 0.0)

    def test_03_draw_detections(self):
        """Verify draw_detections runs and returns an annotated image of same shape."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_dets = [{
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.85,
            "bbox": [50.0, 50.0, 200.0, 150.0]
        }]
        annotated = self.detector.draw_detections(test_frame, dummy_dets)
        self.assertEqual(annotated.shape, test_frame.shape)

    def test_04_video_opening_and_processing(self):
        """Verify video processing on real road sample for 15 frames."""
        self.assertTrue(self.test_video_path.exists(), "Sample video file must exist")
        
        output_file = settings.SAMPLE_DATA_DIR / "test_annotated_output.mp4"
        benchmarks = self.processor.process_video(
            input_path=str(self.test_video_path),
            output_path=str(output_file),
            frame_skip=1,
            max_frames=15
        )

        # Validate benchmark metrics presence
        self.assertEqual(benchmarks["processed_frames_count"], 15)
        self.assertGreater(benchmarks["total_inference_time_sec"], 0.0)
        self.assertGreater(benchmarks["avg_inference_time_ms_per_frame"], 0.0)
        self.assertGreater(benchmarks["model_inference_fps"], 0.0)
        self.assertGreater(benchmarks["complete_pipeline_fps"], 0.0)
        self.assertIn("class_counts", benchmarks)

        # Validate output video exists and has non-zero size
        self.assertTrue(output_file.exists())
        self.assertGreater(output_file.stat().st_size, 0)
        
        # Cleanup test output
        if output_file.exists():
            output_file.unlink()

if __name__ == "__main__":
    unittest.main()
