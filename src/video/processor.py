"""
SIH26124: Video Processor Module (Integrated with Event Generator & SQLite Database)
Handles video ingestion, dual AI inference, HUD annotation, simulated GPS geotagging,
event deduplication, SQLite persistence, and CPU benchmarking.
"""
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import cv2
import numpy as np

from src.detection.vehicle_detector import VehicleDetector
from src.detection.road_damage_detector import RoadDamageDetector
from src.events.geo_tagger import GeoTagger
from src.events.generator import EventGenerator
from src.storage.db_manager import DatabaseManager

class VideoProcessor:
    """
    Processes road video feeds, runs dual AI detection, generates geotagged events,
    deduplicates consecutive detections, persists records to SQLite, and computes CPU benchmarks.
    """
    def __init__(
        self,
        vehicle_detector: Optional[VehicleDetector] = None,
        road_damage_detector: Optional[RoadDamageDetector] = None,
        db_manager: Optional[DatabaseManager] = None,
        geo_tagger: Optional[GeoTagger] = None,
        detector: Optional[VehicleDetector] = None  # Backward compatibility alias
    ):
        self.vehicle_detector = vehicle_detector or detector or VehicleDetector(model_name="yolov8n.pt")
        self.road_damage_detector = road_damage_detector or RoadDamageDetector()
        self.db_manager = db_manager or DatabaseManager()
        self.geo_tagger = geo_tagger or GeoTagger()

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        enable_vehicle_detection: bool = True,
        enable_damage_detection: bool = True,
        save_to_db: bool = True,
        source_id: str = "BUS_DEMO_01",
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process an input video file with dual AI inference and event persistence.
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(str(input_file))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video file: {input_path}")

        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_video_frames <= 0:
            total_video_frames = 100

        frames_to_read = total_video_frames
        if max_frames is not None and max_frames > 0:
            frames_to_read = min(total_video_frames, max_frames)

        if output_path is None:
            output_file = input_file.parent / f"annotated_{input_file.name}"
        else:
            output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_file),
            fourcc,
            input_fps / frame_skip if frame_skip > 0 else input_fps,
            (orig_width, orig_height)
        )

        event_generator = EventGenerator(geo_tagger=self.geo_tagger, source_id=source_id)

        frame_idx = 0
        processed_count = 0
        skipped_count = 0
        total_vehicle_inf_sec = 0.0
        total_damage_inf_sec = 0.0
        
        vehicle_class_counts: Dict[str, int] = {}
        damage_severity_counts: Dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        total_vehicles = 0
        total_damages = 0
        all_new_events = []

        overall_start_time = time.perf_counter()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames is not None and frame_idx >= max_frames):
                break

            if frame_skip > 1 and (frame_idx % frame_skip != 0):
                skipped_count += 1
                frame_idx += 1
                continue

            annotated_frame = frame.copy()
            veh_detections: List[Dict[str, Any]] = []
            dam_detections: List[Dict[str, Any]] = []
            veh_time_ms = 0.0
            dam_time_ms = 0.0

            # 1. Vehicle Detection
            if enable_vehicle_detection and self.vehicle_detector:
                veh_res = self.vehicle_detector.detect(frame)
                veh_detections = veh_res["detections"]
                veh_time_ms = veh_res["inference_time_ms"]
                total_vehicle_inf_sec += (veh_time_ms / 1000.0)
                
                for v in veh_detections:
                    cname = v["class_name"]
                    vehicle_class_counts[cname] = vehicle_class_counts.get(cname, 0) + 1
                    total_vehicles += 1
                    
                annotated_frame = self.vehicle_detector.draw_detections(annotated_frame, veh_detections)

            # 2. Road Damage Detection
            if enable_damage_detection and self.road_damage_detector:
                dam_res = self.road_damage_detector.detect(frame)
                dam_detections = dam_res["detections"]
                dam_time_ms = dam_res["inference_time_ms"]
                total_damage_inf_sec += (dam_time_ms / 1000.0)

                for d in dam_detections:
                    sev = d.get("severity", "low").lower()
                    damage_severity_counts[sev] = damage_severity_counts.get(sev, 0) + 1
                    total_damages += 1

                annotated_frame = self.road_damage_detector.draw_detections(annotated_frame, dam_detections)

            # 3. Event Generation & Deduplication
            new_events = event_generator.process_frame_detections(
                frame_index=frame_idx,
                total_frames=frames_to_read,
                vehicle_detections=veh_detections,
                damage_detections=dam_detections,
                video_fps=input_fps
            )
            all_new_events.extend(new_events)

            if save_to_db and new_events:
                self.db_manager.insert_events(new_events)

            processed_count += 1
            total_frame_inf_ms = veh_time_ms + dam_time_ms

            # 4. Telemetry HUD Overlay
            hud_line_1 = f"Frame: {frame_idx+1}/{frames_to_read} | Veh: {len(veh_detections)} | Damage: {len(dam_detections)}"
            hud_line_2 = f"CPU Inf: {total_frame_inf_ms:.1f}ms | Events: {len(event_generator.all_generated_events)} (Dedup: {event_generator.total_duplicates_filtered})"
            dam_mode_label = f"Damage: {self.road_damage_detector.detection_mode} | GPS: SIMULATED"
            
            cv2.rectangle(annotated_frame, (10, 10), (460, 75), (25, 25, 25), -1)
            cv2.putText(annotated_frame, hud_line_1, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, hud_line_2, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, dam_mode_label, (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 220, 255), 1, cv2.LINE_AA)

            writer.write(annotated_frame)

            if progress_callback is not None:
                elapsed = time.perf_counter() - overall_start_time
                current_fps = processed_count / elapsed if elapsed > 0 else 0.0
                try:
                    progress_callback(processed_count, frames_to_read // frame_skip, current_fps, len(veh_detections), len(dam_detections))
                except TypeError:
                    progress_callback(processed_count, frames_to_read // frame_skip, current_fps, len(veh_detections))

            frame_idx += 1

        cap.release()
        writer.release()

        total_processing_time_sec = time.perf_counter() - overall_start_time
        total_dual_inf_sec = total_vehicle_inf_sec + total_damage_inf_sec

        avg_veh_ms = (total_vehicle_inf_sec / processed_count * 1000.0) if processed_count > 0 else 0.0
        avg_dam_ms = (total_damage_inf_sec / processed_count * 1000.0) if processed_count > 0 else 0.0
        avg_total_ms = (total_dual_inf_sec / processed_count * 1000.0) if processed_count > 0 else 0.0
        
        dual_inf_fps = (processed_count / total_dual_inf_sec) if total_dual_inf_sec > 0 else 0.0
        complete_fps = (processed_count / total_processing_time_sec) if total_processing_time_sec > 0 else 0.0

        db_stats = self.db_manager.get_event_statistics()

        return {
            "model_name": self.vehicle_detector.model_name if self.vehicle_detector else "yolov8n.pt",
            "vehicle_model_name": self.vehicle_detector.model_name if self.vehicle_detector else "None",
            "damage_model_name": self.road_damage_detector.model_name if self.road_damage_detector else "None",
            "road_damage_mode": self.road_damage_detector.detection_mode if self.road_damage_detector else "NONE",
            "road_damage_mode_disclosure": self.road_damage_detector.mode_disclosure_text if self.road_damage_detector else "",
            "gps_mode": "SIMULATED",
            "database_path": str(self.db_manager.db_path),
            "input_video_path": str(input_file),
            "output_video_path": str(output_file),
            "input_resolution": f"{orig_width}x{orig_height}",
            "input_video_fps": round(input_fps, 2),
            "total_video_frames": total_video_frames,
            "processed_frames_count": processed_count,
            "skipped_frames_count": skipped_count,
            "frame_skip_interval": frame_skip,
            # Core benchmark metrics expected by all test suites
            "total_inference_time_sec": round(total_dual_inf_sec, 3),
            "avg_inference_time_ms_per_frame": round(avg_total_ms, 2),
            "model_inference_fps": round(dual_inf_fps, 2),
            "avg_vehicle_inference_ms": round(avg_veh_ms, 2),
            "avg_damage_inference_ms": round(avg_dam_ms, 2),
            "avg_combined_inference_ms": round(avg_total_ms, 2),
            "dual_model_inference_fps": round(dual_inf_fps, 2),
            "total_processing_time_sec": round(total_processing_time_sec, 3),
            "complete_pipeline_fps": round(complete_fps, 2),
            "total_raw_detections": event_generator.total_raw_detections_count,
            "total_generated_events": len(event_generator.all_generated_events),
            "total_duplicates_filtered": event_generator.total_duplicates_filtered,
            "class_counts": vehicle_class_counts,
            "vehicle_class_counts": vehicle_class_counts,
            "total_detections_count": total_vehicles + total_damages,
            "total_vehicles_detected": total_vehicles,
            "damage_severity_counts": damage_severity_counts,
            "total_damages_detected": total_damages,
            "database_statistics": db_stats,
            "generated_events": [e.to_dict() for e in event_generator.all_generated_events]
        }
