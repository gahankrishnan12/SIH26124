"""
SIH26124: Vehicle Detector Module
Lightweight CPU-friendly vehicle & pedestrian detection using Ultralytics YOLOv8n.
"""
import time
from typing import List, Dict, Any
import numpy as np
import cv2
from ultralytics import YOLO
from config import settings

class VehicleDetector:
    """
    Lightweight YOLOv8 vehicle detector for CPU execution.
    Filters target COCO classes: person, car, motorcycle, bus, truck.
    """
    # Color palette for distinct visualization per class (BGR format)
    CLASS_COLORS = {
        "car": (0, 200, 0),        # Green
        "bus": (255, 140, 0),      # Blue/Amber
        "truck": (0, 165, 255),    # Orange
        "motorcycle": (200, 0, 200), # Magenta
        "person": (255, 200, 0)    # Cyan
    }

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.35):
        """
        Initialize the vehicle detector.
        Measures actual model loading time.
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.target_classes = settings.VEHICLE_CLASS_MAP # {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        # Load YOLO model and measure load time
        start_load = time.perf_counter()
        # Save/load checkpoint in models directory or default cache
        model_path = settings.MODELS_DIR / model_name
        if model_path.exists():
            self.model = YOLO(str(model_path))
        else:
            self.model = YOLO(model_name)
            # Save downloaded weights to models/ for local reproducibility
            try:
                if not model_path.exists() and hasattr(self.model, 'ckpt_path') and self.model.ckpt_path:
                    import shutil
                    shutil.copy(self.model.ckpt_path, str(model_path))
            except Exception:
                pass
                
        self.load_time_sec = time.perf_counter() - start_load

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a single frame (BGR numpy array).
        Returns structured detections and measured inference latency.
        """
        if frame is None or frame.size == 0:
            return {"detections": [], "inference_time_ms": 0.0}

        start_inf = time.perf_counter()
        # Run inference strictly on CPU
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            classes=list(self.target_classes.keys()),
            device="cpu",
            verbose=False
        )
        inference_time_ms = (time.perf_counter() - start_inf) * 1000.0

        detections: List[Dict[str, Any]] = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                    
                    class_name = self.target_classes.get(cls_id, f"class_{cls_id}")
                    
                    detections.append({
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": round(conf, 4),
                        "bbox": xyxy  # [x1, y1, x2, y2]
                    })

        return {
            "detections": detections,
            "inference_time_ms": round(inference_time_ms, 2)
        }

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes, labels, and confidence tags on a copy of the frame.
        """
        annotated = frame.copy()
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            class_name = det["class_name"]
            conf = det["confidence"]
            color = self.CLASS_COLORS.get(class_name, (0, 255, 0))

            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label banner
            label = f"{class_name.upper()} {conf*100:.1f}%"
            (lbl_w, lbl_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Ensure label banner does not go outside top border
            top_y = max(y1 - lbl_h - 6, 0)
            cv2.rectangle(annotated, (x1, top_y), (x1 + lbl_w + 6, top_y + lbl_h + 6), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, top_y + lbl_h + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        return annotated
