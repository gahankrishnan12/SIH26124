from src.detection.vehicle_detector import VehicleDetector
import cv2
from collections import Counter

print("Starting extended vehicle detection test...")

model = VehicleDetector(model_name="yolov8n.pt")

video_path = "data/sample/real_road_sample.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    raise SystemExit(1)

counts = Counter()
frames = 0
detections = 0

while frames < 500:
    ret, frame = cap.read()

    if not ret:
        break

    result = model.detect(frame)

    for detection in result["detections"]:
        class_name = detection["class_name"]
        counts[class_name] += 1
        detections += 1

    frames += 1

cap.release()

print()
print("========== EXTENDED RESULT ==========")
print("Frames inspected:", frames)
print("Raw detections:", detections)
print("Detected classes:", dict(counts))
print("======================================")
