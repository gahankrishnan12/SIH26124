from src.detection.vehicle_detector import VehicleDetector
import cv2

model = VehicleDetector(model_name="yolov8n.pt")

cap = cv2.VideoCapture("data/sample/real_road_sample.mp4")

if not cap.isOpened():
    print("ERROR: Could not open video")
    raise SystemExit(1)

for frame_number in range(500):
    ret, frame = cap.read()

    if not ret:
        break

    result = model.detect(frame)

    car_detections = [
        d for d in result["detections"]
        if d["class_name"] in ["car", "motorcycle", "bus", "truck"]
    ]

    if car_detections:
        print()
        print("========== VEHICLE FOUND ==========")
        print("Frame:", frame_number)
        print("Vehicle detections:", car_detections)
        print("===================================")
        break

else:
    print("No vehicle found in first 500 frames.")

cap.release()
