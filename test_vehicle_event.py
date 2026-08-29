from src.detection.vehicle_detector import VehicleDetector
from src.events.generator import EventGenerator
import cv2

model = VehicleDetector(model_name="yolov8n.pt")

cap = cv2.VideoCapture("data/sample/real_road_sample.mp4")

if not cap.isOpened():
    print("ERROR: Could not open video")
    raise SystemExit(1)

target_frame = 199

for frame_number in range(target_frame + 1):
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame", frame_number)
        raise SystemExit(1)

    if frame_number == target_frame:
        result = model.detect(frame)

cap.release()

print("========== YOLO RESULT ==========")
print("Detections:", result["detections"])

event_generator = EventGenerator(source_id="TEST_VEHICLE")

events = event_generator.process_frame_detections(
    frame_index=target_frame,
    total_frames=500,
    vehicle_detections=result["detections"],
    damage_detections=[],
    video_fps=25
)

print()
print("========== EVENT GENERATOR RESULT ==========")
print("Events generated:", len(events))

for event in events:
    print(event.to_dict())

print("============================================")
