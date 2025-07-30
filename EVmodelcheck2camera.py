from ultralytics import YOLO
import cv2
from collections import defaultdict

# Load models
emergency_model = YOLO("runs/detect/emergency_v116/weights/best.pt").to('cuda')
indian_vehicle_model = YOLO("runs/detect/emergency_v119/weights/best.pt").to('cuda')

# Class mappings
EMERGENCY_CLASSES = {0: "ambulance", 1: "police", 2: "fire_truck"}
INDIAN_VEHICLES = {0: "vehicles"}

# Initialize webcam (use 0 for default camera, or 1/2 for external cameras)
cap = cv2.VideoCapture(0)  # ← Changed to webcam input
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Optional: Set resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detectors
    emergency_results = emergency_model.track(frame, conf=0.5)
    indian_results = indian_vehicle_model.track(frame, conf=0.5)
    
    # Process detections (same as your existing code)
    for box in emergency_results[0].boxes:
        cls_id = int(box.cls)
        if cls_id in EMERGENCY_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, "EMERGENCY", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    
    for box in indian_results[0].boxes:
        cls_id = int(box.cls)
        if cls_id in INDIAN_VEHICLES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
    
    # Display
    cv2.imshow("Live Traffic Monitoring", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()