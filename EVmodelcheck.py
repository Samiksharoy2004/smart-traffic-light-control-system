from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

# Load both models
vehicle_model = YOLO("yolov8n.pt").to('cuda')  # Standard vehicle detector
emergency_model = YOLO("runs/detect/emergency_v116/weights/best.pt").to('cuda')  # Your custom model
# Class mappings
EMERGENCY_CLASSES = {0: "ambulance", 1: "police", 2: "fire_truck"}  # From your custom model
STANDARD_VEHICLES = {2: "car", 5: "bus", 7: "truck"}  # From yolov8n.pt

# Performance settings
FRAME_SKIP = 2
CONF_THRESHOLD = 0.5
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Vehicle tracking
vehicle_history = set()  # Tracks all unique vehicles ever seen
current_frame_vehicles = set()  # Tracks vehicles in current frame
max_id = 0  # To ensure unique IDs for emergency vehicles when not tracked

cap = cv2.VideoCapture("C:/Users/Tony/Desktop/A-Z/Mini Project/AITLC/Object-Detection-101/Videos/Tony.mp4")

# Window setup
cv2.namedWindow("Traffic Monitoring", cv2.WINDOW_NORMAL)

while cap.isOpened():
    for _ in range(FRAME_SKIP):
        ret, frame = cap.read()
    if not ret:
        break
    
    # Run both detectors in parallel
    vehicle_results = vehicle_model.track(frame, persist=True, conf=CONF_THRESHOLD)
    emergency_results = emergency_model.track(frame, conf=CONF_THRESHOLD)
    
    current_frame_vehicles.clear()
    emergency_detected = False
    
    # Process standard vehicle detections
    for box in vehicle_results[0].boxes:
        cls_id = int(box.cls)
        if cls_id not in STANDARD_VEHICLES:
            continue
            
        # Get or assign ID
        if hasattr(box, 'id'):
            vehicle_id = box.id.item()
        else:
            continue  # Skip if no ID and not an emergency vehicle
            
        current_frame_vehicles.add(vehicle_id)
        vehicle_history.add(vehicle_id)
        
        # Update max_id for emergency vehicle ID assignment
        if vehicle_id > max_id:
            max_id = vehicle_id
            
        # Visualize standard vehicles (green boxes)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    
    # Process emergency vehicle detections
    for box in emergency_results[0].boxes:
        cls_id = int(box.cls)
        if cls_id not in EMERGENCY_CLASSES:
            continue
            
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        
        # Assign new ID if not tracked
        if hasattr(box, 'id'):
            vehicle_id = box.id.item()
        else:
            max_id += 1
            vehicle_id = max_id
            
        current_frame_vehicles.add(vehicle_id)
        vehicle_history.add(vehicle_id)
        
        # Visual confirmation (purple boxes)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 3)
        cv2.putText(frame, 
                   f"{EMERGENCY_CLASSES[cls_id]} {conf:.2f}",
                   (x1,y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.9, (255,0,255), 2)
        emergency_detected = True
    
    # Display info
    cv2.putText(frame, f"Current Vehicles: {len(current_frame_vehicles)}", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Total Unique: {len(vehicle_history)}", (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    
    if emergency_detected:
        cv2.putText(frame, "EMERGENCY DETECTED!", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    # Resize display
    resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    cv2.imshow("Traffic Monitoring", resized_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
