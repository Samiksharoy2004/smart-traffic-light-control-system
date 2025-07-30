from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

# Load all three models
# vehicle_model = YOLO("yolov8s.pt").to('cuda')  # Standard vehicle detector
emergency_model = YOLO("runs/detect/emergency_v116/weights/best.pt").to('cuda')  # Emergency vehicles
indian_vehicle_model = YOLO("runs/detect/emergency_v119/weights/best.pt").to('cuda')  # New Indian vehicle model

# Class mappings
EMERGENCY_CLASSES = {0: "ambulance", 1: "police", 2: "fire_truck"}
# STANDARD_VEHICLES = {2: "car", 5: "bus", 7: "truck"}
INDIAN_VEHICLES = {0: "vehicles"}  # Update with your actual classes

# Performance settings
FRAME_SKIP = 2
CONF_THRESHOLD = 0.5
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Vehicle tracking
vehicle_history = set()
current_frame_vehicles = set()
max_id = 0

cap = cv2.VideoCapture("C:/Users/Tony/Desktop/A-Z/Mini Project/AITLC/Object-Detection-101/Videos/Tony1.mp4")
cv2.namedWindow("Traffic Monitoring", cv2.WINDOW_NORMAL)

while cap.isOpened():
    for _ in range(FRAME_SKIP):
        ret, frame = cap.read()
    if not ret:
        break
    
    # Run all detectors
    # vehicle_results = vehicle_model.track(frame, persist=True, conf=CONF_THRESHOLD)
    emergency_results = emergency_model.track(frame, conf=CONF_THRESHOLD)
    indian_results = indian_vehicle_model.track(frame, conf=CONF_THRESHOLD)
    
    current_frame_vehicles.clear()
    emergency_detected = False
    
    # # ===== 1. Process STANDARD vehicles (Green boxes) =====
    # for box in vehicle_results[0].boxes:
    #     cls_id = int(box.cls)
    #     if cls_id not in STANDARD_VEHICLES:
    #         continue
            
    #     if hasattr(box, 'id'):
    #         vehicle_id = box.id.item()
    #         current_frame_vehicles.add(vehicle_id)
    #         vehicle_history.add(vehicle_id)
    #         max_id = max(max_id, vehicle_id)
            
    #         x1, y1, x2, y2 = map(int, box.xyxy[0])
    #         cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)  # Green
    
    # ===== 2. Process EMERGENCY vehicles (Purple boxes) =====
    for box in emergency_results[0].boxes:
        cls_id = int(box.cls)
        if cls_id not in EMERGENCY_CLASSES:
            continue
            
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        
        vehicle_id = box.id.item() if hasattr(box, 'id') else (max_id := max_id + 1)
        current_frame_vehicles.add(vehicle_id)
        vehicle_history.add(vehicle_id)
        
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 3)  # Purple
        cv2.putText(frame, f"{EMERGENCY_CLASSES[cls_id]} {conf:.2f}", 
                   (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 2)
        emergency_detected = True
    
    # ===== 3. Process INDIAN vehicles (Orange boxes) =====
    for box in indian_results[0].boxes:
        cls_id = int(box.cls)
        if cls_id not in INDIAN_VEHICLES:
            continue
            
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        
        vehicle_id = box.id.item() if hasattr(box, 'id') else (max_id := max_id + 1)
        current_frame_vehicles.add(vehicle_id)
        vehicle_history.add(vehicle_id)
        
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,165,255), 2)  # Orange
        # cv2.putText(frame, f"{INDIAN_VEHICLES[cls_id]}", 
        #            (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
    
    # ===== Display Stats =====
    cv2.putText(frame, f"Current Vehicles: {len(current_frame_vehicles)}", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Total Unique: {len(vehicle_history)}", (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    
    if emergency_detected:
        cv2.putText(frame, "EMERGENCY DETECTED!", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    cv2.imshow("Traffic Monitoring", resized_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
