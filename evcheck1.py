from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

# Initialize model with GPU if available
model = YOLO("yolov8n.pt")  # Fastest model
EMERGENCY_CLASSES = {3: "ambulance", 4: "police", 5: "fire_truck"}
VEHICLE_CLASSES = {2: "car", 6: "truck", 7: "bus"}

# Performance settings
FRAME_SKIP = 2  # Set to 2 for 2x speed
CONF_THRESHOLD = 0.5  # Lower for faster detection
DISPLAY_SCALE = 0.8  # Scale down display if needed

# Vehicle tracking
vehicle_counter = defaultdict(int)
current_frame_ids = set()

cap = cv2.VideoCapture("C:/Users/Tony/Desktop/A-Z/Mini Project/ITSS/Tony4.mp4")
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate display size while maintaining aspect ratio
display_width = int(original_width * DISPLAY_SCALE)
display_height = int(original_height * DISPLAY_SCALE)

while cap.isOpened():
    for _ in range(FRAME_SKIP):
        ret, frame = cap.read()
    if not ret:
        break
    
    # Process at full resolution for accuracy
    results = model.track(frame, persist=True, conf=CONF_THRESHOLD)
    
    current_frame_ids.clear()
    emergency_detected = False
    
    for box in results[0].boxes:
        if not hasattr(box, 'id'): continue
        
        cls_id = int(box.cls)
        vehicle_id = box.id.item()
        current_frame_ids.add(vehicle_id)
        
        # Count vehicles
        if cls_id in VEHICLE_CLASSES or cls_id in EMERGENCY_CLASSES:
            vehicle_counter[vehicle_id] += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)    
        # Emergency detection
        if cls_id in EMERGENCY_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]
            
            # Fast flashing light check
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            red_mask = cv2.inRange(hsv, (0,100,100), (10,255,255)) | cv2.inRange(hsv, (170,100,100), (180,255,255))
            if np.sum(red_mask) > 500:  # Emergency confirmed
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 3)
                cv2.putText(frame, EMERGENCY_CLASSES[cls_id], (x1,y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 2)
                emergency_detected = True
    
    # Display info directly on frame (not terminal)
    cv2.putText(frame, f"Live Vehicles: {len(current_frame_ids)}", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Total Unique: {len(vehicle_counter)}", (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    
    if emergency_detected:
        cv2.putText(frame, "EMERGENCY VEHICLE DETECTED!", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    # Resize for display while maintaining aspect ratio
    resized_frame = cv2.resize(frame, (display_width, display_height))
    
    # Create resizable window
    cv2.namedWindow("Traffic Monitoring", cv2.WINDOW_NORMAL)
    cv2.imshow("Traffic Monitoring", resized_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()