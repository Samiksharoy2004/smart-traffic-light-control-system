from ultralytics import YOLO
import cv2
import numpy as np
model = YOLO("yolov8n.pt")  # Nano version (fastest)

# model = YOLO("runs/detect/traffic_model/weights/best.pt")
EMERGENCY_CLASSES = {3: "ambulance", 4: "police", 5: "fire_truck"}  # Update with your class IDs
VEHICLE_CLASSES = {2: "car", 6: "truck", 7: "bus", 3: "ambulance"}  # Include emergency vehicles

cap = cv2.VideoCapture("C:/Users/Tony/Desktop/A-Z/Mini Project/ITSS/Tony4.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Reset counters for this frame
    vehicle_count = 0
    emergency_count = 0
    
    results = model.track(frame, persist=True)
    
    for box in results[0].boxes:
        cls_id = int(box.cls)
        
        # Count all vehicles (including emergency vehicles)
        if cls_id in VEHICLE_CLASSES:
            vehicle_count += 1
            
            # Check if emergency vehicle
            if cls_id in EMERGENCY_CLASSES:
                emergency_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Optional: Flashing light verification
                hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)  # Only check vehicle ROI
                red_mask = cv2.inRange(hsv, (0,70,50), (10,255,255)) | cv2.inRange(hsv, (170,70,50), (180,255,255))
                if np.sum(red_mask) > 1000:  # Lower threshold since we're checking smaller area
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 3)
                    cv2.putText(frame, "EMERGENCY", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 2)
    
    # Display frame counts (not cumulative)
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"Current Vehicles: {vehicle_count}", (20, 540), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
    cv2.putText(annotated_frame, f"Emergencies: {emergency_count}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    resized_frame = cv2.resize(annotated_frame, (1920, 1080))

    cv2.namedWindow("Video Detection", cv2.WINDOW_NORMAL)  # Makes window resizable
    cv2.resizeWindow("Video Detection", 1920, 1080)  # Initial size
    
    cv2.imshow("Video Detection", resized_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()