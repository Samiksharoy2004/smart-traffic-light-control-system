import cv2
import numpy as np
import threading
from ultralytics import YOLO
import time
from gpiozero import LED  # For Raspberry Pi GPIO

# Initialize
model = YOLO("yolov8n.pt")  # Fastest model
lanes = {
    "north": {"cam": 0, "count": 0, "emergency": False, "led_green": LED(17), "led_red": LED(27)},
    "south": {"cam": 1, "count": 0, "emergency": False, "led_green": LED(22), "led_red": LED(23)},
    "east": {"cam": 2, "count": 0, "emergency": False, "led_green": LED(24), "led_red": LED(25)},
    "west": {"cam": 3, "count": 0, "emergency": False, "led_green": LED(5), "led_red": LED(6)}
}

def process_camera(lane):
    cap = cv2.VideoCapture(lanes[lane]["cam"])
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Vehicle Counting
        results = model.track(frame, persist=True)
        lanes[lane]["count"] = len([box for box in results[0].boxes if box.cls == 2])  # Class 2 = car
        
        # Emergency Check (Flashing lights)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) | cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        blue_mask = cv2.inRange(hsv, (100, 70, 50), (130, 255, 255))
        lanes[lane]["emergency"] = np.sum(red_mask) > 10000 or np.sum(blue_mask) > 10000

def control_signals():
    while True:
        emergency_lane = next((lane for lane in lanes if lanes[lane]["emergency"]), None)
        
        if emergency_lane:  # Emergency override
            for lane in lanes:
                lanes[lane]["led_green"].off()
                lanes[lane]["led_red"].on()
            lanes[emergency_lane]["led_green"].on()
            lanes[emergency_lane]["led_red"].off()
            time.sleep(10)  # Hold green for 10s
        else:  # Normal operation
            busiest_lane = max(lanes, key=lambda x: lanes[x]["count"])
            green_time = max(5, min(60, lanes[busiest_lane]["count"] * 2))
            
            for lane in lanes:
                lanes[lane]["led_green"].off()
                lanes[lane]["led_red"].on()
            lanes[busiest_lane]["led_green"].on()
            lanes[busiest_lane]["led_red"].off()
            
            time.sleep(green_time)

# Start camera threads
for lane in lanes:
    threading.Thread(target=process_camera, args=(lane,), daemon=True).start()

# Start signal controller
threading.Thread(target=control_signals, daemon=True).start()

# Keep main thread alive
while True:
    time.sleep(1)