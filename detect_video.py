from ultralytics import YOLO
import cv2

# model = YOLO("runs/detect/traffic_model/weights/best.pt")
video_path = "C:/Users/Tony/Desktop/A-Z/Mini Project/ITSS/Tony4.mp4"
model = YOLO("yolov8n.pt")  # Nano version (fastest)
# model = YOLO("yolov8s.pt")  # Small version (balanced)
cap = cv2.VideoCapture(video_path)

# results = model.track(frame)  # Your detection
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()

    resized_frame = cv2.resize(annotated_frame, (1280, 720))

    cv2.namedWindow("Video Detection", cv2.WINDOW_NORMAL)  # Makes window resizable
    cv2.resizeWindow("Video Detection", 1280, 720)  # Initial size
    
    cv2.imshow("Video Detection", resized_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
model.save("output/results.mp4")