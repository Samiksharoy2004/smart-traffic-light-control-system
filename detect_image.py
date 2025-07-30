from ultralytics import YOLO
model = YOLO("C:/Users/Tony/Desktop/A-Z/Mini Project/ITSS/runs/detect/traffic_model/weights/best.pt")
results = model.predict("C:/Users/Tony/Desktop/A-Z/Mini Project/ITSS/highway-traffic.webp", save=True)