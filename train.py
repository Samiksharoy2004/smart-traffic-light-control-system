from ultralytics import YOLO

# Load a pretrained model (YOLOv8n = small version)
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="C:/Users/Tony/Desktop/A-Z/Mini Project/ITSS/data.yaml",  # Path to your dataset YAML
    epochs=50,                         # Number of training cycles
    imgsz=640,                         # Image size
    batch=16,                           # Reduce if your PC is slow
    name="traffic_model"               # Save folder name
)