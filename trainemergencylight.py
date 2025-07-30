from ultralytics import YOLO
import torch
import multiprocessing

def main():
    # Initialize model
    model = YOLO("yolov8n.pt")
    
    # Train the model
    results = model.train(
        data="C:/Users/Tony/Desktop/A-Z/Mini Project/ITSS/Indian Dataset/DB/data.yaml",
        epochs=250,
        imgsz=640,
        batch=64,
        device=0 if torch.cuda.is_available() else None,
        name="emergency_v1",
        workers=4,  # Reduce workers if needed
        amp=True,   # Enable mixed precision
        single_cls=False,
        optimizer="auto"
    )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()