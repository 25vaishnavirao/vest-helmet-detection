from ultralytics import YOLO

# Load a pre-trained model 
model = YOLO("yolo11n.pt")


model.train(
    data="D:/vest_helmet_detection/dataset/data.yaml",
    epochs=50,                      
    imgsz=640,                      # Input image size
    device="cpu"                    # Use "cuda" if you have a GPU
)
