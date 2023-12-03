from ultralytics import YOLO


# Load model 
model = YOLO("yolov8n.yaml")

# Use the model 
results = model.train(data="local_config.yaml", epochs=2)
