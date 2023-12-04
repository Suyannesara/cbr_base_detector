#  NOTES ON HOW TO TRAIN MODEL ON YOUR MACHINE

from ultralytics import YOLO

# Load model 
model = YOLO("yolov8n.yaml")

# Use the model 
results = model.train(data="./config/local_config.yaml", epochs=2)
