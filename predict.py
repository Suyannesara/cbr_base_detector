from ultralytics import YOLO

# Test model 
model = YOLO('./weights/best.pt')
model.predict(source="0", show=True, conf=0.5)
