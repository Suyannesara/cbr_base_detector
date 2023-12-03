from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("cbr_base_detector_test")
model = project.version(2).model

# infer on a local image
print(model.predict("your_image.jpg", confidence=50, overlap=50).json())