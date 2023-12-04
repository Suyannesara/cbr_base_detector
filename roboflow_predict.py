from roboflow import Roboflow
import os
import cv2
import base64
import numpy as np
import requests
from dotenv import load_dotenv

#  Load variables
load_dotenv()
KEY = os.environ.get("ROBOFLOW_API_KEY")
MODEL = os.environ.get("ROBOFLOW_MODEL")
PROJECT = os.environ.get("ROBOFLOW_PROJECT")


# Connect to roboflow api
rf = Roboflow(api_key=KEY)
project = rf.workspace().project(PROJECT)
model = project.version(2).model

upload_url = "".join([
    "https://detect.roboflow.com/",
    MODEL,
    "?api_key=",
    KEY,
    "&format=image",
    "&stroke=5"
])

# Video capture
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

def infer():
    # Get the current image from the webcam
    ret, img = cap.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = 400 / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    try:
        # Get prediction from Roboflow Infer API
        resp = requests.post(upload_url, data=img_str, headers={
            "Content-Type": "application/x-www-form-urlencoded"
        }, stream=True)

        # Check if the request was successful
        resp.raise_for_status()

        # Parse result image
        image = np.asarray(bytearray(resp.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        return image

    except requests.exceptions.RequestException as e:
        print(f"Error making request to Roboflow Infer API: {e}")
        return None

while cap.isOpened():
    image = infer()

    out.write(image)
    cv2.imshow('image', image)

    # On "q" keypress, exit
    if cv2.waitKey(1) == 27:
        break
        
# Release video and VideoWriter objects
cap.release()
out.release()

cv2.destroyAllWindows()