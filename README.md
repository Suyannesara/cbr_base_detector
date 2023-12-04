### 🤖 CBR Base Detector

This repository aims to detect the CBR base used in OBR.

_Use the base_images folder to get the cbr_base images to test these samples_

#### 1. Detection Model

The weights generated during training are available in the `/detect/weights` folder.
The `detect/config` folder contains the configuration yamls, in case you wish run this in colab, there is a config for it as well.

#### 1.1. Dependencies

- Ultralytics
- OpenCV (cv2)

You can install the dependencies using:

```bash
pip install -U ultralytics opencv-python
```

#### 1.2. Test the Model

To test the model, run the following command:

```bash
python predict_video.py
```

A generated video will be found in `./videos` folder with the detections running.
This folder also contains a base video with the cbr_base appearing. You can use it, or, make your own video and substitute the one using same name and extension "base.mp4" in order to test it.

### 2. Roboflow Integration

The `/roboflow` folder contains a sample using the Roboflow API directly. 
So that, it uses a different model trained by **roboflow**.
If you wish to test it, fllow the steps:

#### 2.1. Dependencies used

- python-dotenv
- Roboflow
- OpenCV (cv2)
- NumPy
- Requests

Install the dependencies using:

```bash
pip install python-dotenv roboflow opencv-python numpy requests
```

#### 2.2. Running Roboflow Prediction

2. Create a `.env` file in the root and fill in the values:

```env
ROBOFLOW_API_KEY="YOUR_ROBOFLOW_API_KEY"
ROBOFLOW_MODEL="cbr_base_detector_test/version_you_wish"
ROBOFLOW_PROJECT="cbr_base_detector_test"
```

3. Run the Roboflow prediction script:

```bash
python roboflow_predict.py
```