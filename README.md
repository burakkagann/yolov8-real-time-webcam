# Real-Time Object Detection with YOLOv8 and Webcam

This project implements real-time object detection using YOLOv8 and your laptop's webcam. It can detect and classify various objects in real-time video feed.

## Features
- Real-time object detection using YOLOv8
- Webcam integration (supports external webcams)
- Display of detection boxes and labels
- Confidence score display
- Support for multiple object classes

## Setup Instructions

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the detection script:**
   ```bash
   python detect.py
   ```

## Usage
- The script will try to use your external webcam first (index 1), then fallback to the default (index 0).
- The webcam feed will show detected objects with bounding boxes and labels.
- Confidence scores are displayed for each detection.
- Press 'q' to quit the application.

## Camera Selection & Troubleshooting
- If you have multiple cameras, the script tries to use the external webcam first.
- If you want to use a specific camera, change the index in the `try_camera()` function in `detect.py`.
- If you see errors like "Could not open any camera!", make sure your webcam is connected and not used by another application.
- If the window does not appear, check if your system blocks camera access for Python or OpenCV.
- You can test your webcam in the Windows Camera app to ensure it works.

## How the Code Works (Annotated)

- The script tries to open camera index 1 (external webcam), then index 0 (default/built-in webcam).
- It loads the YOLOv8 model and sets the camera resolution to 1280x720 (HD).
- For each frame from the webcam:
  - Runs YOLOv8 object detection.
  - Draws bounding boxes and labels for detected objects.
  - Displays the frame in a window.
- The detection loop continues until you press 'q'.
- All resources are released on exit.

## Project Structure
```
yolov8-real-time-webcam/
├── detect.py           # Main detection script (fully annotated)
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── .gitignore          # Git ignore file
```

## Example Output
When you run the script, you should see a window displaying your webcam feed with bounding boxes and labels for detected objects (e.g., person, bottle, chair, etc.).

---

**References:**
- [YOLOv8 Webcam Detection Guide](https://yolov8.org/yolov8-webcam/)
- [Real-time Object Detection with YOLO and Webcam (Medium)](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)
- [OpenCV VideoCapture Documentation](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) 