from ultralytics import YOLO
import cv2
import math
import time

# Function to try opening a camera at a given index
# Returns (True, cap) if successful, (False, None) otherwise
def try_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Use DirectShow backend for better Windows compatibility
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            return True, cap
    return False, None

# Main function for real-time object detection
def main():
    print("Trying to open cameras...")
    
    # Try to open external webcam first (index 1), then default (index 0)
    success, cap = try_camera(1)
    if not success:
        print("Trying camera index 0...")
        success, cap = try_camera(0)
    
    if not success:
        print("Error: Could not open any camera!")
        return
    
    print("Camera opened successfully!")

    # Load YOLOv8 model (nano version for speed)
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    # Set camera resolution to 1280x720 (HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Wait for camera to initialize
    time.sleep(2)

    # List of class names YOLOv8 is trained to detect
    classNames = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
        "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]

    print("Starting detection loop...")
    print("Press 'q' to quit")
    
    while True:
        # Read a frame from the webcam
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Run YOLOv8 object detection on the frame
        results = model(img, stream=True)

        # Process each detection result
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Get bounding box coordinates (top-left and bottom-right)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw bounding box on the frame
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Get confidence score (rounded to 2 decimals)
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # Get class name from class index
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # Prepare text for label (class name and confidence)
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                # Put label text on the frame
                cv2.putText(img, f"{class_name} {confidence}", org, font, fontScale, color, thickness)

        # Show the frame with detections in a window
        cv2.imshow('Webcam', img)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 