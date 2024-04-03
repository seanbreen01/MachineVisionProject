import cv2
import numpy as np


def gstreamer_pipeline(sensor_id=0, sensor_mode=3, capture_width=1280, capture_height=720, display_width=640, display_height=640, framerate=30, flip_method=2):
    return (
        f'nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! '
        f'video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, '
        f'format=(string)NV12, framerate=(fraction){framerate}/1 ! '
        f'nvvidconv flip-method={flip_method} ! '
        f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! '
        f'videoconvert ! '
        f'video/x-raw, format=(string)BGR ! appsink'
    )

import torch
import cv2
from time import time
from models.yolo import Model  # Import the YOLO model class

# Load the YOLOv5 model
model = Model(cfg='./Data/BDD.yaml')  # Use the correct configuration file for your model
model.load_state_dict(torch.load('best_mydetector.pt')['model'])  # Replace 'best_mydetector.pt' with your model file
model.eval()

# Function to apply object detection on a frame
def detect(frame):
    results = model(frame)
    return results.render()[0]

def main():
    # Capture video from a camera (change '0' to a file path for a video file)
    pipeline =gstreamer_pipeline()
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        start_time = time()
        frame = detect(frame)
        end_time = time()
        fps = 1 / (end_time - start_time)

        # Display FPS
        cv2.putText(frame, f'FPS: {fps:.2f}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow('YOLOv5 Object Detection', frame)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
