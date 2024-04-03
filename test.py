import torch
import cv2
from time import time
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


print('Loading YOLOv5 model...')
# Load the YOLOv5 model
model = torch.load('best_mydetector.pt')
model.eval()
print('Model loaded successfully.')

# Function to apply object detection on a frame
def detect(frame):
    # Preprocess the frame for YOLOv5 model
    # Convert frame to a tensor, resize it, and normalize it
    # Note: You might need to adjust the resizing and normalization to match your model's training configuration

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame = torch.from_numpy(frame).to('cuda' if torch.cuda.is_available() else 'cpu').float()
    frame /= 255.0  # Normalize
    frame = frame.permute(2, 0, 1)  # Rearrange dimensions
    frame = frame.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        results = model(frame)

    # Process results
    # YOLOv5's output is a tensor with shape: [number of detections, 6]
    # Each detection has the format: [x_center, y_center, width, height, confidence, class]
    detections = results[0].detach().cpu().numpy()

    # Scale coordinates to original frame size
    frame_height, frame_width = frame.shape[:2]
    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        x_center, y_center, width, height, confidence, class_id = detection

        x = int((x_center - width / 2) * frame_width)
        y = int((y_center - height / 2) * frame_height)
        w = int(width * frame_width)
        h = int(height * frame_height)

        boxes.append([x, y, w, h])
        confidences.append(float(confidence))
        class_ids.append(int(class_id))

    # Draw bounding boxes and labels on the frame
    for i, box in enumerate(boxes):
        x, y, w, h = box
        label = str(class_ids[i])  # Replace with actual class name if available
        confidence = confidences[i]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    # Capture video from a camera (change '0' to a file path for a video file)
    pipeline = gstreamer_pipeline()
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture(0)

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