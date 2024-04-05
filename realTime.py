import cv2
import numpy as np
import tensorflow.lite as tflite
import os

def gstreamer_pipeline(sensor_id=0, sensor_mode=3, capture_width=1280, capture_height=720, display_width=640, display_height=480, framerate=30, flip_method=2):
    return (
        f'nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! '
        f'video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, '
        f'format=(string)NV12, framerate=(fraction){framerate}/1 ! '
        f'nvvidconv flip-method={flip_method} ! '
        f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! '
        f'videoconvert ! '
        f'video/x-raw, format=(string)BGR ! appsink'
    )

labels = ['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor', 'traffic_light', 'traffic sign', 'train']

# Initialize the TensorFlow Lite interpreter
interpreter = tflite.Interpreter(model_path='best_mydetector-fp16.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    frame_normalized = frame_resized / 255.0  # Normalize if your model expects this
    return np.expand_dims(frame_normalized, axis=0).astype(np.float32)

def classFilter(classdata):
    classes = [] 
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

def detect(output_data):                    # input = interpreter, output is boxes(xyxy), classes, scores
    output_data = output_data[0]                # x(1, 25200, 7) to x(25200, 7)
    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    return xyxy, classes, scores  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]


pipeline = gstreamer_pipeline(flip_method=0)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to fit the model
    frame = cv2.resize(frame, (640, 640))
    copyFrame = frame.copy()

    # Preprocess the frame
    input_data = preprocess_frame(copyFrame)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # Run inference
    output_data = interpreter.get_tensor(output_details[0]['index'])  # get tensor  x(1, 25200, 7)
    xyxy, classes, scores = detect(output_data) 

    # Convert xyxy to the format expected by OpenCV NMS and prepare scores
    boxes = []
    confidences = []
    for i in range(len(scores)):
        if scores[i] > 0.4:
            H = frame.shape[0]
            W = frame.shape[1]
            xmin = max(1, (xyxy[0][i] * W))
            ymin = max(1, (xyxy[1][i] * H))
            xmax = min(H, (xyxy[2][i] * W))
            ymax = min(W, (xyxy[3][i] * H))
            boxes.append([xmin, ymin, int(xmax - xmin), int(ymax - ymin)])  # Format: [x, y, width, height]
            confidences.append(float(scores[i]))

    # Apply Non-Maximum Suppression
    nms_threshold = 0.4  # NMS threshold, can be adjusted
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.4, nms_threshold=nms_threshold)

    # Draw the rectangles and labels for NMS filtered detections
    for i in indices:
        i = i[0]  # Unpack the index
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        # Make sure coordinates are integers
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Calculate the bottom-right corner of the rectangle
        bottom_right_x = x + w
        bottom_right_y = y + h

        cv2.rectangle(frame, (x, y), (bottom_right_x, bottom_right_y), (10, 255, 0), 2)
        cv2.putText(frame, labels[classes[i]], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
        formatted_confidence = "{:.2f}".format(confidences[i])
        cv2.putText(frame, formatted_confidence, (x + 100, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
