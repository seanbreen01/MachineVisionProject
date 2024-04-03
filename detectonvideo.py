import cv2
import numpy as np
import tensorflow.lite as tflite

# Define a function to preprocess the frame
def preprocess_frame(frame):
    # Adjust these preprocessing steps as per your model requirements
    frame_resized = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    frame_normalized = frame_resized / 255.0  # Normalize if your model expects this
    return np.expand_dims(frame_normalized, axis=0).astype(np.float32)

def draw_detection(frame, box, color=(255, 0, 0), thickness=2):
    """Draw a single detection box on the frame."""
    height, width = frame.shape[:2]
    ymin, xmin, ymax, xmax = box
    start_point = (int(xmin * width), int(ymin * height))
    end_point = (int(xmax * width), int(ymax * height))
    frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
    return frame

def postprocess_frame(frame, boxes, scores, classes, threshold=0.01):
    # boxes: Bounding box coordinates of detected objects
    # scores: Confidence of detected objects
    # classes: Class index of detected objects
    for i in range(len(scores)):
        if scores[i] > threshold:
            box = boxes[i]  # y_min, x_min, y_max, x_max
            class_id = int(classes[i])
            confidence = scores[i]
            label = labels[class_id] if class_id < len(labels) else 'Unknown'
            frame = draw_detection(frame, box)
            # Optionally, add text for label and confidence score
            label_text = f'{label}: {confidence:.2f}'
            start_point = (int(box[1] * frame.shape[1]), int(box[0] * frame.shape[0]))  # x_min, y_min
            cv2.putText(frame, label_text, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame


def classFilter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

def YOLOdetect(output_data):  # input = interpreter, output is boxes(xyxy), classes, scores
    output_data = output_data[0]                # x(1, 25200, 7) to x(25200, 7)
    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    return xyxy, classes, scores  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]


# TensorFlow Lite model and labels
model_path = 'best_mydetector-fp16.tflite'
labels = ['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor', 'traffic_light', 'traffic sign', 'train']

# Initialize the TensorFlow Lite interpreter
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the video
video_path = 'Testvideo.mp4'
cap = cv2.VideoCapture(video_path)

# Get original video properties
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, original_fps, (original_width, original_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for the model input
    frame_resized = cv2.resize(frame, (640, 640))

    # Preprocess the frame
    input_data = preprocess_frame(frame_resized)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    xyxy, classes, scores = YOLOdetect(output_data)

    # Scale bounding box coordinates back to original frame size
    for i in range(len(scores)):
        if scores[i] > 0.25:
            ymin, xmin, ymax, xmax = xyxy[0][i], xyxy[1][i], xyxy[2][i], xyxy[3][i]
            xmin, xmax = xmin * original_width / 640, xmax * original_width / 640
            ymin, ymax = ymin * original_height / 640, ymax * original_height / 640
            frame = draw_detection(frame, (ymin, xmin, ymax, xmax))

    # Save the frame with bounding boxes
    out.write(frame)

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()