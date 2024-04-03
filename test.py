import cv2
import numpy as np
import tensorflow.lite as tflite

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

print('input',input_details)
print('output', output_details)

outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

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


pipeline = gstreamer_pipeline()
# Initialize video capture with GStreamer pipeline (adjust this as needed)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
print('cap declared')
while cap.isOpened():
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (640, 640))
    cv2.imshow('frame', frame)
    if not ret:
        break

    # Preprocess the frame
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # cv2.imshow('preprocessed frame', input_data)

    # Run inference
    interpreter.invoke()

    # boxes = interpreter.get_tensor(output_details[0]['index']) # Bounding box coordinates of detected objects
    # print('boxes:', boxes)
    # classes = interpreter.get_tensor(output_details[0]['index']) # Class index of detected objects
    # print('classes:', classes)
    # scores = interpreter.get_tensor(output_details[scores_idx]['index']) # Confidence of detected objects
    # print('scores:', scores)


    # # Retrieve detection results
    # output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

    # Inside your while loop, after the model inference:
    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    boxes, classes, scores = output_data[boxes_idx], output_data[classes_idx], output_data[scores_idx]
    postprocess_frame(frame, boxes, scores, classes)

    # Postprocess and display the frame
    # postprocess_frame(frame, output_data)
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()


