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

def postprocess_frame(frame, output_data):
    # Assuming output_data[0] contains detection boxes with the format [ymin, xmin, ymax, xmax]
    # and output_data[1] contains confidence scores for each detection
    print('output 1:', output_data[0][0])
    print('output2:', output_data[1][0])
    detection_boxes = output_data[0][0]
    confidence_scores = output_data[0][1]

    for i in range(len(detection_boxes)):
        box = detection_boxes[i]
        confidence = confidence_scores[i]

        # Check if the detection is confident enough
        if confidence > 0.5:  # Adjust this threshold as needed
            frame = draw_detection(frame, box)

    return frame


pipeline = gstreamer_pipeline()
# Initialize video capture with GStreamer pipeline (adjust this as needed)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
print('cap declared')
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 640))
    cv2.imshow('frame', frame)
    if not ret:
        break

    # Preprocess the frame
    #input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], frame)

    # cv2.imshow('preprocessed frame', input_data)

    # Run inference
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index']) # Bounding box coordinates of detected objects
    print('boxes:', boxes)
    classes = interpreter.get_tensor(output_details[1]['index']) # Class index of detected objects
    print('classes:', classes)
    scores = interpreter.get_tensor(output_details[scores_idx]['index']) # Confidence of detected objects
    print('scores:', scores)


    # Retrieve detection results
    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

    # Postprocess and display the frame
    postprocess_frame(frame, output_data)
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()


