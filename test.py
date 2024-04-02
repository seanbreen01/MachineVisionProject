import cv2
import numpy as np
import tensorflow.lite as tflite

def gstreamer_pipeline(sensor_id=0, sensor_mode=3, capture_width=640, capture_height=480, display_width=640, display_height=480, framerate=30, flip_method=2):
    return (
        f'nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! '
        f'video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, '
        f'format=(string)NV12, framerate=(fraction){framerate}/1 ! '
        f'nvvidconv flip-method={flip_method} ! '
        f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! '
        f'videoconvert ! '
        f'video/x-raw, format=(string)BGR ! appsink'
    )



# Initialize the TensorFlow Lite interpreter
interpreter = tflite.Interpreter(model_path='your_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define a function to preprocess the frame
def preprocess_frame(frame):
    # Adjust these preprocessing steps as per your model requirements
    frame_resized = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    frame_normalized = frame_resized / 255.0  # Normalize if your model expects this
    return np.expand_dims(frame_normalized, axis=0).astype(np.float32)

# Define a function for postprocessing
def postprocess_frame(frame, output_data):
    # Adjust this postprocessing as per your model's output format
    for detection in output_data[0]:
        # Draw detection boxes on the frame
        pass

pipeline = gstreamer_pipeline()
# Initialize video capture with GStreamer pipeline (adjust this as needed)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
print('cap declared')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

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


