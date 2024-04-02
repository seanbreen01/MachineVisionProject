import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import tensorflow as tf



device = torch.device('cuda')

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



# Take in created model
model = torch.load('your_model.pth')
model.eval()
model.cuda()

# TODO include other imports needed to run TFLITE? model

personFrameCount = 0



# Define preprocessing function
# TODO adjust to model requirements
def preprocess(frame):
    # Resize, normalize, etc. as required by your model
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add any other necessary transformations here
    ])
    return transform(frame).cuda()

TF_LITE_MODEL = './compressedModel.tflite'
#TF_LITE_MODEL = './lite-model_efficientdet_lite0_detection_metadata_1.tflite'
# LABEL_MAP = './labelmap.txt'
THRESHOLD = 0.5
RUNTIME_ONLY = True

# TODO replace LABEL_MAP with labels

labels = ['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor', 'traffic_light', 'traffic sign', 'train']


if RUNTIME_ONLY:
    from tflite_runtime.interpreter import Interpreter
    interpreter = Interpreter(model_path=TF_LITE_MODEL)
else:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=TF_LITE_MODEL)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


cap  = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

while cap.isOpened():
    ret, frame = cap.read()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Perform inference
    with torch.no_grad():
        detections = model(input_tensor)

    # Display the result
        

# run detection on camera frame

# Return result to screen
        
    cv2.imshow('Detections Output', frame)

# extra credit: Count people in frame
    
    if detections is not None:
        personFrameCount = 0
        for person in detections:
            personFrameCount += 1

        cv2.putText(frame, f'People in frame: {personFrameCount}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Perople Counting Algorithm Output', frame)


        

TEST_FILE = './test.jpg'
TF_LITE_MODEL = './lite-model_yolo-v5-tflite_tflite_model_1.tflite'
LABEL_MAP = './labelmap.txt'
BOX_THRESHOLD = 0.5
CLASS_THRESHOLD = 0.5
LABEL_SIZE = 0.5
RUNTIME_ONLY = True


import cv2
import numpy as np


if RUNTIME_ONLY:
    from tflite_runtime.interpreter import Interpreter
    interpreter = Interpreter(model_path=TF_LITE_MODEL)
else:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=TF_LITE_MODEL)


interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


_, height, width, _ = interpreter.get_input_details()[0]['shape']


with open(LABEL_MAP, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


img = cv2.imread(TEST_FILE, cv2.IMREAD_COLOR)
IMG_HEIGHT, IMG_WIDTH = img.shape[:2]


pad = round(abs(IMG_WIDTH - IMG_HEIGHT) / 2)
x_pad = pad if IMG_HEIGHT > IMG_WIDTH else 0
y_pad = pad if IMG_WIDTH > IMG_HEIGHT else 0
img_padded = cv2.copyMakeBorder(img, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad,
                                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
IMG_HEIGHT, IMG_WIDTH = img_padded.shape[:2]


img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)
input_data = np.expand_dims(img_resized / 255, axis=0).astype('float32')


interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()


outputs = interpreter.get_tensor(output_details[0]['index'])[0]


boxes = []
box_confidences = []
classes = []
class_probs = []