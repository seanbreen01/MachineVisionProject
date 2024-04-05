import cv2
import numpy as np
import tensorflow.lite as tflite

labels = ['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor', 'traffic_light', 'traffic sign', 'train']

# Initialize the TensorFlow Lite interpreter
interpreter = tflite.Interpreter(model_path='best_mydetector-fp16.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define a function to preprocess the frame
def preprocess_frame(frame):
    # Adjust these preprocessing steps as per your model requirements
    frame_resized = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    frame_normalized = frame_resized / 255.0  # Normalize if your model expects this
    return np.expand_dims(frame_normalized, axis=0).astype(np.float32)

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


frame = cv2.imread('testImage.jpg')
frame = cv2.resize(frame, (640, 640))

copyFrame = frame.copy()
copyFrame = cv2.resize(copyFrame, (640, 640))
# Preprocess the frame
input_data = preprocess_frame(copyFrame)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

"""Output data"""
output_data = interpreter.get_tensor(output_details[0]['index'])  # get tensor  x(1, 25200, 7)
xyxy, classes, scores = YOLOdetect(output_data) #boxes(x,y,x,y), classes(int), scores(float) [25200]

# Convert xyxy to the format expected by OpenCV NMS and prepare scores
boxes = []
confidences = []
for i in range(len(scores)):
    if scores[i] > 0.75:
        H = frame.shape[0]
        W = frame.shape[1]
        xmin = max(1, (xyxy[0][i] * W))
        ymin = max(1, (xyxy[1][i] * H))
        xmax = min(H, (xyxy[2][i] * W))
        ymax = min(W, (xyxy[3][i] * H))
        boxes.append([xmin, ymin, int(xmax - xmin), int(ymax - ymin)])  # Format: [x, y, width, height]
        confidences.append(float(scores[i]))

# Apply Non-Maximum Suppression
# nms_threshold = 0.4  # NMS threshold, can be adjusted
# indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.75, nms_threshold=nms_threshold)

for i in range(len(scores)):
    if scores[i] > 0.75:  # Keep the confidence threshold
        H = frame.shape[0]
        W = frame.shape[1]
        xmin = max(1, int(xyxy[0][i] * W))
        ymin = max(1, int(xyxy[1][i] * H))
        xmax = min(W, int(xyxy[2][i] * W))
        ymax = min(H, int(xyxy[3][i] * H))

        # Calculate the bottom-right corner of the rectangle
        bottom_right_x = xmin + (xmax - xmin)
        bottom_right_y = ymin + (ymax - ymin)

        cv2.rectangle(frame, (xmin, ymin), (bottom_right_x, bottom_right_y), (10, 255, 0), 2)
        cv2.putText(frame, labels[classes[i]], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
        formatted_confidence = "{:.2f}".format(scores[i])
        cv2.putText(frame, formatted_confidence, (xmin + 100, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

cv2.namedWindow('detect_result', cv2.WINDOW_NORMAL)
cv2.imshow('detect_result', frame)
cv2.waitKey(0)


