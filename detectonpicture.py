import cv2
import numpy as np
import tensorflow.lite as tflite

labels = ['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor', 'traffic_light', 'traffic sign', 'train']

# Initialize the TensorFlow Lite interpreter
interpreter = tflite.Interpreter(model_path='best_mydetector-fp16.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print('input',input_details)
print('output', output_details)



# Define a function to preprocess the frame
def preprocess_frame(frame):
    # Adjust these preprocessing steps as per your model requirements
    frame_resized = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    frame_normalized = frame_resized / 255.0  # Normalize if your model expects this
    return np.expand_dims(frame_normalized, axis=0).astype(np.float32)

# def draw_detection(frame, box, color=(255, 0, 0), thickness=2):
#     """Draw a single detection box on the frame."""
#     height, width = frame.shape[:2]
#     ymin, xmin, ymax, xmax = box
#     start_point = (int(xmin * width), int(ymin * height))
#     end_point = (int(xmax * width), int(ymax * height))
#     frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
#     return frame

# def postprocess_frame(frame, boxes, scores, classes, threshold=0.2):
#     # boxes: Bounding box coordinates of detected objects
#     # scores: Confidence of detected objects
#     # classes: Class index of detected objects
#     for i in range(len(scores)):
#         if scores[i] > threshold:
#             box = boxes[i]  # y_min, x_min, y_max, x_max
#             class_id = int(classes[i])
#             confidence = scores[i]
#             label = labels[class_id] if class_id < len(labels) else 'Unknown'
#             frame = draw_detection(frame, box)
#             # Optionally, add text for label and confidence score
#             label_text = f'{label}: {confidence:.2f}'
#             start_point = (int(box[1] * frame.shape[1]), int(box[0] * frame.shape[0]))  # x_min, y_min
#             cv2.putText(frame, label_text, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#     return frame


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
#cv2.imshow('frame', frame)

#H, W = frame.shape[:2]  # Get frame height and width


copyFrame = frame.copy()
copyFrame = cv2.resize(copyFrame, (640, 640))
# Preprocess the frame
input_data = preprocess_frame(copyFrame)
interpreter.set_tensor(input_details[0]['index'], input_data)

# cv2.imshow('preprocessed frame', input_data)

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
nms_threshold = 0.4  # NMS threshold, can be adjusted
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.75, nms_threshold=nms_threshold)

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



# Print output details for debugging
# for i, detail in enumerate(output_details):
#     print(f"Output {i}: {detail}")

# Make sure the length of output_data is as expected
#print(f"Number of output tensors: {len(output_data)}")


#boxes, classes, scores = output_data[boxes_idx], output_data[classes_idx], output_data[scores_idx]
# postprocess_frame(frame, boxes, scores, classes)

# Postprocess and display the frame
# frame = postprocess_frame(frame, output_data, scores, classes)
#cv2.imwrite('sean.jpg', frame)
cv2.namedWindow('detect_result', cv2.WINDOW_NORMAL)
cv2.imshow('detect_result', frame)
cv2.waitKey(0)
