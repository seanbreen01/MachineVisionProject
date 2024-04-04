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

H, W = frame.shape[:2]  # Get frame height and width


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

# Convert xyxy format to the format expected by cv2.dnn.NMSBoxes (x, y, width, height)
boxes_for_nms = []
for i in range(len(scores)):
    if scores[i] > 0.2:  # Adjust this threshold as needed
        x_center, y_center, box_width, box_height = xyxy[0][i], xyxy[1][i], xyxy[2][i] - xyxy[0][i], xyxy[3][i] - xyxy[1][i]
        boxes_for_nms.append([x_center, y_center, box_width, box_height])

# Apply Non-Maximum Suppression
indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores, score_threshold=0.2, nms_threshold=0.4)  # Adjust these thresholds as needed

# Now draw the boxes and annotations using the indices returned by NMS
for i in indices:
    i = i[0]  # NMSBoxes returns a list of lists
    xmin = int(max(1, (xyxy[0][i] * W)))
    ymin = int(max(1, (xyxy[1][i] * H)))
    xmax = int(min(H, (xyxy[2][i] * W)))
    ymax = int(min(W, (xyxy[3][i] * H)))

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
    cv2.putText(frame, labels[classes[i]], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    cv2.putText(frame, str(scores[i]), (xmin + 150, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


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
