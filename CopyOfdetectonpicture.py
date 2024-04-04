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

for i in range(len(scores)):
    if ((scores[i] > 0.75) and (scores[i] <= 1.0)):
        H = frame.shape[0]
        W = frame.shape[1]
        xmin = int(max(1,(xyxy[0][i] * W)))
        ymin = int(max(1,(xyxy[1][i] * H)))
        xmax = int(min(H,(xyxy[2][i] * W)))
        ymax = int(min(W,(xyxy[3][i] * H)))

        # print('xmin:', xmin)
        # print('ymin:', ymin)   
        # print('xmax:', xmax)
        # print('ymax:', ymax)
        # print('class:', classes[i])
        print('score:', scores[i])
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        cv2.putText(frame, labels[classes[i]], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(frame, str(scores[i]), (xmin+150, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        print('rectangle drawn')


cv2.namedWindow('detect_result', cv2.WINDOW_NORMAL)
cv2.imshow('detect_result', frame)
cv2.waitKey(0)
