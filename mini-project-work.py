import cv2
import numpy as np

from jetcam.csi_camera import CSICamera

import torch
import torchvision
import tortchvision.transforms as transforms

device = torch.device('cuda')

# Take in CSI Camera input 
camera = CSICamera(width=224, height=224, capture_device=0)
camera.running = True
print("camera created")

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




while True:
    frame = camera.read()
    # Preprocess the frame for the model
    input_tensor = preprocess(frame)

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


        

