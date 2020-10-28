import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.5
FONT_THICKNESS = 2
FRAME_THICKNESS = 2




def read_object_data():
    global net
    global classes
    
    print('Loading YOLO...')
    net = cv2.dnn.readNet('YOLO\yolov3.weights','YOLO\yolov3.cfg') # load the YOLO weights and cfg YOLO(320,320)

    classes = []

    with open('YOLO/coco.names','r') as f:
        classes = f.read().splitlines()
    print('YOLO loaded successfully...')

def detect_objects(frame):
    
    height, width, _ = frame.shape # read the height and width of the frame from the video input
    blob = cv2.dnn.blobFromImage(frame, 1/255, (320,320), swapRB=True, crop=False) # construct a blob from the image
    net.setInput(blob) # The blob object is given as an input to the network
    
    ln = net.getUnconnectedOutLayersNames() # Layer names of the network
    layerOutputs = net.forward(ln) # Get output

    # Identify objects
    boxes = []  # stores the bounding boxes detected from a frame
    confidences = [] # stores the confidence of each bounding box
    class_ids = [] # stores the class id (name) of the detected items

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
            
                x = int(center_x - w/2)
                y = int(center_y - h/2)
            
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # non max suppression
        
    # Draw bounding boxes in the frame
    if len(indexes) > 0: 
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,180,0), FRAME_THICKNESS)
            cv2.putText(frame, label + " " + confidence, (x, y+20), FONT , FONT_SIZE , (255,255,255), FONT_THICKNESS )
    
    return frame
