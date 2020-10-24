import cv2
import numpy as np
import face_recognition
import pickle
import dlib


#================================ ** FACE RECOGNTION ** =======================================
#=================================== INITIALIZATION ===========================================

#***********face_recognition
KNOWN_FACES_DIR = 'knownFaces'
TOLERANCE = 0.6
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.5
FONT_THICKNESS = 2
FRAME_THICKNESS = 2

#=================================== READ KNOWN FACES =========================================
print('Loading encodings and names of known faces...')

known_faces = [] # stores the encoded data for the known faces
known_names = [] # stores the name string for the known faces

filename = 'faceEncodings'
infile = open(f'{KNOWN_FACES_DIR}/{filename}','rb')
known_faces = pickle.load(infile)
infile.close()

filename = 'faceNames'
infile = open(f'{KNOWN_FACES_DIR}/{filename}','rb')
known_names = pickle.load(infile)
infile.close()
       
print('Encodings and names loaded successfully...')


#==================================== ** YOLO ** ==============================================
#=================================== INITIALIZATION ===========================================
print('Loading YOLO...')

net =  cv2.dnn.readNet('YOLO\yolov3.weights','YOLO\yolov3.cfg') # load the YOLO weights and cfg YOLO(320,320)

classes = [] # stores the class names from the COCO dataset
with open('YOLO\coco.names', 'r') as f:
    classes = f.read().splitlines()

print('YOLO loaded successfully...')


#=============================== ** IMPLEMENT FR AND OD ** ====================================
if dlib.DLIB_USE_CUDA == True:
    print('Using GPU acceleration.')
else:
    print('Not using GPU acceleration. Please set up dlib with CUDA support.')

print('Reading video input from camera 0 (webcam)')
vid = cv2.VideoCapture(0)

while True:
    success, frame = vid.read()
    if(success):
        # ===================== OB part ===================
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
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,180,0), 2)
                cv2.putText(frame, label + " " + confidence, (x, y+20), FONT , FONT_SIZE , (255,255,255), 2 )
    
        
        # ================== FR part =====================

        locations = face_recognition.face_locations(frame,1,'cnn') # find the location of faces detected in a frame
        encodings = face_recognition.face_encodings(frame, locations) # encode the faces detected
        
        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
            match = None
            if True in results:
                match = known_names[results.index(True)]
                
                #face_location rectangle
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])
                color = (0,180,0)
                cv2.rectangle(frame, top_left, bottom_right, color, FRAME_THICKNESS)
                
                #face_name rectangle and text
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)
                cv2.rectangle(frame, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(frame, match, (face_location[3] + 10, face_location[2] + 15), FONT , FONT_SIZE , (255,255,255), FONT_THICKNESS)

        cv2.imshow('vid', frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            print('User pressed ESC. Exiting...')
            break
    else:
        cv2.destroyAllWindows()
        break
        
    


     