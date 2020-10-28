import cv2
import pickle
import face_recognition

#=================================== INITIALIZATION ===========================================
KNOWN_FACES_DIR = 'knownFaces'
TOLERANCE = 0.6
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.5
FONT_THICKNESS = 2
FRAME_THICKNESS = 2
MODEL = 'cnn'

known_faces = [] # stores the encoded data for the known faces
known_names = [] # stores the name string for the known faces

#=================================== READ KNOWN FACES =========================================
def read_learnt_encodings():
    global known_faces
    global known_names
    
    print('Loading encodings and names of known faces...')

    filename = 'faceEncodings'
    infile = open(f'{KNOWN_FACES_DIR}/{filename}','rb')
    known_faces = pickle.load(infile)
    infile.close()

    filename = 'faceNames'
    infile = open(f'{KNOWN_FACES_DIR}/{filename}','rb')
    known_names = pickle.load(infile)
    infile.close()
       
    print('Encodings and names loaded successfully...')


#================================= PROCESS UNKNOWN FACES ======================================

def recognise_faces(frame):
    
    locations = face_recognition.face_locations(frame,1,'cnn') # find the location of faces detected in a frame
    encodings = face_recognition.face_encodings(frame, locations) # encode the faces detected
        
    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
        else:
            match = 'Unknown'
            
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
    
    return frame


#=============================================================================================