import cv2
import dlib

import facerec
import objdet


facerec.read_learnt_encodings()
objdet.read_object_data()

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
        
        # ================== OR part =====================
        frame = objdet.detect_objects(frame)
        # ================== FR part =====================
        frame = facerec.recognise_faces(frame)

        cv2.imshow('vid', frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            print('User pressed ESC. Exiting...')
            break
    else:
        cv2.destroyAllWindows()
        break
