import cv2
import face_recognition
import os
import pickle

class Face_recognition():

    def __init__(self):
        self.KNOWN_FACES_DIR = "knownFaces"
        self.KNOWN_FACES_IMAGE_DIR = 'knownFaces(images)'
        self.known_faces = []
        self.known_names = []
        self.TOP_LEFT = []
        self.BOTTOM_RIGHT = []
        self.LABEL = []

    def learnFace(self):

        print('\n\n** LEARNING FACES **')
        print('Loading known face images...')
        print('Faces learnt:')
        face_count = 1

        self.known_faces.clear()
        self.known_names.clear()

        for name in os.listdir(self.KNOWN_FACES_IMAGE_DIR):
            for filename in os.listdir(f'{self.KNOWN_FACES_IMAGE_DIR}/{name}'):
                image = face_recognition.load_image_file(f'{self.KNOWN_FACES_IMAGE_DIR}/{name}/{filename}')
                encoding = face_recognition.face_encodings(image)[0]
                self.known_faces.append(encoding)
                self.known_names.append(name)
            print(f'{face_count}. {name}')
            face_count = face_count + 1

        print('\nLearning Faces completed successfully.')
        print('Saving learnt faces... \n')

        filename = 'faceEncodings'
        outfile =  open(f'{self.KNOWN_FACES_DIR}/{filename}', 'wb')
        pickle.dump(self.known_faces, outfile)
        outfile.close()
        print(f'Encoded faces stored at: {self.KNOWN_FACES_DIR}/{filename}')
        
        filename = 'faceNames'
        outfile =  open(f'{self.KNOWN_FACES_DIR}/{filename}', 'wb')
        pickle.dump(self.known_names, outfile)
        outfile.close()
        print(f'Face names stored at: {self.KNOWN_FACES_DIR}/{filename}')

        print('Number of faces learnt: ' + str(len(self.known_names)))
        self.read_learnt_encodings()

    def read_learnt_encodings(self):
        
        print('\n\n** READING LEARNT ENCODINGS **')

        filename = 'faceEncodings'
        infile = open(f'{self.KNOWN_FACES_DIR}/{filename}','rb')
        self.known_faces = pickle.load(infile)
        infile.close()

        filename = 'faceNames'
        infile = open(f'{self.KNOWN_FACES_DIR}/{filename}','rb')
        self.known_names = pickle.load(infile)
        infile.close()
        
        print('Encodings and names loaded successfully...')

    def recognise_faces(self, frame, tolerance = 0.55):

        locations = face_recognition.face_locations(frame,1,'cnn')
        encodings = face_recognition.face_encodings(frame, locations)
        
        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance)
            if True in results:
                self.LABEL.append(self.known_names[results.index(True)])
            else:
                self.LABEL.append('Unknown')
                
            #face_location rectangle
            self.TOP_LEFT.append((face_location[3], face_location[0]))
            self.BOTTOM_RIGHT.append((face_location[1], face_location[2]))
