import os
import face_recognition
import pickle

KNOWN_FACES_IMAGE_DIR = 'knownFaces(images)'
KNOWN_FACES_DIR = 'knownFaces'

known_faces = [] # stores the encoded data for the known faces
known_names = [] # stores the name string for the known faces

def learnFace():
    print('** LEARNING FACES ** \n')
    print('Loading known face images...')
    print('Faces learnt:')
    face_count = 1
    for name in os.listdir(KNOWN_FACES_IMAGE_DIR):
        for filename in os.listdir(f'{KNOWN_FACES_IMAGE_DIR}/{name}'):
            image = face_recognition.load_image_file(f'{KNOWN_FACES_IMAGE_DIR}/{name}/{filename}')
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(name)
        print(f'{face_count}. {name}')
        face_count = face_count + 1

    print('\nLearning Faces completed successfully.')
    print('Saving learnt faces... \n')

    filename = 'faceEncodings'
    outfile =  open(f'{KNOWN_FACES_DIR}/{filename}', 'wb')
    pickle.dump(known_faces, outfile)
    outfile.close()
    print(f'Encoded faces stored at: {KNOWN_FACES_DIR}/{filename}')
     
    filename = 'faceNames'
    outfile =  open(f'{KNOWN_FACES_DIR}/{filename}', 'wb')
    pickle.dump(known_names, outfile)
    outfile.close()
    print(f'Face names stored at: {KNOWN_FACES_DIR}/{filename}')

    
if __name__ == '__main__':
    learnFace()
    print('\nDone')