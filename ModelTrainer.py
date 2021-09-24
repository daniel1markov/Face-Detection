# Step 2: Import the face images we took in Step 1
# --> Train the face Recognition Model with this program.

import cv2
import numpy as np
from PIL import Image
import os

# Path for face images database
imagesPath = 'Images'

# LBPH algorithm for training the face recognition.
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
# Using face detection HAAR cascade classifier.
faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def getImagesAndLabels(path):
    # Take all the files saved from part 1.
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    Ids = []
    for imagePath in imagePaths:
        # Importing the face images.
        PIL_img = Image.open(imagePath).convert('L')
        # PIL_img.show()
        img_numpy = np.array(PIL_img, dtype=np.uint8)  # turn into 0-255 array
        Id = int(os.path.split(imagePath)[-1].split(".")[1])  # get the person ID
        Faces = faceDetector.detectMultiScale(img_numpy)  # detect the faces
        for (x, y, w, h) in Faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])  # collecting the faces in 0-255 array.
            Ids.append(Id)  # collecting the Ids.
    return faceSamples, Ids


faces, ids = getImagesAndLabels(imagesPath)

# Training the faces with their Ids.
faceRecognizer.train(faces, np.array(ids))

# Save the model into trainedModel/trainedModel.yml
faceRecognizer.write('trainedModel/Model.yml')
