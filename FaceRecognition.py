# Screen Recorder

import sys
from PIL import ImageGrab
import os
import numpy as np
import cv2
import threading


click_on = False  # Click status
confidenceThreshold = 1
point1 = (0, 0)  # First click (first position in crop)
point2 = (1, 1)  # On release  (last position in crop)
Option = -1  # In which way we dragged the image see lines 58-71.
Id = 0  # default is 0, fullNames[Id] -> will give us the recognized face.
fullNames = ['None']  # will expand with the known people names.
path = 'Images'  # path to the face images

# LBPH algorithm for implementing the face recognition.
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainedModel/Model.yml')  # loading our Model.

# Using face detection HAAR cascade classifier.
faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX  # font for the people names shown on the screen


# Adding names according to the images and ID taken in step 1
def Add_Names():
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # Collecting all the names from our Images that trained.
    for names in imagePaths:
        fullName = str(os.path.split(names)[-1].split(".")[2])
        if fullName not in fullNames:
            fullNames.append(fullName)
    # Starting a new thread for the screen recorder
    thread_1 = threading.Thread(target=Screen)
    thread_1.start()


# When we drag an image on the "recording screen"
def click(event, x, y, flag, screen):
    global click_on, point1, point2, Option
    if event == cv2.EVENT_LBUTTONDOWN:
        # if mouse down, store the (x,y) position of the mouse
        click_on = True
        point1 = (x, y)
        # taking the start coordinates

    elif event == cv2.EVENT_MOUSEMOVE and click_on:
        # when dragging pressed, draw rectangle in image
        img_copy = screen.copy()
        cv2.rectangle(img_copy, point1, (x, y), (0, 0, 255), 2)
        # Showing on the main screen the "sub image" we want to choose
        cv2.imshow("Screen Recorder", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        # on mouse up, click = false and taking the closing point (x,y).
        click_on = False
        point2 = (x, y)
        # There are 4 options to capture image, we covered them all.
        if y > point1[1] and x > point1[0]:
            Option = 1
            # if we moving down and right
        elif y < point1[1] and x > point1[0]:
            Option = 2
            # if we moving up and right
        elif y > point1[1] and x < point1[0]:
            Option = 3
            # if we moving down and left
        elif y < point1[1] and x < point1[0]:
            Option = 4
            # if we moving up and left
        else:
            Option = -1
            # refreshing the screen from the previous choices.


def Screen():
    global Id, Option, confidenceThreshold
    while True:
        if not click_on:  # you can resize the window ---> 1000 is for width, 800 is for height
            screen_view = np.array(ImageGrab.grab(bbox=(0, 0, 1200, 1000)))  # screen capturing
            screen_view = screen_view[:, :, ::-1]  # from BGR to RGB
            screen_copy = screen_view.copy()  # manipulating the screen image
            #  When mouse clicks the thread starts calling the click() function.
            thread_2 = threading.Thread(target=cv2.setMouseCallback, args=("Screen Recorder", click, screen_view, ))
            thread_2.start()
            if Option == 1:
                # turning the sub image we chose to a gray scale image.
                gray = cv2.cvtColor(screen_copy[point1[1]:point2[1], point1[0]:point2[0]], cv2.COLOR_BGR2GRAY)
                # sending the gray sub image to the face detector.
                faces = faceDetector.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5,
                )
                # for all the faces we found.
                for (x, y, w, h) in faces:
                    # we sending the faces to our face recognizer
                    Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                    # grabbing the face image size for changing the confidence Threshold
                    faceSize = gray[y:y + h, x:x + w].shape[0] * gray[y:y + h, x:x + w].shape[1]
                    # As bigger the face as lower the threshold needs to be.
                    if faceSize > 15000:
                        confidenceThreshold = 80
                    elif faceSize > 8500:
                        confidenceThreshold = 88
                    else:
                        confidenceThreshold = 100

                    if confidence < confidenceThreshold:
                        # the name of the recognized face.
                        Id = fullNames[Id]
                        # Drawing a rectangle around the face.
                        cv2.rectangle(screen_copy, (point1[0] + x, point1[1] + y),
                                      (point1[0] + x + w, point1[1] + y + h), (0, 255, 0), 2)
                        # writing the name on top of the rectangle.
                        cv2.putText(screen_copy, str(Id), (point1[0] + x - 50, point1[1] + y - 5),
                                    font, 1, (0, 0, 255), 2)
                    else:
                        # Drawing a rectangle around the face.
                        cv2.rectangle(screen_copy, (point1[0] + x, point1[1] + y),
                                      (point1[0] + x + w, point1[1] + y + h), (0, 0, 255), 2)
                        # writing the name on top of the rectangle.
                        Id = "unknown"
                        cv2.putText(screen_copy, str(Id), (point1[0] + x + 5, point1[1] + y - 5),
                                    font, 1, (0, 255, 0), 2)

            elif Option == 2:
                # for each option we do the same like in option 1, except the screen crop read lines --> 90 - 131
                gray = cv2.cvtColor(screen_copy[point2[1]:point1[1], point1[0]:point2[0]], cv2.COLOR_BGR2GRAY)
                faces = faceDetector.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5,
                )
                for (x, y, w, h) in faces:
                    Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                    faceSize = gray[y:y + h, x:x + w].shape[0] * gray[y:y + h, x:x + w].shape[1]

                    if faceSize > 15000:
                        confidenceThreshold = 75
                    elif faceSize > 8500:
                        confidenceThreshold = 90
                    else:
                        confidenceThreshold = 100

                    if confidence < confidenceThreshold:
                        Id = fullNames[Id]
                        cv2.rectangle(screen_copy, (point1[0] + x, point2[1] + y),
                                      (point1[0] + x + w, point2[1] + y + h), (0, 255, 0), 2)
                        cv2.putText(screen_copy, str(Id), (point1[0] + x - 50, point2[1] + y - 5),
                                    font, 1, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(screen_copy, (point1[0] + x, point2[1] + y),
                                      (point1[0] + x + w, point2[1] + y + h), (0, 0, 255), 2)
                        Id = "unknown"
                        cv2.putText(screen_copy, str(Id), (point1[0] + x + 5, point2[1] + y - 5),
                                    font, 1, (0, 255, 0), 2)

            elif Option == 3:
                # for each option we do the same like in option 1, except the screen crop read lines --> 90 - 131
                gray = cv2.cvtColor(screen_copy[point1[1]:point2[1], point2[0]:point1[0]], cv2.COLOR_BGR2GRAY)
                faces = faceDetector.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5,
                )

                for (x, y, w, h) in faces:
                    Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                    faceSize = gray[y:y + h, x:x + w].shape[0] * gray[y:y + h, x:x + w].shape[1]

                    if faceSize > 15000:
                        confidenceThreshold = 75
                    elif faceSize > 8500:
                        confidenceThreshold = 90
                    else:
                        confidenceThreshold = 100

                    if confidence < confidenceThreshold:
                        Id = fullNames[Id]
                        cv2.rectangle(screen_copy, (point2[0] + x, point1[1] + y),
                                      (point2[0] + x + w, point1[1] + y + h), (0, 255, 0), 2)
                        cv2.putText(screen_copy, str(Id), (point2[0] + x - 50, point1[1] + y - 5),
                                    font, 1, (0, 0, 255), 2)

                    else:
                        cv2.rectangle(screen_copy, (point2[0] + x, point1[1] + y),
                                      (point2[0] + x + w, point1[1] + y + h), (0, 0, 255), 2)
                        Id = "unknown"
                        cv2.putText(screen_copy, str(Id), (point2[0] + x + 5, point1[1] + y - 5),
                                    font, 1, (0, 255, 0), 2)

            elif Option == 4:
                # for each option we do the same like in option 1, except the screen crop read lines --> 90 - 131
                gray = cv2.cvtColor(screen_copy[point2[1]:point1[1], point2[0]:point1[0]], cv2.COLOR_BGR2GRAY)
                faces = faceDetector.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5,
                )

                for (x, y, w, h) in faces:
                    Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                    faceSize = gray[y:y + h, x:x + w].shape[0] * gray[y:y + h, x:x + w].shape[1]

                    if faceSize > 15000:
                        confidenceThreshold = 75
                    elif faceSize > 8500:
                        confidenceThreshold = 90
                    else:
                        confidenceThreshold = 100

                    if confidence < confidenceThreshold:
                        Id = fullNames[Id]
                        cv2.rectangle(screen_copy, (point2[0] + x, point2[1] + y),
                                      (point2[0] + x + w, point2[1] + y + h), (0, 255, 0), 2)
                        cv2.putText(screen_copy, str(Id), (point2[0] + x - 50, point2[1] + y - 5),
                                    font,  1, (0, 0, 255), 2)

                    else:
                        cv2.rectangle(screen_copy, (point2[0] + x, point2[1] + y),
                                      (point2[0] + x + w, point2[1] + y + h), (0, 0, 255), 2)
                        Id = "unknown"
                        cv2.putText(screen_copy, str(Id), (point2[0] + x + 5, point2[1] + y - 5),
                                    font, 1, (0, 255, 0), 2)

            cv2.imshow("Screen Recorder", screen_copy)  # showing the Screen recorder
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' for exit video
            break
    sys.exit()
