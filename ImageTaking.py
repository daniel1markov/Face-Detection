# Step 1 : take images you want to recognize with this program
# --> Each person will have his personal ID starting from 1, 2, 3.... etc.
import cv2

# using web camera to capture face images
camera = cv2.VideoCapture(0)
# Using face detection HAAR cascade classifier.
faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# For each person, enter a unique id and a Full Name
faceId = input('\n Enter user ID and press Enter : ')
name = input('\n Enter user full name and press Enter : ')
count = 0  # Counting Faces

while True:
    # importing the image from the webcam
    ret, img = camera.read()
    # converting to gray image.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detecting faces with HAAR cascade classifier
    faces = faceDetector.detectMultiScale(gray_img, 1.3, 5)
    for (x, y, w, h) in faces:
        # Drawing a rectangle around the face.
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        count += 1
        # Save the captured image into the Images folder
        cv2.imwrite("Images/User." + str(faceId) + '.' +  # We saving each picture with the unique ID and full name
                    str(name) + "." + str(count) + ".jpg", gray_img[y:y + h, x:x + w])  # for easier import in step 2.
        cv2.imshow('image', img)  # Showing the image
    if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' for exit video
        break
    elif count >= 200:  # Take 200 face sample and stop video
        break

camera.release()
cv2.destroyAllWindows()
