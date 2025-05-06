#Algorithm defined by datacamp.com

import cv2
import matplotlib.pyplot as plt

imagePath = 'input_image.jpg'


#Read Image

img = cv2.imread(imagePath)

print(img.shape)

#Convert to grey scale

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(gray_img.shape)

#Load Classifier

face_classifier = cv2.CascadeClassifier(
    ".venv/lib/python3.13/site-packages/cv2/data/haarcascade_frontalface_default.xml"
)

#face detection
face = face_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))

#draw box
for(x, y, w, h) in face:
    cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), 4)
    
#display image
##convert to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
##use matplotlib to display

plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')