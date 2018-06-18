#coding=utf-8
#性别识别

import cv2
from keras.models import load_model
import numpy as np
import chineseText

img = cv2.imread("E:\\Code\\Python\\img\\gather_2.jpg")
face_classifier = cv2.CascadeClassifier(
    "F:\\Python3.6.5\\Lib\\site-packages\\opencv-master\\data\\haarcascades\\haarcascade_frontalface_default.xml"
)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(
    gray, scaleFactor=1.2, minNeighbors=3, minSize=(140, 140))

gender_classifier = load_model(
    "E:\\Github\\faceai\\faceai\\classifier\\gender_models\\simple_CNN.81-0.96.hdf5")
gender_labels = {0: '女', 1: '男'}
color = (255, 255, 255)

for (x, y, w, h) in faces:
    face = img[(y - 60):(y + h + 60), (x - 30):(x + w + 30)]
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, 0)
    face = face / 255.0
    gender_label_arg = np.argmax(gender_classifier.predict(face))
    gender = gender_labels[gender_label_arg]
    cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
    img = chineseText.cv2ImgAddText(img, gender, x + h, y, color, 30)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
