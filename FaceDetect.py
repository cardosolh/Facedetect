# -*- coding: UTF-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (224, 224)

# origem video: https://www.youtube.com/watch?v=c9r4Er8aFmY


face_cascade = cv2.CascadeClassifier(
    'modelo/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


img = cv2.imread('imagem/serie.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

video = cv2.VideoCapture('video/run_torwards_camera.mp4')
success, image = video.read()
count = 0
faceping = 1
#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=6,
    minSize=(30, 30),
)
while success:

    for (x, y, w, h) in faces:
        img = cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.imwrite("Pessoas/serie_face" + faceping + ".jpg", roi_color)
        faceping += 1
    success, image = video.read()
    count += 1

cv2.imwrite("imagem/serie_face.jpg", img)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
