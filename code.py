import cv2 as cv
import numpy as np

img=cv.imread('friends.jpg')
cv.imshow('img',img)

grey=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('grey',grey)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

face_rect= haar_cascade.detectMultiScale(grey,scaleFactor=1.1,minNeighbors=2)

print(f'Number of faces found = {len(face_rect)}')

for (x,y,w,h) in face_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('detect face',img)

cv.waitKey(0)
