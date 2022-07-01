import numpy as np
import cv2
from scipy.misc import face

def faceDetection():
    cascade=cv2.CascadeClassifier('haarCascade.xml')
    camera=cv2.VideoCapture(0)
    camera.set(3,640)
    camera.set(4,480)

    while(True):
        ret,image=camera.read()
        grayscale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        faces=cascade.detectMultiScale(grayscale,scaleFactor=1.2,minNeighbors=5,minSize=(20,20))

        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            gray=grayscale[y:y+h,x:x+w]
            color=image[y:y+h,x:x+w]
        cv2.imshow('video',image)
        k=cv2.waitKey(30) & 0xff
        if k==27:
            break
    camera.release()
    cv2.destroyAllWindows()

    

