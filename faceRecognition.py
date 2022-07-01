import cv2
import numpy as np
import os
import pickle
def recognition():
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath="haarCascade.xml"
    faceCascade=cv2.CascadeClassifier(cascadePath)
    font=cv2.FONT_HERSHEY_SIMPLEX

    id=0
    f=open("name.pkl","rb")
    names=pickle.load(f)

    camera=cv2.VideoCapture(0)
    camera.set(3,640)
    camera.set(4,480)

    minW=0.1*camera.get(3)
    minH=0.1*camera.get(4)

    while(True):
        ret,image=camera.read()
        grayscale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(grayscale,scaleFactor=1.2,minNeighbors=5,minSize=(int(minW),int(minH)))
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            id,confidence=recognizer.predict(grayscale[y:y+h,x:x+w])
            print(id)
            
            

            if(confidence<100):
                
                ids=names[id]
               
                confidence="{0}%".format(round(100-confidence))
            else:
                id="unkown"
                confidence="{0}%".format(round(100-confidence))
            cv2.putText(image,str(ids),(x+5,y-5),font,1,(255,255,255),2)
            cv2.putText(image,str(confidence),(x+5,y+h-5),font,1,(255,255,0),1)
        cv2.imshow('camera',image)
        k=cv2.waitKey(10) & 0xff
        if k==27:
            break
    camera.realease()
    cv2.destroyAllWindows()
recognition()