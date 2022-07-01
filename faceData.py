import cv2
import os
import pickle
from trainer import train
def faceData(id,name):
    camera=cv2.VideoCapture(0)
    camera.set(3,640)
    camera.set(4,480)
    cascade=cv2.CascadeClassifier('haarCascade.xml')
    face_id=int(id)
    value=name
    file=open('name.pkl','rb+')
    names=pickle.load(file)
    names[id]=value
    pickle.dump(names,file)
    count=0
    while(True):
        ret,image=camera.read()
        grayscale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces=cascade.detectMultiScale(grayscale,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            count+=1
            cv2.imwrite("dataset/User."+str(face_id)+"."+str(count)+".jpg",grayscale[y:y+h,x:x+w])

            cv2.imshow("image",image)

        k=cv2.waitKey(100) & 0xff
        if k==27:
            break
        elif count==30:
            break
    train()
    camera.release()
    cv2.destroyAllWindows()



