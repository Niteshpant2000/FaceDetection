from tkinter import *
from tkinter import ttk
from faceData import faceData
from faceDetection import faceDetection
from faceRecognition import recognition
from trainer import train
def temp(input1,input2):
    id=input1.get()
    name=input2.get()
    faceData(id,name)

def new_Frame():
    root=Tk()
    frm=ttk.Frame(root,padding=100,height=1000,width=900)

    frm.grid()
    t=Label(frm,text="Enter id must be integer")
    t.pack()
    input=Entry(frm)
    input.pack()
    input.focus_set()
    
    x=Label(frm,text="Enter Name of The Person")
    x.pack()
    input2=Entry(frm)
    input2.pack()
    input2.focus_set()
   
    
    b=Button(frm,text="Set",command= lambda : temp(input,input2))
    b.pack(side='bottom')
    root.mainloop()
    root.destroy()




root=Tk()
frm=ttk.Frame(root,padding=100,height=1000,width=720)
frm.grid()
ttk.Button(frm,text="Data Collection",command=new_Frame).grid(column=0,row=0)
ttk.Button(frm,text="Face Detection",command=faceDetection).grid(column=0,row=1)
ttk.Button(frm,text="Face Recognition",command=recognition).grid(column=0,row=2)

ttk.Button(frm,text="Quit",command=root.destroy).grid(column=0,row=3)
frm.pack(fill=None,expand=False)
root.mainloop()


