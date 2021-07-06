import tkinter as tk
from tkinter import Message ,Text
import cv2, os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import PIL.Image,PIL.ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window = tk.Tk()
#helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
window.title("Face_Recogniser")


#window.geometry('720x1280')
window.configure(background='white')
window.attributes('-fullscreen', True)


cv_img = cv2.cvtColor(cv2.imread("bg.jpg"), cv2.COLOR_BGR2RGB)
height,width,no_channels=cv_img.shape
canvas = tk.Canvas(window, width = width, height= height)
canvas.pack()
ph = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
canvas.create_image(0, 0, image=ph, anchor=tk.NW)


message = tk.Label(window, text="Facial Scan" ,bg="White"  ,fg="dark slate gray"  ,width=17  ,height=2,font=('times', 30, 'italic bold underline')) 

message.place(x=550, y=40)

#lbl = tk.Label(window, text="Pass Number",width=20  ,height=2  ,fg="dark olive green"  ,bg="white" ,font=('times', 15, ' bold ') ) 
#lbl.place(x=20, y=200)

#txt = tk.Entry(window,width=20  ,bg="white" ,fg="dark olive green",font=('times', 15, ' bold '))
#txt.place(x=280, y=215)

#lbl2 = tk.Label(window, text="Name",width=20  ,fg="dark olive green"  ,bg="white"    ,height=2 ,font=('times', 15, ' bold ')) 
#lbl2.place(x=11, y=300)

#txt2 = tk.Entry(window,width=20  ,bg="white"  ,fg="dark olive green",font=('times', 15, ' bold ')  )
#txt2.place(x=280, y=315)

#lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="dark olive green"  ,bg="white"  ,height=2 ,font=('times', 15, ' bold underline ')) 
#lbl3.place(x=10, y=400)

#message = tk.Label(window, text="" ,bg="white"  ,fg="dark olive green"  ,width=30  ,height=2, activebackground = "white" ,font=('times', 15, ' bold ')) 
#message.place(x=700, y=400)

lbl3 = tk.Label(window, text="Details : ",width=20  ,fg="dark olive green"  ,bg="white"  ,height=2 ,font=('times', 15, ' bold  underline')) 
lbl3.place(x=400, y=650)


message2 = tk.Label(window, text="" ,fg="dark olive green"   ,bg="white",activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold ')) 
message2.place(x=700, y=650)
 
    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    Roll_No=(txt.get())
    name=(txt2.get())
    if(is_number(Roll_No) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name +"."+Roll_No +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>50:
                break
        cam.release()
        
        cv2.destroyAllWindows()

         
        
        res = "Images Saved"
        row = [Roll_No , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Roll_No)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Roll_No"
            message.configure(text= res)
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Roll_No = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Roll_No))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Done"#+",".join(str(f) for f in Roll_No)
    message.configure(text= res)

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty Roll_No list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Roll_No from the image
        Roll_No=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Roll_No)        
    return faces,Ids


def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Roll_No','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Roll_No, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Roll_No'] == Roll_No]['Name'].values
                tt=str(Roll_No)+"-"+aa
                attendance.loc[len(attendance)] = [Roll_No,aa,date,timeStamp]
                
            else:
                Roll_No='Unknown'                
                tt=str(Roll_No)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Roll_No'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()

    
    cv2.destroyAllWindows()

    
    
    print(attendance)
    res=attendance
    message2.configure(text= res)

    

  
    
#takeImg = tk.Button(window, text="Capture", command=TakeImages  ,fg="dark olive green"  ,bg="white"  ,width=15  ,height=1, activebackground = "dark olive green" ,font=('times', 15, ' bold '))
#takeImg.place(x=200, y=705)
#trainImg = tk.Button(window, text="Next", command=TrainImages  ,fg="dark olive green"  ,bg="white"  ,width=15  ,height=1, activebackground = "dark olive green" ,font=('times', 15, ' bold '))
#trainImg.place(x=500, y=705)
trackImg = tk.Button(window, text="Mark Entry", command=TrackImages  ,fg="dark olive green"  ,bg="white"  ,width=15  ,height=1, activebackground = "dark olive green" ,font=('times', 15, ' bold '))
trackImg.place(x=800, y=705)
quitWindow = tk.Button(window, text="Back", command=window.destroy  ,fg="dark olive green"  ,bg="white"  ,width=15  ,height=1, activebackground = "dark olive green" ,font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=705)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('times', 30, 'italic bold underline'))
copyWrite.tag_configure("superscript", offset=10)
"""copyWrite.insert("insert", "Developed by Kunal","", "TEAM", "superscript")
copyWrite.configure(state="disabled",fg="white"  )
copyWrite.pack(side="left")
copyWrite.place(x=800, y=750)"""
 
window.mainloop()
