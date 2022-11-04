from msilib import sequence
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

face_classifier = cv2.CascadeClassifier(r'E:\New folder\ZEN\zen\home\haarcascade_frontalface_default.xml')
classifier =load_model(r'E:\New folder\ZEN\zen\home\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise', 'Closed_Eyes']

cap = cv2.VideoCapture(0)
dmy = pd.date_range('2022-04-02', periods=10, freq='H')  
df = pd.DataFrame(dmy, columns=['Time'])
list_date = []
list_time = []
list_emotion = []

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
         
        #t = pd.to_timedelta(np.arange(1), unit='s')
        
        #df['Date'] = pd.to_datetime('now').strftime("%Y-%m-%d %H:%M:%S")
        
        date = pd.datetime.now().date().strftime("%Y-%m-%d")
        list_date.append(date)
        time = pd.datetime.now().time().strftime("%H:%M:%S")
        list_time.append(time)
        emotion = label
        list_emotion.append(emotion)
        df_show = pd.DataFrame(
            {'Date': list_date,
            'Time': list_time,
            'Emotion': list_emotion
            })
        print(df_show)

    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):

        df_show.to_csv('Report.csv', sep='\t')

        df_show.to_excel("Report1.xlsx")

        break
cap.release()
cv2.destroyAllWindows() 

