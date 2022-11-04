from fileinput import filename
from django.forms import ImageField
from django.shortcuts import redirect, render
from msilib import sequence
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
import base64
from home import models
import os
from django.conf import settings
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import mimetypes
from django.http.response import HttpResponse

def main(request):
    return render(request, 'home/home.html')

def graph_1():

    df = pd.read_excel("Report1.xlsx")
    df['Emotion'].value_counts().sort_values().plot.line()
    plt.savefig('zen/static/assets/images/graph4.png')

def graph_2():
    df = pd.read_excel("Report1.xlsx")
    df.Emotion.value_counts()
    emotion_map = {0: 'fear', 1: 'sad', 2: 'neutral', 3: 'happy', 4: 'surprise', 5: 'disgust', 6: 'angry'}
    emotion_counts = df['Emotion'].value_counts(sort=False).reset_index()
    emotion_counts.columns = ['Emotion','number']
    emotion_counts['Emotion'] = emotion_counts['Emotion']
    x = emotion_counts.Emotion
    y = emotion_counts.number
    plt.figure(figsize=(6,4))
    sns.scatterplot(x, y,data =df)
    plt.savefig('zen/static/assets/images/graph2.png')

def graph_3():
    df = pd.read_excel("Report1.xlsx")
    emotion_counts = df['Emotion'].value_counts(sort=False).reset_index()
    emotion_counts.columns = ['Emotion','number']
    emotion_counts['Emotion'] = emotion_counts['Emotion']
    x = emotion_counts.Emotion
    y = emotion_counts.number

    plt.axis('equal')
    l = []
    for i in range(0,len(y)):
        l.append(0.1)
    plt.pie(y,labels=x,radius=1.5,autopct='%0.1f%%',shadow=True,explode=l)
    plt.savefig('zen/static/assets/images/graph3.png')

def graph_4():

    df = pd.read_excel("Report1.xlsx")
    df.Emotion.value_counts()
    emotion_map = {0: 'fear', 1: 'sad', 2: 'neutral', 3: 'happy', 4: 'surprise', 5: 'disgust', 6: 'angry'}
    emotion_counts = df['Emotion'].value_counts(sort=False).reset_index()
    emotion_counts.columns = ['Emotion','number']
    emotion_counts['Emotion'] = emotion_counts['Emotion']
    emotion_counts

    plt.figure(figsize=(6,4))
    sns.barplot(emotion_counts.Emotion, emotion_counts.number)
    plt.title('Emotion distribution')
    plt.ylabel('Number', fontsize=12)
    plt.xlabel('Emotions', fontsize=12)
    plt.savefig('zen/static/assets/images/graph1.png')


def session(request):
    
        
    face_classifier = cv2.CascadeClassifier(r'C:\Users\HP\OneDrive\Desktop\ZEN\zen\home\haarcascade_frontalface_default.xml')
    classifier = load_model(r'C:\Users\HP\OneDrive\Desktop\ZEN\zen\home\model.h5')

    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

    cap = cv2.VideoCapture(0)
    dmy = pd.date_range('2022-04-02', periods=1, freq='H')  
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

    
    graph_1()
    graph_2()
    graph_3()
    graph_4()

    image1 = Image.open("zen/static/assets/images/graph1.png")
    image2 = Image.open("zen/static/assets/images/graph2.png")
    image3 = Image.open("zen/static/assets/images/graph3.png")
    image4 = Image.open("zen/static/assets/images/graph4.png")

    im1 = image1.convert('RGB')
    im2 = image2.convert('RGB')
    im3 = image3.convert('RGB')
    im4 = image4.convert('RGB')
    im_list = [im2, im3, im4]

    pdf1_filename = "Final-Report.pdf"

    im1.save(pdf1_filename, "PDF" ,resolution=100.0, save_all=True, append_images=im_list)
        
    return render(request,'home/session.html')


# def download_file():
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     filename = 'Final-Report.pdf'
#     filepath = BASE_DIR + '\\' + filename
#     path = open(filepath, 'rb')
#     mime_type, _ = mimetypes.guess_type(filepath)
#     response = HttpResponse(path, content_type=mime_type)
#     response['Content-Disposition'] = "attachment; filename=%s" % filename
#     return response

def download_file(request):
    if True:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filename = 'Final-Report.pdf'
        filepath = BASE_DIR + '//' + filename
        path = open(filepath, 'rb')
        mime_type, _ = mimetypes.guess_type(filepath)
        response = HttpResponse(path, content_type=mime_type)
        response['Content-Disposition'] = "attachment; filename=%s" % filename
        return response
    else:
        return render(request, 'home/download.html')