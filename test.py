#EMRE FUCKIN YILDIZ
#Gerekli kütüphaneleri ekliyoruz
import cv2
import os
import numpy as np
from keras.models import load_model
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#Modeli yüklüyoruz
model=load_model("./checkpoint/model.model")

#Ciktilari belirliyoruz
results={0:'Maskesiz',1:'Maskeli'}

#Ciktilarin renklerini belirliyoruz
GR_dict={0:(0,0,255),1:(0,255,0)}

rect_size = 4
cap = cv2.VideoCapture(0) 

#OpenCv Yüz Tanıma kütüphanesini ekliyoruz
haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

while True:
    rval, im = cap.read()
    im=cv2.flip(im,1,1) 

    
    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
        
        face_img = im[y:y+h, x:x+w]
        rerect_sized=cv2.resize(face_img,(150,150))
        normalized=rerect_sized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)

        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
        cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    
    if key == 27: 
        break

cap.release()

cv2.destroyAllWindows()