import numpy as np
from cv2 import cv2
from numpy import loadtxt
from keras.models import load_model


model = load_model('digitsocr.h5')
width = 640
height = 480

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

def preprocessing(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
  img = img.astype(np.uint8)
  img = cv2.equalizeHist(img) 
  return img

while True:
  success, originalimg = cap.read()
  img = np.asarray(originalimg)
  print(img.shape)
  img = img.astype(np.float32)
  print(img.shape)
  img = cv2.resize(img,(32,32))
  img = preprocessing(img)

  img = img.reshape(1, 32, 32, 1)
  classindex = int(model.predict_classes(img))
  prob = np.amax(model.predict(img))
  #print(classindex, prob)

  if prob>0.5:
    cv2.putText(originalimg, str(classindex)+" "+str(prob), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    cv2.imshow('output', originalimg)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break