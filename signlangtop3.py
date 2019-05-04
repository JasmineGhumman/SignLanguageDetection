# -*- coding: utf-8 -*-
"""
Created on Sat May  4 23:34:22 2019

@author: jasmi
"""


import cv2
import numpy as np

def nothing(x):
    pass

image_x, image_y = 224,224
from keras.models import load_model
classifier = load_model('Trained_model.h5')

def predictor():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(224, 224))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       print(result)
       '''l=0
       k=0
       for i in range(0,26):
            if result[0][i]>l:
              l=result[0][i]
              k=i
       result[0][k]=1'''
       dic={}
       for letter in range(97,122):
           dic[chr(letter)]=result[0][letter-97]
       print(dic)
       listt=list()
       for key, value in sorted(dic.items(), key=lambda item: item[1]):
           ltemp=[key,value]
           listt.append(ltemp)
       #retlist=list()
       #cv2.putText(frame,l)
       print(listt)
       print(len(listt))
       s=listt[25]
       #s.join(str(listt[25]))
       return (s)
       
      

       

cam = cv2.VideoCapture(0)


cv2.namedWindow("test")

img_counter = 0

img_text = ''
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)
    imcrop = img[102:298, 427:623]
    #hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    #mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("test", frame)
    #cv2.imshow("mask", mask)
    
    #if cv2.waitKey(1) == ord('c'):
    img_name = "1.png"
    save_img = cv2.resize(imcrop, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    print("{} written!".format(img_name))
    img_text = predictor()
    print(img_text)    

    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()