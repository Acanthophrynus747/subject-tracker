# https://www.instructables.com/Face-Tracking-Device-Python-Arduino/
# https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
# https://github.com/opencv/opencv/tree/master/data/haarcascades

import cv2
import serial
import time
import os
import numpy as np

# ard = serial.Serial("COM3", 115200) #look into whether different baud rates are better

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #haar cascade file fixed according to https://stackoverflow.com/questions/30508922/error-215-empty-in-function-detectmultiscale

vid = cv2.VideoCapture(0) # webcam 0

while True:
    _, frame = vid.read() # current frame to var. frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #frame to greyscale

    faces = face_cascade.detectMultiScale(gray, minSize=(80, 80), minNeighbors=3)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # x_pos = x + (w/2)
        # y_pos = y + (h/2)

        # if x_pos > 280:
        #     ard.write('L'.encode())
        #     time.sleep(0.01)
        # elif x_pos < 360:
        #     ard.write('R'.encode())
        #     time.sleep(0.01)
        # else:
        #     ard.write('S'.encode())
        #     time.sleep(0.01)

        # if y_pos > 280:
        #     ard.write('D'.encode())
        #     time.sleep(0.01)
        # elif y_pos < 200:
        #     ard.write('U'.encode())
        #     time.sleep(0.01)
        # else:
        #     ard.write('S'.encode())
        #     time.sleep(0.01)
        # break

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)&0xFF
    if(k == ord('q')): #if q is pressed exit while loop
        break

cv2.destroyAllWindows()
# ard.close()
vid.release()

