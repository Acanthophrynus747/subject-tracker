# https://www.instructables.com/Face-Tracking-Device-Python-Arduino/
# https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
# https://github.com/opencv/opencv/tree/master/data/haarcascades

import cv2
import serial
import time
import os
import numpy as np

usb = serial.Serial("COM4", 115200) #look into whether different baud rates are better. orginally COM3 but hopefully COM4 will work, COM3 being used by platformio I think?

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #haar cascade file fixed according to https://stackoverflow.com/questions/30508922/error-215-empty-in-function-detectmultiscale

vcap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # webcam 0, idk what cap_dshow does but https://stackoverflow.com/questions/19448078/python-opencv-access-webcam-maximum-resolution

#use max possible resolution. maybe not even necessary
vcap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000) #lol worked, above cap_dshow may not be needed?

try:

    if vcap.isOpened():
        vid_width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH) #or width = vcap.get(3)
        vid_height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = vcap.get(cv2.CAP_PROP_FPS)

        print(f"Started video capture window: {vid_width} x {vid_height}px")

    while True:
        _, frame = vcap.read() # current frame to var. frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #frame to greyscale

        faces = face_cascade.detectMultiScale(gray, minSize=(80, 80), minNeighbors=3)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            x_pos = x + (w/2) #position of center of face
            y_pos = y + (h/2)

            x_origin = vid_width / 2
            y_origin = vid_height / 2 #TODO: add these into below control logic so that it can work with multiple webcam resolutions
            # if x_pos > x_origin send left or something

            if x_pos > 280:
                usb.write('L'.encode())
                time.sleep(0.01)
            elif x_pos < 360:
                usb.write('R'.encode())
                time.sleep(0.01)
            else:
                usb.write('S'.encode())
                time.sleep(0.01)

            if y_pos > 280:
                usb.write('D'.encode())
                time.sleep(0.01)
            elif y_pos < 200:
                usb.write('U'.encode())
                time.sleep(0.01)
            else:
                usb.write('S'.encode())
                time.sleep(0.01)
            break

        # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', frame)

        k = cv2.waitKey(1)&0xFF
        if(k == ord('q')): #if q is pressed exit while loop
            break

    cv2.destroyAllWindows()
    usb.close()
    vcap.release()

except KeyboardInterrupt:
    exit()