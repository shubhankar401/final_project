"""
Code for camera calibration
"""
import cv2 as cv
import numpy as np

i = 0
cap_l = cv.VideoCapture(i+4)
cap_r = cv.VideoCapture(i)
while True:
    ret,framel = cap_l.read()
    ret,framer = cap_r.read()
    h = 230
    cv.rectangle(framer,(0,h),(640,h),(0,255,0),-1)
    cv.rectangle(framel,(0,h),(640,h),(0,255,0),-1)
    cv.imshow("Left", framel)
    cv.imshow("Right", framer)
    # l,r =union(framel,framer,dif=10)
    concat = np.concatenate((framel,framer), axis=1)
    cv.imshow("Mix", concat)
    # concat2 = np.concatenate((r,l), axis=1)
    # cv.imshow("Mix2", concat2)
    if cv.waitKey(10) & 0xFF == ord("q"):
       break
