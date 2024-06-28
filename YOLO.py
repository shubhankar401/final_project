from ultralytics import YOLO as yolo
import numpy as np
import cv2 as cv

model=yolo("yolov3-tinyu.pt")
results=model(source=0,show=True)
