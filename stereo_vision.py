from ultralytics import YOLO as yolo
import cv2 as cv
import numpy as np

## Function to detect the difference between two frames.
def diff(prev, frame):

    # prev = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # | Compute the Mean Squared Error (MSE) |
    mse = np.mean((prev - frame) ** 2)
    print(mse,"Mse") ## for debugging purpose only
    return mse

## Function takes in arguments as the box from left image and rights image and thresh and returns the x coordinate of right image.
def Right_scan(Rimg, Limg, xywh, thresh:float = 1, start:int = 0):
    x,y,w,h = xywh
    h = int(h/2)
    w = int(w/2)
    Limg = Limg[y-h:y+h,x-w:x+w,:]
    dict1 = {}
    j = w + start

    while j<=(Rimg.shape[1]-w): ### This loop searches in the array for one iteration using a large box
        d = diff(Limg,Rimg[y-h:y+h,j-w:j+w,:])
        dict1[j] = d
        j+=(w)
    if Rimg.shape[1]%w!=0: ### A case to take the last part of the right image into consideration
        j = Rimg.shape[1] - w
        d = diff(Limg,Rimg[y-h:y+h,j-w:j+w,:])
        dict1[j] = d
    min_key = min(dict1, key=dict1.get)
    j = min_key - w
    while j<=(min_key+w):
        if j >= (Rimg.shape[1]-w) or j<=w:
            break
        
        d = diff(Limg,Rimg[y-h:y+h,j-w:j+w,:])
        dict1[j] = d
        j+=1
    dif = min(dict1.values())
    # print(dif)
    if dif > thresh:
        return None
    x_r = min(dict1, key=dict1.get)
    return x_r

## Function takes in arguments as the box from left image and rights image and thresh and returns the x coordinate of right image.
def Right_scan2(Rimg, Limg, xywh, thresh:float = 1, start:int = 0):
    x,y,w,h = xywh
    h = int(h/2)
    w = int(w/2)
    Limg = Limg[y-h:y+h,x-w:x+w,:]
    dict1 = {}
    j = w + start

    if w <= Rimg.shape[1]*0.15:
      while j<=(Rimg.shape[1]-w): ### This loop searches in the array for one iteration using a large box
        d = diff(Limg,Rimg[y-h:y+h,j-w:j+w,:])
        dict1[j] = d
        j+=(w)
      if Rimg.shape[1]%w!=0: ### A case to take the last part of the right image into consideration
          j = Rimg.shape[1] - w
          d = diff(Limg,Rimg[y-h:y+h,j-w:j+w,:])
          dict1[j] = d
      min_key = min(dict1, key=dict1.get)
      j = min_key - w
      while j<=(min_key+w):
          if j >= (Rimg.shape[1]-w):
              break
          d = diff(Limg,Rimg[y-h:y+h,j-w:j+w,:])
          dict1[j] = d
          j+=1
    else:
       while j<=(Rimg.shape[1]-w):
        d = diff(Limg,Rimg[y-h:y+h,j-w:j+w,:])
        dict1[j] = d
        j+=1
    dif = min(dict1.values())
    # print(dif)                          ## Debugging
    if dif > thresh:
        return None
    x_r = min(dict1, key=dict1.get)
    return x_r

"""
Main function starts here to determine the Pose of the subject.
"""

PoseModel = yolo('yolov8n-pose.pt', task='pose')
ObjectModel = yolo('yolov3-tinyu.pt')

cap_left = cv.VideoCapture("outpu1.avi")  # open Left Camera
cap_right = cv.VideoCapture("outpu2.avi")   # open Right Camera

# if (cap_left.isOpened() == False) or (cap_right.isOpened() == False):
#     print("Error opening video file")

ret_l, past_l = cap_left.read()

# left_thresh = 3
while cap_left.isOpened():
    ret_l, prev_l = cap_left.read()
    ret_r, prev_r = cap_right.read()
    # if diff(past_l,prev_l) > left_thresh:
    Oresult = ObjectModel(source = prev_l, conf=0.7, iou=0.6)
    #    limg = Oresult[0].plot()
    # else:
    limg = Oresult[0].plot(img=prev_l)
    print(diff(prev_l,past_l),"difference")
    xywh = Oresult[0].boxes.xywh   ## List containing the boxes of the Objects detected
    xywh = sorted(xywh, key=lambda x: x[0])  ## Sorting the list from left to right
    rimg1 = prev_r.copy()
    start = 0
    for box in xywh:
        print("B_For")
        box = box.int()
        x,y,w,h = box
        print("Right scan")
        x_r = Right_scan2(prev_r, prev_l, box,thresh=90,start=start)
        if x_r == None:
            print(0)
            continue
        start = x_r
        xl = int(x_r - w/2)
        xr = int(x_r + w/2)
        yl = int(y - h/2)
        yr = int(y + h/2)
        cv.rectangle(rimg1, (xl,yl), (xr,yr), (0,255,0), 2)
        cv.imshow("Left Annoted",limg)
        cv.imshow("Right Annoted", rimg1)
    past_l = prev_l
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cap_left.release()
cap_right.release()
cv.destroyAllWindows()