"""
This code calculates the depth of any object as detected by two cameras calliarated and put in parallel to ech other. The object must be in the common
field of view of both the cameras. To calculate the depth the point used in this code is the centroid of the bounding box, object are reidentified using
MSE of the bounding boxes.
"""

import numpy as np
import cv2 as cv
import time
from ultralytics import YOLO as yolo

def mean(x,y):
    return int((x+y)/2)

def mid(x,y):
    return int(abs(x-y)/2)

def CameraDetect():
    for i in range(11):
        cap1 = cv.VideoCapture(i)
        cap2 = cv.VideoCapture(i+4)
        if cap1.read()[0] and cap2.read()[0]:
            return i

## Function to detect the difference between two frames.
def diff(prev, frame):
    # | Compute the Mean Squared Error (MSE) |
    # print("prev",prev.shape)
    # print("frame",frame.shape)
    prev = np.array(prev)
    frame = np.array(frame)
    mse = np.mean((prev - frame) ** 2)
    # print("MSE",mse) ## for debugging purpose only
    return mse

## Function takes in arguments as the box from left image and rights image and thresh and returns the x coordinate of right image.
def Right_scan2(Rimg, Limg, xywh, thresh:float = 1, start:int = 0):
    x,y,w,h = xywh
    h = int(h/2)
    w = int(w/2)
    dict1 = {}
    dict1[-1] = thresh
    j = w + start
    while j<=(Rimg.shape[1]-w-1):
        d = diff(Limg,Rimg[:,j-w:j+w,:])
        dict1[j] = d
        j+=1
    dif = min(dict1.values())
    print("MSE",dif)                          ## Debugging
    if dif >= thresh:
        return None
    return 1

def objectMatch(left, right, datal, datar, dif: float = 5):
    datal = datal
    datar = datar
    if datal[2]-datal[0] <= datar[2]-datar[0]:
        # print("Ifffff")
        img1 = left.copy()
        img2 = right.copy()
        x1,y1,x2,y2 = datal[:4]
        a1,b1,a2,b2 = datar[:4]
    else:
        # print("Else")
        img1 = right.copy()
        img2 = left.copy()
        x1,y1,x2,y2 = datar[:4]
        a1,b1,a2,b2 = datal[:4]

    # print(x1,y1,x2,y2)      ###debugging only       
    # print(a1,b1,a2,b2)      ###debugging only

    w = mid(x2,x1)    ### width/2
    # h = int((y2-y1)/2)    ### hight/2
    x1 = mean(x1,x2)   ### X_centre of small box
    y1 = mean(y1,y2)   ### Y_centre of small box
    a1 = mean(a1,a2)   ### X_centre of large box
    b1 = mean(b1,b2)   ### Y_centre of large box

    # print(x1,y1,w,h)
    # print(a1,b1)

    # img1 = img1[y1-h:y1+h,x1-w:x1+w,:]
    img1 = img1[:,x1-w:x1+w,:] ###debugging only
    # print(img1)
    # img2 = img2[b1-h:b1+h,a1-w:a1+w,:]
    img2 = img2[:,a1-w:a1+w,:]  ###debugging only
    # plt.imshow(img1)
    # plt.show()
    # plt.imshow(img2)
    # plt.show()
    if diff(img1, img2) <= dif:
        return True
    else:
        return False

def objectMatchIteration(left, right, datal, datar, dif: float = 5):
    datal = datal
    datar = datar
    if datal[2]-datal[0] <= datar[2]-datar[0]:
        # print("Ifffff")
        img1 = left.copy()
        img2 = right.copy()
        x1,y1,x2,y2 = datal[:4]
        a1,b1,a2,b2 = datar[:4]
    else:
        # print("Else")
        img1 = right.copy()
        img2 = left.copy()
        x1,y1,x2,y2 = datar[:4]
        a1,b1,a2,b2 = datal[:4]

    # print(x1,y1,x2,y2)      ###debugging only
    # print(a1,b1,a2,b2)      ###debugging only
    img2 = img2[:,a1:a2,:]

    ws = mid(x2,x1)    ### width/2 Small object
    x1 = mean(x1,x2)   ### X_centre of small box
    y1 = mean(y1,y2)   ### Y_centre of small box
    img1 = img1[:,x1-ws:x1+ws,:]
    h = mid(y2,y1)    ### hight/2 Small box
    xywh = np.array([x1,y1,2*ws,2*h])
    R = Right_scan2(img2,img1,xywh,thresh=dif)
    # print("R",R)
    if R == None:
        return False
    else:
        return True

colors = [
    (0, 0, 255),      # Red
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 255, 255),    # Yellow
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (128, 128, 128),  # Gray
    (50, 50, 50),     # Dark Gray
    (200, 200, 200),  # Light Gray
    (0, 0, 128),      # Maroon
    (0, 128, 128),    # Olive
    (128, 0, 128),    # Purple
    (128, 128, 0),    # Teal
    (128, 0, 0),      # Navy
    (0, 165, 255),    # Orange
    (19, 69, 139),    # Brown
    (203, 192, 255),  # Pink
    (230, 216, 173)   # Light Blue
]

"""
Main function or main code starts here.
"""
One = time.time()

i = CameraDetect()
cap_l = cv.VideoCapture(i+4)
cap_r = cv.VideoCapture(i)
if (cap_l.isOpened() == False) or (cap_r.isOpened() == False):
    print("Error opening video file")


# cap_r = cv.VideoCapture('output1.avi')
# cap_l = cv.VideoCapture('output2.avi')


ObjectModel = yolo("yolov8n_openvino_model",task='detect')

k = 0
while cap_l.isOpened():
    strt = time.time()
    ret, past_l = cap_l.read()
    ret, past_r = cap_r.read()
    if ret:
########### This If statement ensures that only one frame out of 5 is fed into the model for inference and rest are shown directly with the result of the one frame that was fed into the model.
        if k%10!=0:
            k+=1
            img1 = past_r.copy()
            img2 = past_l.copy()
            constn = 5000
            for i in comnBox:
                rnd = np.random.randint(len(colors))
                x_r = mean(i[0][0],i[0][2])
                x_l = mean(i[1][0],i[1][2])
                disp = abs(x_l - x_r + 0.01)
                z = round(constn/disp,2)
                # print(i,x_r,x_l)
                # cv.rectangle(img1, tuple((i[0][:2] - [0,20])),tuple((i[0][:2] + [20,0])),colors[rnd],-1)   ### to put depth
                text = "depth:" + str(z) + "cm"
                cv.putText(img1, text, tuple(i[0][:2]), cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[rnd],thickness=1, lineType=cv.LINE_AA)
                cv.rectangle(img1, tuple(i[0][:2]),tuple(i[0][2:4]),colors[rnd],1)
                cv.putText(img2, text, tuple(i[1][:2]), cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[rnd],thickness=1, lineType=cv.LINE_AA)
                cv.rectangle(img2, tuple(i[1][:2]),tuple(i[1][2:4]),colors[rnd],1)
            concat = np.concatenate((img2,img1), axis=1)
            cv.imshow("Mix",concat)
            if cv.waitKey(33) & 0xFF == ord("q"):
                break
            end = time.time()
            print("*******")
            print(end-strt)
            print("*******")
            continue
        else:
            print("k=5")
            k = 0
        k+=1

############# The frame skiping ends here

        Oresult_l = ObjectModel(source = past_l,conf=0.7,iou=0.6)
        Oresult_r = ObjectModel(source = past_r,conf=0.7,iou=0.6)

        l_classes = Oresult_l[0].boxes.cls
        r_classes = Oresult_r[0].boxes.cls

        l_classes = np.array(l_classes)
        r_classes = np.array(r_classes)

        common = np.intersect1d(l_classes, r_classes)
        comnBox = []
        # print(intersection)
        # sorting wrt the width of the boxes
        data_r = sorted(Oresult_r[0].boxes.data.int().tolist(), key=lambda x : x[2])
        data_l = sorted(Oresult_l[0].boxes.data.int().tolist(), key=lambda x : x[2])

        for c in common:
            # print("\n")
            # print(c,1)
            for i in data_r:
                # print("i",i)
                if i[-1] != c:
                    continue
                for j in data_l:
                    # print("j",j)
                    if j[-1] != c:
                        continue
                    # a = objectMatchIteration(past_l,past_r,j,i,dif = 60)
                    a = objectMatch(past_l,past_r,j,i,dif = 90)
                    if a:
                        data_l.remove(j)
                        # print(111111)
                        comnBox.append([i,j])
                        break
        img1 = past_r.copy()
        img2 = past_l.copy()
        constn = 5000
        for i in comnBox:
            rnd = np.random.randint(len(colors))
            x_r = mean(i[0][0],i[0][2])
            x_l = mean(i[1][0],i[1][2])
            disp = abs(x_l - x_r + 0.01)
            z = round(constn/disp,2)
            # cv.rectangle(img1, tuple((i[0][:2] - [0,20])),tuple((i[0][:2] + [20,0])),colors[rnd],-1)   ### to put depth
            text = "depth:" + str(z) + "cm"
            cv.putText(img1, text, tuple(i[0][:2]), cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[rnd],thickness=1, lineType=cv.LINE_AA)
            cv.rectangle(img1, tuple(i[0][:2]),tuple(i[0][2:4]),colors[rnd],1)
            cv.putText(img2, text, tuple(i[1][:2]), cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[rnd],thickness=1, lineType=cv.LINE_AA)
            cv.rectangle(img2, tuple(i[1][:2]),tuple(i[1][2:4]),colors[rnd],1)
        concat = np.concatenate((img2,img1), axis=1)
        cv.imshow("Mix",concat)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
        end = time.time()
        print("*******")
        print(end-strt)
        print("*******")
    else:
        break
cap_l.release()
cap_r.release()
cv.destroyAllWindows()
TheEnd = time.time()
print("Overall time = ",TheEnd-One)