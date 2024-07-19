
from ultralytics import YOLO as yolo
import numpy as np
import cv2 as cv
import time

"""
Defining Various Functions
"""

def CameraDetect():
    for i in range(11):
        cap1 = cv.VideoCapture(i)
        cap2 = cv.VideoCapture(i+4)
        if cap1.read()[0] and cap2.read()[0]:
            return i

## Function to detect the difference between two frames.
def diff(prev, frame):
    prev = np.array(prev)
    frame = np.array(frame)
    mse = np.mean((prev - frame) ** 2)
    # print("MSE",mse)
    return mse

def jointVector(img,key):
    l = []
    for i in key:
        if i.all():
            l.append(img[i[1]-1,i[0]-1])
        else:
            l.append([0,0,0])
    l = np.array(l)
    return l

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
Main Code starts here,
"""
One = time.time()

thresh = 2000
model = yolo('yolov8n-pose_openvino_model',task='pose')

i = CameraDetect()
# i = 2
cap_l = cv.VideoCapture(i+4)
cap_r = cv.VideoCapture(i)
if (cap_l.isOpened() == False) or (cap_r.isOpened() == False):
    print("Error opening video file")

# cap_r = cv.VideoCapture('output1.avi')
# cap_l = cv.VideoCapture('output2.avi')

k = 0
while True:
    strt = time.time()
    ret, frame_l = cap_l.read()
    ret, frame_r = cap_r.read()
    if ret:
        ##### Code for frame skipping 
        if k%5!=0:
            k+=1
            img1 = frame_r.copy()
            img2 = frame_l.copy()
            constn = 5000
            for i,j in enumerate(same):
                rnd = np.random.randint(len(colors))
                color = colors[rnd]
                xr = j[0][:,0]
                xl = j[1][:,0]
                for index in range(len(xr)):
                    if xr[index] == 0 or xl[index] == 0:
                        xr[index] = 9999999999999
                        xl[index] = 0
                disp = abs(xr-xl)
                z = constn/disp
                d = 0
                count = 0
                for a in z:
                    if a > 0.1:
                        d = d + a
                        count+=1
                if count:
                    # print("!@#$%^&*()",count,d)
                    a = round(d/count,2)
                    # print("$$$$$$ - ",a)
                    text = str(a)+" cm"
                    cv.rectangle(img1,tuple(same_box[i][0][:2]),tuple(same_box[i][0][2:]),color,1)
                    cv.rectangle(img2,tuple(same_box[i][1][:2]),tuple(same_box[i][1][2:]),color,1)
                    cv.putText(img1,text,tuple(same_box[i][0][:2]),cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color,thickness=1, lineType=cv.LINE_AA)
                    cv.putText(img2,text,tuple(same_box[i][1][:2]),cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color,thickness=1, lineType=cv.LINE_AA)
            cont = np.concatenate((img2,img1),axis=1)
            cv.imshow("Result",cont)
            if cv.waitKey(10) & 0xFF == ord("q"):
                break
        else:
            print("k = ",k)
            k = 0
            k+=1
            l_result = model(source=frame_l,conf=0.7,iou=0.6)[0]
            r_result = model(source=frame_r,conf=0.7,iou=0.6)[0]

            # key_l = np.array(l_result.keypoints.data)
            # key_r = np.array(r_result.keypoints.data)

            key_l = np.array(l_result.keypoints.xy).astype(int)
            key_r = np.array(r_result.keypoints.xy).astype(int)

            # key_r = key_r[:,:,:2].astype(int)
            # key_l = key_l[:,:,:2].astype(int)

            boxl = np.array(l_result.boxes.xyxy).astype(int)
            boxr = np.array(r_result.boxes.xyxy).astype(int)

            same = []
            same_box = []

            for i,j in enumerate(key_r):
                if j.size==0:
                    continue
                for a,b in enumerate(key_l):
                    if b.size==0:
                        continue
                    rgb1 = jointVector(frame_r,j)
                    rgb2 = jointVector(frame_l,b)
                    if diff(rgb1,rgb2) < thresh:
                        # print(diff(rgb1,rgb2))
                        same_box.append([boxr[i],boxl[a]])
                        same.append([j,b])
                        boxl = np.delete(boxl,a,axis=0)
                        key_l = np.delete(key_l,a,axis=0)
                        break
            img1 = frame_r.copy()
            img2 = frame_l.copy()
            constn = 5000
            for i,j in enumerate(same):
                rnd = np.random.randint(len(colors))
                color = colors[rnd]
                xr = j[0][:,0]
                xl = j[1][:,0]
                for index in range(len(xr)):
                    if xr[index] == 0 or xl[index] == 0:
                        xr[index] = 9999999999999
                        xl[index] = 0
                disp = abs(xr-xl)
                z = constn/disp
                d = 0
                print(z)
                count = 0
            # for i,j in enumerate(same):
            #     rnd = np.random.randint(len(colors))
            #     color = colors[rnd]
            #     xr = j[0][:,0]
            #     xl = j[1][:,0]
            #     for index in range(len(xr)):
            #         if xr[index] == 0 or xl[index] == 0:
            #             xr[index] = 9999999999999
            #             xl[index] = 0
            #     disp = abs(xr-xl)
            #     z = constn/disp
            #     d = 0
            #     count = 0
                for a in z:
                    if a > 0.1:
                        d = d + a
                        count+=1
                if count:
                    # print("!@#$%^&*()",count,d)
                    a = round(d/count,2)
                    # print("$$$$$$ - ",a)
                    text = str(a)+" cm"
                    cv.rectangle(img1,tuple(same_box[i][0][:2]),tuple(same_box[i][0][2:]),color,1)
                    cv.rectangle(img2,tuple(same_box[i][1][:2]),tuple(same_box[i][1][2:]),color,1)
                    cv.putText(img1,text,tuple(same_box[i][0][:2]),cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color,thickness=1, lineType=cv.LINE_AA)
                    cv.putText(img2,text,tuple(same_box[i][1][:2]),cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color,thickness=1, lineType=cv.LINE_AA)
            cont = np.concatenate((img2,img1),axis=1)
            cv.imshow("Result",cont)
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
