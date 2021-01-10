# import cv2
# import numpy as np

# cap = cv2.VideoCapture(0)

# while(1):

#     # ret is the boolean which retruns true if video is being captured, frame is capturing actual frame
#     ret, frame = cap.read()

#     # Convert BGR to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # define range of blue color in HSV
#     lower_blue = np.array([110,50,50])
#     upper_blue = np.array([130,255,255])

#     #COLOR RANGE YELLOW
#     #lower_YELLOW = np.array([20,255,255])
#     upper_YELLOW = np.array([130,255,255])

#     # Threshold the HSV image to get only blue colors
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)

#     # Bitwise-AND mask and original image
#     res = cv2.bitwise_and(frame,frame, mask= mask)
#     cv2.rectangle(frame,(384,0),(510,128),(0,255,0),3)

#     cv2.imshow('frame',frame)
#     cv2.imshow('mask',mask)
#     cv2.imshow('res',res)
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break

# cv2.destroyAllWindows()

import cv2
import numpy as np


cap = cv2.VideoCapture(0)

def empty(a):
    pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",640,240)
cv2.createTrackbar("HUE Min","HSV",0,179,empty)
cv2.createTrackbar("HUE Max","HSV",179,179,empty)
cv2.createTrackbar("SAT Min","HSV",0,255,empty)
cv2.createTrackbar("SAT Max","HSV",255,255,empty)
cv2.createTrackbar("VALUE Min","HSV",0,255,empty)
cv2.createTrackbar("VALUE Max","HSV",255,255,empty)

while True:

    _, img = cap.read()
    imgHsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


    h_min = cv2.getTrackbarPos("HUE Min","HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHsv,lower,upper)
    result = cv2.bitwise_and(img,img, mask = mask)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([img,mask,result])
    #cv2.imshow('Original', img)
    #cv2.imshow('HSV Color Space', imgHsv)
    #cv2.imshow('Mask', mask)
    #cv2.imshow('Result', result)
    cv2.imshow('Horizontal Stacking', hStack)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()