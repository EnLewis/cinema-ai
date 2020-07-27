import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture('vid_test.mp4')
img = cv.imread('test_colour.jpg')
# while True:

#Need to implement adaptive thresholding
_, frame = cap.read()
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
ad_thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 401, 2)

cv.imshow('gray', gray)
cv.imshow('thresh', thresh)
cv.imshow('ad_thresh', ad_thresh)

#Edge Detection
# kernel = np.ones((3,3), np.uint8)
# opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

# sure_bg = cv.dilate(opening, kernel, iterations=3)

# dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
# ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(),255,0)

# sure_fg = np.uint8(sure_fg)
# unkown = cv.subtract(sure_bg, sure_fg)

# cv.imshow('sure_fg', sure_fg)
# cv.imshow('sure_bg', sure_bg)

while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
