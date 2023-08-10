import cv2
import cv2 as cv2
import numpy as np
from time import time
import threading

MAX_TIME = 10

LOWER_BLUE = np.array([78, 158, 87])
UPPER_BLUE = np.array([156, 255, 255])

lower_bound = LOWER_BLUE
upper_bound = UPPER_BLUE

back_sub = cv2.createBackgroundSubtractorMOG2(history=0, detectShadows=False)
# backSub = cv.createBackgroundSubtractorKNN(history=60)

contours = ()
max_contours = 0

cap = cv2.VideoCapture(0)

_, first_frame = cap.read()

gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
fgmask = back_sub.apply(first_frame)

previous = time()
delta = 0

while True:
    current = time()
    delta += current - previous
    previous = current
    print(delta)

    ret, frame = cap.read()

    # Create color mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if delta > MAX_TIME:
        # Create background substitution mask
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        fgmask = back_sub.apply(frame)

        delta = 0

    # Create intersection frame between color mask and BS mask
    intersection_frame = cv2.bitwise_and(mask, fgmask, mask=None)

    # Draw contours around trains on intersection frame
    result = frame.copy()
    intersection_contours = cv2.findContours(intersection_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    intersection_contours = intersection_contours[0] if len(intersection_contours) > 0 else intersection_contours[1]


    # final_contours = np.intersect1d(intersection_contours, mask_contours)
    # print(final_contours)

    for cntr in intersection_contours:
        x, y, w, h = cv2.boundingRect(cntr)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # print("x,y,w,h:", x, y, w, h)

    # Show masks/frames
    cv2.imshow('Frame', frame)
    # cv2.imshow('FG Mask', fgmask)
    # cv2.imshow('Color Mask', mask)
    cv2.imshow('Intersection Frame', intersection_frame)
    cv2.imshow('Contours', result)

    if cv2.waitKey(30) & 0xff == ord('q'):
        break # DON'T DELETE THIS OR THE FRAMES WON'T SHOW
