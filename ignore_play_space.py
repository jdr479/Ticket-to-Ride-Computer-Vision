import cv2
import numpy as np

LOWER_RED_1 = np.array([0, 50, 50])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 50, 50])
UPPER_RED_2 = np.array([180, 255, 255])

LOWER_YELLOW = np.array([22, 105, 55])
UPPER_YELLOW = np.array([30, 231, 255])

LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 13])

LOWER_BLUE = np.array([78, 158, 87])
UPPER_BLUE = np.array([156, 255, 255])

LOWER_GREEN = np.array([56, 93, 40])
UPPER_GREEN = np.array([96, 255, 255])

# Bounds set to blue by default
lower_bound = LOWER_BLUE
upper_bound = UPPER_BLUE

cap = cv2.VideoCapture(0)

while True:
    """_, frame = cap.read()
    H = cv2.getTrackbarPos('H', 'Frame:')
    S = cv2.getTrackbarPos('S', 'Frame:')
    V = cv2.getTrackbarPos('V', 'Frame:')
    H2 = cv2.getTrackbarPos('H2', 'Frame:')
    S2 = cv2.getTrackbarPos('S2', 'Frame:')
    V2 = cv2.getTrackbarPos('V2', 'Frame:')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_boundary = np.array([H, S, V])
    upper_boundary = np.array([H2, S2, V2])
    mask = cv2.inRange(hsv, lower_boundary, upper_boundary)
    # final = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Frame:", mask)

    if cv2.waitKey(1) == ord('q'): break"""

    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Contour Test
    result = frame.copy()
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) > 0 else contours[1]
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # print("x,y,w,h:", x, y, w, h)

    cv2.imshow('mask', mask)
    # cv2.imshow('rect_frame', rect_frame)
    cv2.imshow('contours', result)

    if cv2.waitKey(30) & 0xff == ord('q'):
        break # DON'T DELETE THIS OR THE FRAMES WON'T SHOW

cap.release()
cv2.destroyAllWindows()
