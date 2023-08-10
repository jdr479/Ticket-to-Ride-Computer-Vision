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

def get_avg_position(white_pixel_pos, width, height):
    x_avg = 0
    y_avg = 0

    for pixel in white_pixel_pos:
        x_avg += pixel[1]
        y_avg += pixel[0]

    x_avg = int(x_avg / white_pixel_pos.size)
    y_avg = int(y_avg / white_pixel_pos.size)
    print(x_avg, y_avg)

    return x_avg * 2, y_avg * 2


def reject_outliers(data, m = 2):
    mean = np.mean(data)
    standard_deviation = np.std(data)
    distance_from_mean = abs(data - mean)
    max_deviations = 2
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    no_outliers = data[not_outlier]
    return no_outliers


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

    if cv2.waitKey(1) == ord('r'):
        print("Switching to red")
        lower_bound = LOWER_RED_1 + LOWER_RED_2
        upper_bound = UPPER_RED_1 + UPPER_RED_2
    elif cv2.waitKey(1) == ord('g'):
        print("Switching to green")
        lower_bound = LOWER_GREEN
        upper_bound = UPPER_GREEN
    elif cv2.waitKey(1) == ord('b'):
        print("Switching to blue")
        lower_bound = LOWER_BLUE
        upper_bound = UPPER_BLUE
    elif cv2.waitKey(1) == ord('l'):
        print("Switching to black")
        lower_bound = LOWER_BLACK
        upper_bound = UPPER_BLACK
    elif cv2.waitKey(1) == ord('y'):
        print("Switching to yellow")
        lower_bound = LOWER_YELLOW
        upper_bound = UPPER_YELLOW

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Contour Test
    result = frame.copy()
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) > 0 else contours[1]
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # print("x,y,w,h:", x, y, w, h)

    """indices = np.where(morph != [0])
    coordinates = zip(reject_outliers(indices[0]), indices[1])
    white_pixel_pos = np.asarray(list(coordinates))

    if white_pixel_pos.size != 0: x, y = get_avg_position(white_pixel_pos, width, height)
    else: x, y = 100, 100

    rect_frame = cv2.circle(frame, (x, y), 50, (128, 128, 128), 5)"""

    cv2.imshow('mask', morph)
    # cv2.imshow('rect_frame', rect_frame)
    cv2.imshow('contours', result)

cap.release()
cv2.destroyAllWindows()
