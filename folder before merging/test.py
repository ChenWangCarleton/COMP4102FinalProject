import cv2
import numpy as np


# Initiate video capture for video file, here we are using the video file in which pedestrians would be detected
def nothing(x):
    pass

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
cap = cv2.VideoCapture("nbaclip_1.mp4")
last_lower_y=-1
last_upper_y=-1
#avg width:  187  avg height:  375  width std:  41  height std:  13
while True:
    #frame = cv2.imread('smarties.png')
    player_box = []
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_b, u_b)

    res = cv2.bitwise_and(frame, frame, mask=mask)
    img = cv2.medianBlur(res, 5)
    try:

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=3, maxRadius=30)

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.putText(img, 'radius: ' + str(i[2]), (i[0], i[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (127, 255, 127),
                        2, cv2.LINE_AA)
            # draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)


    except:
        print("no circle found")
    cv2.imshow('detected circles', img)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)
    #cv2.waitKey(0)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()