import cv2
import numpy as np

hsv_list = [0,0,0,0,0,0]
# [lh,ls,lv,uh,us,uv]

def nothing(x):
    print(' x is:', x)

def LH_track(x):
    global hsv_list
    hsv_list[0] = x
    print(' LH is:', x)
def LS_track(x):
    global hsv_list
    hsv_list[1] = x
    print(' LS is:', x)
def LV_track(x):
    global hsv_list
    hsv_list[2] = x
    print(' LV is:', x)
def UH_track(x):
    global hsv_list
    hsv_list[3] = x
    print(' UH is:', x)
def US_track(x):
    global hsv_list
    hsv_list[4] = x
    print(' US is:', x)
def UV_track(x):
    global hsv_list
    hsv_list[5] = x
    print(' UV is:', x)
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, LH_track)
cv2.createTrackbar("LS", "Tracking", 0, 255, LS_track)
cv2.createTrackbar("LV", "Tracking", 0, 255, LV_track)
cv2.createTrackbar("UH", "Tracking", 255, 255, UH_track)
cv2.createTrackbar("US", "Tracking", 255, 255, US_track)
cv2.createTrackbar("UV", "Tracking", 255, 255, UV_track)

# Reading the input image
img_ori = cv2.imread('Capture.PNG',)

image = img_ori
cap = cv2.VideoCapture("nbaclip_1.mp4")

while True:
    frame = cv2.imread('Capture.PNG')
    #_, frame = cap.read()
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

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)

    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # draw filled contour on result
    for cnt in contours:
        cv2.drawContours(res, [cnt], 0, (0, 0, 255), 2)
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100:
            continue

        x, y, w, h = rect
        cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # detect edges in mask
    edges = cv2.Canny(mask, 100, 100)
    # to save an image use cv2.imwrite('filename.png',img)
    # show images
    cv2.imshow("Result_with_contours", res)
    cv2.imshow("Mask", mask)
    cv2.imshow("Edges", edges)


    key = cv2.waitKey(1)
    if key == 27:
        break



hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow('Original image',image)
cv2.imshow('HSV image', hsvImage)

cv2.waitKey(0)
cv2.destroyAllWindows()
hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
img_hue = hsvImage[:,:,0]
print(hsvImage.shape)
cv2.imshow('Original image', image)
cv2.imshow('HSV image', hsvImage)

cv2.imshow('Hue image', img_hue)








#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(threshi, img_bw) = cv2.threshold( img_hue, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow('bw image', img_bw)



cv2.waitKey(0)
cv2.destroyAllWindows()

# Taking a matrix of size 5 as the kernel
kernel = np.ones((5, 5), np.uint8)

# The first parameter is the original image,
# kernel is the matrix with which image is
# convolved and third parameter is the number
# of iterations, which will determine how much
# you want to erode/dilate a given image.
img_erosion = cv2.erode(img_hue, kernel, iterations=1)
img_dilation = cv2.dilate(img_hue, kernel, iterations=1)

cv2.imshow('Input', img_hue)
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()