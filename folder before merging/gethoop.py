import numpy as np
import cv2

# load the image
img_src = "balldetectiontest.PNG"
image = cv2.imread(img_src, 1)

# red color boundaries [B, G, R]
lower = [161, 75, 68]
upper = [252, 255, 255]

img=image
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array(lower)
upper_red = np.array(upper)
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my masks
mask = mask0+mask1

# set my output img to zero everywhere except my mask
output_img = img.copy()
output_img[np.where(mask==0)] = 0

# or your HSV image, which I *believe* is what you want
output_hsv = img_hsv.copy()
output_hsv[np.where(mask==0)] = 0
cv2.imshow('red hoop',output_hsv)
cv2.waitKey(0)