import cv2
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq

import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt

def combine_line(point_info_list, min_lenth=300, min_same_lines=2, max_diff_k = 0.01, max_diff_b = 20):
    grouped_lines = []
    for point_info in point_info_list:
        placed = False
        print('k',point_info[1][0])
        for current_group in grouped_lines:
            if abs(current_group[0][0] - point_info[1][0]) <= max_diff_k and abs(current_group[0][1] - point_info[1][1]) <= max_diff_b:
                current_group[1].append(point_info[0][0])
                current_group[1].append(point_info[0][1])
                current_group[0][0] = (current_group[0][0] + point_info[1][0])/len(current_group[1]) # new average k
                current_group[0][1] = (current_group[0][1] + point_info[1][1])/len(current_group[1]) # new average b
                placed = True
                break

        if not placed:
            temp = []
            point_list = []
            temp.append(point_info[1])
            point_list.append(point_info[0][0])
            point_list.append(point_info[0][1])
            temp.append(point_list)
            grouped_lines.append(temp)
    final_lines = []
    for current_group in grouped_lines:
        if len(current_group[1]) < min_same_lines:
            continue
        min_x = current_group[1][0][0]
        min_y = current_group[1][0][1]
        max_x = current_group[1][0][0]
        max_y = current_group[1][0][1]
        #print(min_x)
        for i in range(1, len(current_group[1])):
            if current_group[1][i][0] > max_x:
                max_x = current_group[1][i][0]
                max_y = current_group[1][0][1]
            if current_group[1][i][0] < min_x:
                min_x = current_group[1][i][0]
                min_y = current_group[1][0][1]
        if (max_x - min_x) * (max_x - min_x) + (max_y - min_y) * (max_y - min_y) < min_lenth*min_lenth:
            continue
        else:
            final_lines.append([min_x,min_y,max_x,max_y])
            print('k=',get_point_info(min_x,min_y,max_x,max_y)[1][0])
    print("total final lines: ", len(final_lines))
    return final_lines
def get_furtherest(final_lines, min_diff_b=400):
    # get furtherest two lines based on b
    temp = []

    for line in final_lines:
        temp.append(get_point_info(line[0],line[1],line[2],line[3]))
    min_ind = 0
    max_ind = 0
    cur_min = temp[0][1][1]
    cur_max =  temp[0][1][1]
    for i in range(1, len(temp)):
        if temp[i][1][1] < cur_min:
            min_ind = i
            cur_min = temp[i][1][1]
        elif temp[i][1][1] > cur_max:
            max_ind = i
            cur_max = temp[i][1][1]
    if cur_max-cur_min < min_diff_b:
        print("max distance between upper and lower boundary smaller than predefined threshold")
    return [temp[min_ind][0], temp[max_ind][0]]
def get_point_info(x1, y1, x2, y2):
    point = [(x1,y1),(x2,y2)]
    x_coords, y_coords = zip(*point)
    A = vstack([x_coords, ones(len(x_coords))]).T
    k, b = lstsq(A, y_coords)[0]
    return [point, [k, b]]
# Reading the input image
img_ori = cv2.imread('Capture.PNG',)

image = img_ori
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

img_edge = img_bw
temp_img = img_ori.copy()
edges = cv2.Canny(img_edge,75,150)

lines  = cv2.HoughLinesP(edges, 1, np.pi/180, 50,minLineLength=100,   maxLineGap=30)

point_infos = []

for line in lines:
    x1, y1, x2, y2 = line[0]
    point_infos.append(get_point_info(x1,y1,x2,y2))
    cv2.line(temp_img, (x1,y1), (x2,y2),(0,255,0),)
final_lines = combine_line(point_infos)
final_lines = get_furtherest(final_lines)
p1, p2 = final_lines[0]
print('p1:', p1, ' p2: ', p2)
cv2.line(temp_img, p1, p2, (255, 0, 0), )
p1, p2 = final_lines[1]
print('p1:', p1, ' p2: ', p2)
cv2.line(temp_img, p1, p2, (0, 0, 255), )
"""for line in final_lines:
    p1, p2 = line
    print('p1:',p1,' p2: ',p2)
    cv2.line(temp_img, p1, p2,(255,0,0),)"""

    #x1, y1, x2, y2 = line
    #cv2.line(temp_img, (x1,y1), (x2,y2),(255,0,0),)


cv2.imshow('bw edge image', temp_img)

img_edge = img_erosion
temp_img = img_ori.copy()
edges = cv2.Canny(img_edge,75,150)

lines  = cv2.HoughLinesP(edges, 1, np.pi/180, 50,   maxLineGap=30)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(temp_img, (x1,y1), (x2,y2),(0,255,0),)

cv2.imshow('erosion edge image', temp_img)


img_edge = img_dilation
temp_img = img_ori.copy()
edges = cv2.Canny(img_edge,75,150)

lines  = cv2.HoughLinesP(edges, 1, np.pi/180, 50,   maxLineGap=30)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(temp_img, (x1,y1), (x2,y2),(0,255,0),)

cv2.imshow('dilation edge image', temp_img)




cv2.waitKey(0)