
from numpy import ones,vstack
from numpy.linalg import lstsq
import cv2
import numpy as np
import utils
import traceback

left_k = [-0.699,-0.674]
left_b = [900.50,889.79]
right_k = [0.672, 0.675]
right_b = [-500.98, -503.67]
def get_kb(x1,y1,x2,y2):
    point = [(x1,y1),(x2,y2)]
    x_coords, y_coords = zip(*point)
    A = vstack([x_coords, ones(len(x_coords))]).T
    k, b = lstsq(A, y_coords)[0]
    return k,b
def filter_borders_extend(point_info_list,img_width,left_threshold=0.02, right_threshold=0.05):
    global left_k, left_b, right_k, right_b
    possible_left_ind = -1
    possible_right_ind = -1

    avg_left_k = sum(left_k)/len(left_k)
    avg_right_k = sum(right_k)/len(right_k)
    for i in range(0, len(point_info_list)):
        point = point_info_list[i]

        if abs(point[1][0] - avg_left_k) <= left_threshold:
            if possible_left_ind < 0:
                possible_left_ind = i
            elif point[1][1] < point_info_list[possible_left_ind][1][1]:
                #only keep the most outer line
                possible_left_ind = i

        elif abs(point[1][0] - avg_right_k) <= right_threshold:
            if possible_right_ind < 0:
                possible_right_ind = i
            elif point[1][1] < point_info_list[possible_right_ind][1][1]:
                #only keep the most outer line
                possible_right_ind = i
    to_return = []
    if possible_left_ind>-1:

        k = point_info_list[possible_left_ind][1][0]
        b = point_info_list[possible_left_ind][1][1]
        point = [(0, int(b)),(int(-b/k),0)]
        to_return.append(point)
    if possible_right_ind>-1:
        k = point_info_list[possible_right_ind][1][0]
        b = point_info_list[possible_right_ind][1][1]
        point = [(img_width, int(img_width*k+b)),(int(-b/k),0)]
        to_return.append(point)
    return to_return

def get_point_info(x1, y1, x2, y2):
    point = [(x1,y1),(x2,y2)]
    x_coords, y_coords = zip(*point)
    A = vstack([x_coords, ones(len(x_coords))]).T
    k, b = lstsq(A, y_coords)[0]
    return [point, [k, b]]
def nothing(x):
    pass
min_v_border = 5
def minleftright(x):
    global  min_v_border
    min_v_border = x
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
cv2.createTrackbar("LFborder", "Tracking", 0, 200, minleftright)
#img_src = 'rightborderslop.PNG'
img_src = 'leftborder.PNG'
cv2.imread('Capture.PNG',)

cap = cv2.VideoCapture("nbaclip_1.mp4")
while True:
    frame = cv2.imread(img_src)
    _, frame = cap.read()
    img_height, img_width, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, 255])
    u_b = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_b, u_b)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    edges = cv2.Canny(res, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=min_v_border, maxLineGap=100)


    temp_img = res.copy()
    try:
        point_info_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            point_info_list.append(utils.get_point_info(x1,y1,x2,y2))
        borders = filter_borders_extend(point_info_list,img_width)
        for border in borders:
            cv2.line(res, border[0], border[1], (0, 255, 0), )

        point_infos = []
        #print('line num', len(lines))

        point_info = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            point_info.append(get_point_info(x1,y1,x2,y2))
        avg_right_k = sum(right_k) / len(right_k)
        counter =0
        for line in lines:
            point= point_info.pop(0)
            x1, y1, x2, y2 = line[0]
            if abs(point[1][0] - avg_right_k) < 0.05:
                counter +=1
                cv2.line(temp_img, (x1, y1), (x2, y2), (0, 255, 0), )
                cv2.putText(temp_img, 'k: ' + str(point[1][0])+ ' b: '+ str(point[1][1]), (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 1, (127, 255, 127), 2,cv2.LINE_AA)
                print('k: ' + str(point[1][0])+ ' b: '+ str(point[1][1]))
        print('total right border line: ',counter)
    except:
        print(traceback.format_exc())
    cv2.putText(mask, '0,0', (0,0), cv2.FONT_HERSHEY_DUPLEX, 1,
                (127, 255, 127), 2, cv2.LINE_AA)
    cv2.putText(mask, 'x=100,0', (100,0), cv2.FONT_HERSHEY_DUPLEX, 1,
                (127, 255, 127), 2, cv2.LINE_AA)
    cv2.putText(mask, '0,y=100', (0,100), cv2.FONT_HERSHEY_DUPLEX, 1,
                (127, 255, 127), 2, cv2.LINE_AA)


    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)
    cv2.imshow("left", temp_img)
    #cv2.waitKey(0)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()