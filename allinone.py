import  numpy as np
import cv2
import utils
from PIL import Image, ImageFilter

avg_width = 187
avg_height = 375
width_std = 41
height_std = 13

min_v_border = 10
lower_hsv = [0,0,255]
upper_hsv = [255,255,255]



filter_mode = 0 # 0 is default
supporting_line = 2 # 0 for showing nothing, 1 for left right border, 2 plus upper lower border, 3 plus player box, 4 plus ball locations, 5 plus unprocessed supporting lines

frame_delay=10

def apply_effect(img, boxes, mode):
    if mode == 0:
        return img
    im = Image.fromarray((img).astype(np.uint8))

    filter = ''
    if mode == 1:
        print("blur")
        filter = ImageFilter.GaussianBlur(radius=10)
    elif mode ==2:
        print("sharp")
        filter = ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
    elif mode ==3:
        print("kernel effect")
        km = (
            -2, -1, 0,
            -1, 1, 1,
            0, 1, 2
        )
        filter = ImageFilter.Kernel(
            size=(3, 3),
            kernel=km,
            scale=sum(km),  # default
            offset=0  # default
        )
    elif mode ==4:
        print("Contour effect")
        filter = ImageFilter.CONTOUR
    elif mode ==5:
        print("EMBOSS effect")
        filter = ImageFilter.EMBOSS
    elif mode ==6:
        print("edge effect")
        filter = ImageFilter.FIND_EDGES
    else:
        print("filter not implemented")
        return img
    for box in boxes:
        crop_img = im.crop(box)
        blur_image = crop_img.filter(filter)
        im.paste(blur_image, box)

    return np.array(im)

def show_supporting_line(x):
    global supporting_line
    supporting_line = x

def minleftright(x):
    global  min_v_border
    min_v_border = x

def getfilter(x):
    global  filter_mode
    filter_mode = x

def lowerhsv_h(x):
    global  lower_hsv
    lower_hsv[0] = x

def lowerhsv_s(x):
    global  lower_hsv
    lower_hsv[1] = x

def lowerhsv_v(x):
    global  lower_hsv
    lower_hsv[2] = x

def upperhsv_h(x):
    global  upper_hsv
    upper_hsv[0] = x

def upperhsv_s(x):
    global  upper_hsv
    upper_hsv[1] = x

def upperhsv_v(x):
    global  upper_hsv
    upper_hsv[2] = x

def change_frame_delay(x):
    global frame_delay
    frame_delay = x
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, lowerhsv_h)
cv2.createTrackbar("LS", "Tracking", 0, 255, lowerhsv_s)
cv2.createTrackbar("LV", "Tracking", 255, 255, lowerhsv_v)
cv2.createTrackbar("UH", "Tracking", 255, 255, upperhsv_h)
cv2.createTrackbar("US", "Tracking", 255, 255, upperhsv_s)
cv2.createTrackbar("UV", "Tracking", 255, 255, upperhsv_v)
cv2.createTrackbar("border_min_length", "Tracking", 10, 200, minleftright)
cv2.createTrackbar("filter_mode", "Tracking", 0, 6, getfilter)
cv2.createTrackbar("show_supporting_line", "Tracking", 2, 5, show_supporting_line)
cv2.createTrackbar("frame_delay", "Tracking", 10, 20, change_frame_delay)


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()


cap = cv2.VideoCapture("nbaclip_1.mp4")

last_lower_y=-1
last_upper_y=-1


def get_ball(circles, lower_y, upper_y):
    if len(circles) == 1:
        return circles[0]

    avg_y = (lower_y+upper_y)/2

    ball_ind = 0
    min_diif_y = abs(circles[0][1]-avg_y)

    for i in range(1,len(circles)):
        if abs(circles[i][1]-avg_y)< min_diif_y:
            ball_ind = i
            min_diif_y = abs(circles[i][1]-avg_y)
    return circles[ball_ind]

apply_filter = False
current_countdown = 0

ori_img_width = 0
ori_immg_height = 0
while True:
    if current_countdown == 0:
        apply_filter = False
    else:
        print("filter delay effect")
        current_countdown -=1
    player_box = []
    leftright_border = []
    balls = []
    to_filter = []

    _, frame = cap.read()
    if frame is None:
        print("video end")
        break
    #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    img_height, img_width, _ = frame.shape


    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # getting upper lower borders
    img_hue = hsvImage[:, :, 0]
    (threshi, img_bw) = cv2.threshold(img_hue, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    img_edge = img_bw
    temp_img = frame.copy()
    edges = cv2.Canny(img_edge, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=30)

    point_infos = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        point_infos.append(utils.get_point_info(x1, y1, x2, y2))
        if supporting_line == 5:
            cv2.line(temp_img, (x1, y1), (x2, y2), (0, 255, 0), )
    final_lines = utils.combine_line(point_infos)
    if len(final_lines) >2:
        final_lines = utils.get_furtherest_v3(final_lines)
        last_lower_y = final_lines[0]
        last_upper_y = final_lines[1]
        for y in final_lines:
            # print(y)
            if supporting_line >1:
                cv2.line(temp_img, (0, y), (img_width - 1, y), (255, 0, 0), )
    # input("pause")
    else:
        print("Lost track of the upper and lower boundary")

    # getting player boxes


    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    rec_infos = utils.get_rec_info(boxes)
    for i in range(len(rec_infos)):
        if rec_infos[i][1]<last_lower_y - 2 * height_std:
            continue
        else:
            player_box.append(rec_infos[i])
            to_filter.append(boxes[i])
    """    for item in rec_infos:
        if item[1]<last_lower_y - 2 * height_std:
            continue
        #elif abs(item[2] - avg_width) > 3*height_std:
            continue
       # elif abs(item[3] - avg_height) > 3*width_std:
            continue
        else:
            player_box.append(item)"""


    #width_height_std = calculate_avg_width_height_std(rec_infos)
    for item in player_box:
        box = item[5]
        xA, yA, xB, yB = box
        if supporting_line>2:
            cv2.rectangle(temp_img, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cv2.circle(temp_img, (item[0], item[1]), int(item[2] / 2), (0, 0, 255))
            cv2.putText(temp_img, 'id: ' + str(item[4]), (item[0], item[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (127, 255, 127), 2,
                        cv2.LINE_AA)

    # getting left right border
    l_b=np.array(lower_hsv)
    u_b=np.array(upper_hsv)


    mask = cv2.inRange(hsvImage, l_b, u_b)

    temp = cv2.bitwise_and(frame.copy(), frame.copy(), mask=mask)

    edges = cv2.Canny(temp, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=min_v_border, maxLineGap=60)

    try:
        point_info_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            point_info_list.append(utils.get_point_info(x1,y1,x2,y2))
        borders = utils.filter_borders_extend(point_info_list, img_width)
        for border in borders:
            if supporting_line>0:
                cv2.line(temp_img, border[0], border[1], (0, 255, 0), )

    except:
        print('no left or right line found')

    # get ball location

    l_b_c = np.array([0, 0, 0])
    u_b_c = np.array([255, 255, 232])
    mask = cv2.inRange(hsvImage, l_b_c, u_b_c)

    res = cv2.bitwise_and(frame.copy(), frame.copy(), mask=mask)
    try:
        img = cv2.medianBlur(res, 5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=3, maxRadius=30)

        circles = np.uint16(np.around(circles))
        circle_list = []
        for i in circles[0, :]:
            # draw the outer circle

            if i[2] < 12: #radius
                circle_list.append([i[0], i[1], i[2]])

        circle = get_ball(circle_list,last_lower_y,last_upper_y)
        balls.append(circle)
        if supporting_line>3:
            cv2.circle(temp_img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            cv2.putText(temp_img, 'radius: ' + str(circle[2]), (circle[0], circle[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (127, 255, 127),
                        2, cv2.LINE_AA)
            # draw the center of the circle
            cv2.circle(temp_img, (circle[0], circle[1]), 2, (0, 0, 255), 3)

        #cv2.imshow('detected circles', img)
    except:
        print("no circle found")


    if len(balls) == 1 or len(borders) >0:
        current_countdown = frame_delay
        apply_filter = True

    if apply_filter:
        print("apply filter!!!")

        temp_img = apply_effect(temp_img, to_filter, filter_mode)
    res = temp_img

    cv2.imshow("frame", frame)

    cv2.imshow("res", res)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()