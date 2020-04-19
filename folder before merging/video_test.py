import cv2
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
from get_width_height_std import calculate_avg_width_height_std, get_rec_info


def combine_line(point_info_list, min_lenth=500, min_same_lines=2, max_diff_k = 0.1, max_diff_b = 20):
    grouped_lines = []
    for point_info in point_info_list:
        placed = False
        #print('k',point_info[1][0])
        for current_group in grouped_lines:
            if abs(current_group[0][0] - point_info[1][0]) <= max_diff_k and abs(current_group[0][1] - point_info[1][1]) <= max_diff_b:
                current_group[1].append(point_info[0][0])
                current_group[1].append(point_info[0][1])
                current_group[0][0] = (current_group[0][0] + point_info[1][0])/len(current_group[1]) # new average k
                current_group[0][1] = (current_group[0][1] + point_info[1][1])/len(current_group[1]) # new average b
                placed = True
                #break

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
    #print("total final lines: ", len(final_lines))
    return final_lines

def get_furtherest_v3(final_lines):
    # simplified to return only the lower and upper y coordinates as boundary
    min_ind = 0
    max_ind = 0
    cur_min_ave_y = (final_lines[0][1] + final_lines[0][3])/2

    cur_max_ave_y = (final_lines[0][1] + final_lines[0][3])/2

    for i in range(1, len(final_lines)):
        if (final_lines[i][1] + final_lines[i][3])/2 < cur_min_ave_y:
            min_ind = i
            cur_min_ave_y = (final_lines[i][1] + final_lines[i][3])/2
        elif (final_lines[i][1] + final_lines[i][3])/2 > cur_max_ave_y:
            max_ind = i
            cur_max_ave_y =  (final_lines[i][1] + final_lines[i][3])/2

    return [int((final_lines[min_ind][1] + final_lines[min_ind][3])/2), int((final_lines[max_ind][1] +final_lines[max_ind][3])/2)]
def get_furtherest_v2(final_lines):
    # return only the furtherest two lines denotes the upper and lower boundary found.
    min_ind = 0
    max_ind = 0
    cur_min_ave_y = (final_lines[0][1] + final_lines[0][3])/2

    cur_max_ave_y = (final_lines[0][1] + final_lines[0][3])/2

    for i in range(1, len(final_lines)):
        if (final_lines[i][1] + final_lines[i][3])/2 < cur_min_ave_y:
            min_ind = i
            cur_min_ave_y = (final_lines[i][1] + final_lines[i][3])/2
        elif (final_lines[i][1] + final_lines[i][3])/2 > cur_max_ave_y:
            max_ind = i
            cur_max_ave_y =  (final_lines[i][1] + final_lines[i][3])/2
    line1 = [(final_lines[min_ind][0],final_lines[min_ind][1]),(final_lines[min_ind][2],final_lines[min_ind][3])]
    line2 = [(final_lines[max_ind][0],final_lines[max_ind][1]),(final_lines[max_ind][2],final_lines[max_ind][3])]
    return [line1, line2]
def get_furtherest(final_lines, min_diff_b=300):
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


avg_width = 187
avg_height = 375
width_std = 41
height_std = 13
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

cap = cv2.VideoCapture("nbaclip_1.mp4")
last_lower_y=-1
last_upper_y=-1
#avg width:  187  avg height:  375  width std:  41  height std:  13
while True:
    #frame = cv2.imread('smarties.png')
    player_box = []
    _, frame = cap.read()
    img_height, img_width, _ = frame.shape

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    img_hue = hsvImage[:, :, 0]
    (threshi, img_bw) = cv2.threshold(img_hue, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    img_edge = img_bw
    temp_img = frame.copy()
    edges = cv2.Canny(img_edge, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=30)

    point_infos = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        point_infos.append(get_point_info(x1, y1, x2, y2))
        cv2.line(temp_img, (x1, y1), (x2, y2), (0, 255, 0), )
    final_lines = combine_line(point_infos)
    if len(final_lines) >2:
        final_lines = get_furtherest_v3(final_lines)
        #final_lines = get_furtherest_v2(final_lines)
        #final_lines = get_furtherest(final_lines)


        last_lower_y = final_lines[0]
        last_upper_y = final_lines[1]
        for y in final_lines:
            #print(y)
            cv2.line(temp_img, (0,y),(img_width-1,y), (255, 0, 0), )
       # input("pause")
    else:
        print("Lost track of the upper and lower boundary")


    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    rec_infos = get_rec_info(boxes)
    for item in rec_infos:
        if item[1]<last_lower_y - 2 * height_std:
            continue
        #elif abs(item[2] - avg_width) > 3*height_std:
            continue
       # elif abs(item[3] - avg_height) > 3*width_std:
            continue
        else:
            player_box.append(item)


    #width_height_std = calculate_avg_width_height_std(rec_infos)
    for item in player_box:
        box = item[5]
        xA, yA, xB, yB = box
        cv2.rectangle(temp_img, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.circle(temp_img, (item[0], item[1]), int(item[2] / 2), (0, 0, 255))
        cv2.putText(temp_img, 'id: ' + str(item[4]), (item[0], item[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (127, 255, 127), 2,
                    cv2.LINE_AA)

    """for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(temp_img, (xA, yA), (xB, yB), (0, 255, 0), 2)
    for item in rec_infos:
        cv2.circle(temp_img, (item[0], item[1]), int(item[2] / 2), (0, 0, 255))
        cv2.putText(temp_img, 'id: ' + str(item[4]), (item[0], item[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (127, 255, 127), 2,
                    cv2.LINE_AA)"""

    res = temp_img

    cv2.imshow("frame", frame)

    cv2.imshow("res", res)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()