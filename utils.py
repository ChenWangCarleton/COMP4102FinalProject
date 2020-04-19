from numpy import ones,vstack
from numpy.linalg import lstsq

import cv2
import numpy as np

left_k = [-0.699,-0.674]
left_b = [900.50,889.79]
right_k = [0.672, 0.675]
right_b = [-500.98, -503.67]

def get_point_info(x1, y1, x2, y2):
    point = [(x1,y1),(x2,y2)]
    x_coords, y_coords = zip(*point)
    A = vstack([x_coords, ones(len(x_coords))]).T
    k, b = lstsq(A, y_coords)[0]
    return [point, [k, b]]

def get_kb(x1,y1,x2,y2):
    point = [(x1,y1),(x2,y2)]
    x_coords, y_coords = zip(*point)
    A = vstack([x_coords, ones(len(x_coords))]).T
    k, b = lstsq(A, y_coords)[0]
    return k,b

def filter_borders(point_info_list,threshold=0.02):
    global left_k, left_b, right_k, right_b
    possible_left_ind = -1
    possible_right_ind = -1

    avg_left_k = sum(left_k)/len(left_k)
    avg_right_k = sum(right_k)/len(right_k)
    for i in range(0, len(point_info_list)):
        point = point_info_list[i]

        if abs(point[1][0] - avg_left_k) <= threshold:
            if possible_left_ind < 0:
                possible_left_ind = i
            elif point[1][1] < point_info_list[possible_left_ind][1][1]:
                #only keep the most outer line
                possible_left_ind = i

        elif abs(point[1][0] - avg_right_k) <= threshold:
            if possible_right_ind < 0:
                possible_right_ind = i
            elif point[1][1] < point_info_list[possible_right_ind][1][1]:
                #only keep the most outer line
                possible_right_ind = i
    to_return = []
    if possible_left_ind>-1:
        to_return.append(point_info_list[possible_left_ind][0])
    if possible_right_ind>-1:
        to_return.append(point_info_list[possible_right_ind][0])
    return to_return

def filter_borders_extend(point_info_list, img_width,threshold=0.02):
    # this version expands the border to x=0 and y=0
    global left_k, left_b, right_k, right_b
    possible_left_ind = -1
    possible_right_ind = -1

    avg_left_k = sum(left_k)/len(left_k)
    avg_right_k = sum(right_k)/len(right_k)
    for i in range(0, len(point_info_list)):
        point = point_info_list[i]

        if abs(point[1][0] - avg_left_k) <= threshold:
            if possible_left_ind < 0:
                possible_left_ind = i
            elif point[1][1] < point_info_list[possible_left_ind][1][1]:
                #only keep the most outer line
                possible_left_ind = i

        elif abs(point[1][0] - avg_right_k) <= threshold:
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


def calculate_avg_width_height_std(rec_infos):

    avg_width = 0
    avg_height = 0
    for item in rec_infos:
        if item[4] != 6 or item[4]!=0:
            avg_width += item[2]
            avg_height += item[3]
    avg_width= int(avg_width/7)
    avg_height= int(avg_height/7)

    width_variance = 0
    height_variance = 0
    for item in rec_infos:
        if item[4] != 6 or item[4]!=0:
            width_variance = (item[2]-avg_width)**2

            height_variance = (item[3]-avg_width)**2
    width_variance/=7
    height_variance/=7
    width_std = int(np.sqrt(width_variance))
    height_std = int(np.sqrt(height_variance))
    print('avg width: ',avg_width,' avg height: ', avg_height, ' width std: ', width_std, ' height std: ', height_std)
    return [avg_width, avg_height, width_std, height_std]
def get_rec_info(boxes):
    rec_infos = []
    id_counter = 0
    for box in boxes:
        #print(box)
        rec_infos.append([int((box[0]+box[2])/2), int((box[1]+box[3])/2), box[2]-box[0], box[3] - box[1], id_counter, box])
        id_counter+=1
    for item in rec_infos:
        print('central point :  ', (item[0], item[1]), " width: ",item[2], "height: ", item[3], " id: ", item[4])

    return rec_infos


