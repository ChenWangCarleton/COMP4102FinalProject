import numpy as np
import cv2



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
"""
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.imread('for_calculating_width_height_std.PNG')

# the output will be written to output.avi
frame =cap.copy()

gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
rec_infos = get_rec_info(boxes)
width_height_std = calculate_avg_width_height_std(rec_infos)
for (xA, yA, xB, yB) in boxes:
    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
for item in rec_infos:
    cv2.circle(frame,(item[0], item[1]), int(item[2]/2), (0,0,255))
    cv2.putText(frame,'id: '+str(item[4]),(item[0], item[1]),cv2.FONT_HERSHEY_DUPLEX, 1, (127,255,127), 2, cv2.LINE_AA)

    # Display the resulting frame
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
# When everything done, release the capture"""
