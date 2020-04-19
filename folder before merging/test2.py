import numpy as np
import cv2
from PIL import Image, ImageFilter
from matplotlib import cm

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.imread('for_calculating_width_height_std.PNG')

# the output will be written to output.avi

def calculate_avg_width_height_std(boxes):
    central_width_height = []
    id_counter = 0
    for box in boxes:
        #print(box)
        central_width_height.append([int((box[0]+box[2])/2), int((box[1]+box[3])/2), box[2]-box[0], box[3] - box[1], id_counter])
        id_counter+=1
    for item in central_width_height:
        print('central point :  ', (item[0], item[1]), " width: ",item[2], "height: ", item[3], " id: ", item[4])

    avg_width = 0
    avg_height = 0
    for item in central_width_height:
        if item[4] != 6 or item[4]!=0:
            avg_width += item[2]
            avg_height += item[3]
    avg_width= int(avg_width/7)
    avg_height= int(avg_height/7)

    width_variance = 0
    height_variance = 0
    for item in central_width_height:
        if item[4] != 6 or item[4]!=0:
            width_variance = (item[2]-avg_width)**2

            height_variance = (item[3]-avg_width)**2
    width_variance/=7
    height_variance/=7
    width_std = int(np.sqrt(width_variance))
    height_std = int(np.sqrt(height_variance))
    print('avg width: ',avg_width,' avg height: ', avg_height, ' width std: ', width_std, ' height std: ', height_std)
    return central_width_height
while (True):
    # Capture frame-by-frame
   # _, frame = cap.read()
    frame =cap.copy()
    # resizing for faster detection
    #frame = cv2.resize(frame, (640, 360))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    centroids = calculate_avg_width_height_std(boxes)
    im = Image.fromarray((frame ).astype(np.uint8))
    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        #cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        box = (xA, yA, xB, yB)
        crop_img = im.crop(box)
        blur_image = crop_img.filter(ImageFilter.GaussianBlur(radius=10))
        im.paste(blur_image, box)

    frame = np.array(im)
    for item in centroids:
        cv2.circle(frame,(item[0], item[1]), int(item[2]/2), (0,0,255))
        cv2.putText(frame,'id: '+str(item[4]),(item[0], item[1]),cv2.FONT_HERSHEY_DUPLEX, 1, (127,255,127), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    #im.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# When everything done, release the capture
cap.release()

# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)