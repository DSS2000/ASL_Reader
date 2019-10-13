# organize imports
import cv2
import imutils
import numpy as np
import time
import json
from PIL import Image
from prediction import get_prediction
# global variables
bg = None
list = [];

def top():
    aWeight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 225, 300, 475
    num_frames = 0
    count = 0
    t = time.time();
    s = "First"
    X = '0'

    while(True):
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[10:300, 225:475]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            ret = ""
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                # cv2.imshow("Thesholded", roi)
                __draw_label(clone, 'Count is = ' + str(num_frames), (20,20), (255,255,255))
                if (num_frames % 40 == 0):
                    count = count + 1
                    # cv2.imshow('f', frame)
                    if (count == 4):
                        count = 0
                        im = Image.fromarray(roi)
                        im.save('test.jpeg')
                        X = get_prediction('test.jpeg')
                        ret = ret + X + " "
                        print(X)
                        # draw the label into the frame
                         # Display the resulting frame
                        # cv2.imshow('Frame',clone)
                    print("amodh")
                    s = s + "A"
        __draw_label(clone, X, (25,334), (255,255,255))

        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        num_frames += 1
        cv2.imshow("Video Feed", clone)
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            return ret


def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    #im = Image.fromarray(thresholded)
    #im.save('test.jpeg')


    # get the contours in the thresholded image
    ( cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


if __name__ == "__main__":
    top()

#camera.release()
#cv2.destroyAllWindows()
