import cv2
import time
import mediapipe
import numpy as nm
import os
import HandTraking as hd

bThickness = 10
eThickness = 20
folderPath = "Header"
newList = os.listdir(folderPath)
overLayList = []
for i in newList:
    image = cv2.imread(f'{folderPath}/{i}')
    overLayList.append(image)
print(len(overLayList))
header = overLayList[0]
drawColor = (0, 255, 255)

h = 720
w = 1280
cam = cv2.VideoCapture(0)
cam.set(3, w)
cam.set(4, h)


D = hd.HandDetector(detection_con=0.8)
xp, yp= 0, 0
imgCanvas = nm.zeros((720, 1280, 3), nm.uint8)

while True:
    success, img = cam.read()
    img = cv2.flip(img, 1)

    img = D.find_hands(img)
    landMark = D.find_position(img, draw=False)

    if len(landMark) != 0:
        # print(landMark)

        x1, y1 = landMark[8][1:]
        x2, y2 = landMark[12][1:]

        fingers = D.fingers_up()
        # print(fingers)
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection mode")
            if y1 < 125:
                if 200 < x1 < 400:
                    header = overLayList[0]
                    drawColor = (0, 255, 255)
                elif 500 < x1 < 700:
                    header = overLayList[1]
                    drawColor = (249, 131, 53)
                elif 760 < x1 < 900:
                    header = overLayList[2]
                    drawColor = (255, 255, 255)
                elif 1000 < x1 < 1160:
                    header = overLayList[3]
                    drawColor = (0, 0, 0)
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 +25), drawColor, cv2.FILLED)

        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 10, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp==0 and yp==0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eThickness)

            cv2.line(img, (xp, yp), (x1, y1), drawColor, bThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, bThickness)

            xp, yp= x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("img", img)
    cv2.imshow("Canvas", imgCanvas)

    cv2.waitKey(1)
