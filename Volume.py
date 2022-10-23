import cv2
import time
import numpy as np
import HandTraking as ht
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

width, height = 480, 550
cam = cv2.VideoCapture(0)
cam.set(3, width)
cam.set(4, height)
pTime = 0

D = ht.HandDetector(detection_con=0.7)

devices = AudioUtilities.GetSpeakers()

face = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(face, POINTER(IAudioEndpointVolume))
vRange = volume.GetVolumeRange()
minVol = vRange[0]
maxVol = vRange[1]

while True:
    success, img = cam.read()
    img = D.find_hands(img)
    # img = cv2.flip(img, 1)
    lmList = D.find_position(img, draw=False)
    if len(lmList) != 0:
        print(lmList[4], lmList[8])
        z, m = lmList[4][1], lmList[4][2]
        x, w = lmList[8][1], lmList[8][2]
        cx, cy = (z + x) // 2, (m + w) // 2

        cv2.circle(img, (z, m), 5, (255, 204, 0), cv2.FILLED)
        cv2.circle(img, (x, w), 5, (255, 204, 0), cv2.FILLED)
        cv2.line(img, (z, m), (x, w), (255, 204, 0), 2)
        cv2.circle(img, (cx, cy), 5, (255, 204, 0), cv2.FILLED)

        length = math.hypot(z - x, m - w)
        print(length)

        vol = np.interp(length, [50, 200], [minVol, maxVol])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)
        if length < 50:
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f' {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 204, 255), 3)

    cv2.imshow("img", img)
    cv2.waitKey(1)
