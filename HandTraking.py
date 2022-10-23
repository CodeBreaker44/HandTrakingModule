import cv2
import mediapipe as mp
import time

"""import gtts
import playsound
import self as self"""

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.results = None
        self.mode = mode
        self.maxHands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detection_con, self.track_con)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for i in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, i, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]
            for ip, lm in enumerate(myHand.landmark):
                # print(x, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(ip, cx, cy)
                self.lmList.append([ip, cx, cy])
                if draw:
                    # if ip==0:
                    cv2.circle(img, (cx, cy), 5, (255, 204, 0), cv2.FILLED)

        return self.lmList

    def fingers_up(self):
        fingers = []

        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for i in range(1, 5):
            if self.lmList[self.tipIds[i]][2] < self.lmList[self.tipIds[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


def main():
    pTime = 0
    cTime = 0
    cam = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cam.read()
        detector.find_hands(img)
        # img = cv2.flip(img, 1)
        lmList = detector.find_position(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fbs = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fbs)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 204, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
