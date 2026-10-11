
import cv2
import mediapipe as mp
import autopy
import time
import math
import numpy as np


class AirMouse:
    def __init__(
        self,
        cameraWidth=640,
        cameraHeight=480,
        frameReduction=100,
        smoothening=5,
        maxHands=1
    ):
        self.cameraWidth = cameraWidth
        self.cameraHeight = cameraHeight
        self.frameReduction = frameReduction
        self.smoothening = max(1, smoothening)

        self.screenWidth, self.screenHeight = autopy.screen.size()

        self.prevX = 0
        self.prevY = 0
        self.lastClicking = False

        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils

        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=maxHands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if draw and self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    img,
                    handLms,
                    self.mpHands.HAND_CONNECTIONS
                )

        return img

    def findPosition(self, img, handNo=0):
        lmList = []

        if (
            not hasattr(self, "results")
            or not self.results.multi_hand_landmarks
            or handNo >= len(self.results.multi_hand_landmarks)
        ):
            return lmList

        hand = self.results.multi_hand_landmarks[handNo]
        h, w, _ = img.shape

        for idx, lm in enumerate(hand.landmark):
            lmList.append([
                idx,
                int(lm.x * w),
                int(lm.y * h)
            ])

        return lmList

    @staticmethod
    def findDistance(p1, p2):
        return math.hypot(
            p2[0] - p1[0],
            p2[1] - p1[1]
        )

    def moveMouse(self, x, y):
        frameR = self.frameReduction

        x = np.clip(
            x,
            frameR,
            self.cameraWidth - frameR
        )
        y = np.clip(
            y,
            frameR,
            self.cameraHeight - frameR
        )

        targetX = np.interp(
            x,
            (frameR, self.cameraWidth - frameR),
            (0, self.screenWidth - 1)
        )
        targetY = np.interp(
            y,
            (frameR, self.cameraHeight - frameR),
            (0, self.screenHeight - 1)
        )

        currentX = self.prevX + (
            targetX - self.prevX
        ) / self.smoothening

        currentY = self.prevY + (
            targetY - self.prevY
        ) / self.smoothening

        currentX = int(np.clip(currentX, 0, self.screenWidth - 1))
        currentY = int(np.clip(currentY, 0, self.screenHeight - 1))

        autopy.mouse.move(currentX, currentY)

        self.prevX = currentX
        self.prevY = currentY

    def processFrame(self, img):
        img = self.findHands(img)
        lmList = self.findPosition(img)

        frameR = self.frameReduction

        cv2.rectangle(
            img,
            (frameR, frameR),
            (
                self.cameraWidth - frameR,
                self.cameraHeight - frameR
            ),
            (255, 0, 0),
            2
        )

        if len(lmList) < 13:
            self.lastClicking = False
            return img

        # Index fingertip and middle fingertip
        indexTip = (lmList[8][1], lmList[8][2])
        middleTip = (lmList[12][1], lmList[12][2])

        # Finger-up states based on landmark positions
        indexUp = lmList[8][2] < lmList[6][2]
        middleUp = lmList[12][2] < lmList[10][2]

        if indexUp and not middleUp:
            # Moving mode
            self.moveMouse(indexTip[0], indexTip[1])

            cv2.circle(
                img,
                indexTip,
                12,
                (255, 0, 255),
                cv2.FILLED
            )

            cv2.putText(
                img,
                "Moving",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            self.lastClicking = False

        elif indexUp and middleUp:
            # Clicking mode
            distance = self.findDistance(indexTip, middleTip)
            clicking = distance < 40

            cv2.line(
                img,
                indexTip,
                middleTip,
                (255, 0, 255),
                3
            )

            if clicking and not self.lastClicking:
                autopy.mouse.click()

            if clicking:
                cv2.putText(
                    img,
                    "Click",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            self.lastClicking = clicking

        else:
            self.lastClicking = False

        return img

    def close(self):
        self.hands.close()


def main():
    cap = cv2.VideoCapture(0)
    detector = None

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pTime = 0

    try:
        detector = AirMouse()

        while True:
            success, img = cap.read()

            if not success:
                break

            img = cv2.flip(img, 1)
            img = detector.processFrame(img)

            cTime = time.time()
            fps = 1 / (cTime - pTime) if pTime else 0
            pTime = cTime

            cv2.putText(
                img,
                f"FPS: {int(fps)}",
                (20, 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 0),
                2
            )

            cv2.imshow("Gesture Air Mouse", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if detector is not None:
            detector.close()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
