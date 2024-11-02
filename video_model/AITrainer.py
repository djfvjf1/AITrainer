import cv2
import numpy as np
import time
import pose_module1 as pm

cap = cv2.VideoCapture("dataset/chel_prisedaet.mp4")
#cap = cv2.VideoCapture(0)

detector = pm.poseDetector()
count = 0
direction = 0
# previous time - надо для расчета fps наху
pTime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280,720))

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    #print(lmList)
    if len(lmList) != 0:
        angle = detector.findAngle(img, 24, 26, 28)
        #angle1 = detector.findAngle(img, 23, 25, 27)

        # Переводим углы в проценты, где минимальный угол в присяде равен 0%, а максимальный - 100%
        per = np.interp(angle, (68,171), (0,100))
        #print(angle, per)

        bar = np.interp(angle, (68,171), (650, 100))

        # Делаем наш счетчик 
        if per == 0:
            if direction == 0:
                count += 0.5
                direction = 1
        if per == 100:
            if direction == 1:
                count += 0.5
                direction = 0
        print(count)

        cv2.rectangle(img, (1100,100), (1175, 650), (34,255,0), 3)
        cv2.rectangle(img, (1100,int(bar)), (1175, 650), (34,255,0), cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,0), 4)


        cv2.rectangle(img, (0,450), (250, 720), (34,255,0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255,0,0), 25)



        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)


        if count > 20:
            break

    cv2.imshow("Image", img)
    cv2.waitKey(1)