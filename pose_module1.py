import cv2
import mediapipe as mp
import time
import math

class poseDetector:
    def __init__(self, mode=False, model_complexity=1, enable_segmentation=False, 
                 smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode, 
            model_complexity=self.model_complexity, 
            enable_segmentation=self.enable_segmentation,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon, 
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 4, (255, 0, 0), cv2.FILLED)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw = True):

        # Получаем наши координаты
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Вычисляем угол
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        #print(angle)

        if angle < 0:
            angle += 360

        # Рисуем наши линии и круглишочки, чтобы визуально было понятно, че к чему
        if draw:
            cv2.line(img, (x1,y1), (x2,y2), (255,255,255), 3)
            cv2.line(img, (x3,y3), (x2,y2), (255,255,255), 3)
            cv2.circle(img, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 14, (255, 0, 0), 2)
            cv2.circle(img, (x2, y2), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 14, (255, 0, 0), 2)
            cv2.circle(img, (x3, y3), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 14, (255, 0, 0), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)


        return angle


def main():
    cap = cv2.VideoCapture('dataset/chel_prisedaet.mp4')
    pTime = 0  # Previous time
    scale_factor = 0.7  # Масштабирование кадра, можно изменить для лучшего вписывания

    detector = poseDetector()
    
    while True:
        success, img = cap.read()
        if not success:
            break  # Остановить, если видео закончилось

        # Изменение размера кадра
        height, width = img.shape[:2]
        img_resized = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))

        img_resized = detector.findPose(img_resized)
        lmList = detector.findPosition(img_resized, draw=False)

        if len(lmList) != 0:
            for idx in [25, 26]:  # Индексы элементов, которые нужно обработать
                print(lmList[idx])
                cv2.circle(img_resized, (lmList[idx][1], lmList[idx][2]), 15, (0, 0, 255), cv2.FILLED)
        
        # Расчет FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img_resized, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # Отображение результата
        cv2.imshow("Resized Image", img_resized)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
