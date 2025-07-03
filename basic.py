import cv2
import numpy as np
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands() # by default parameters
mpDraw = mp.solutions.drawing_utils

ptime = 0
ctime = 0

while True:
    res, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                height, width, channels = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(img, center=(cx, cy), radius=15, color=(255,0,255), thickness=-1)
                
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # if we not write mpHands.HAND_CONNECTIONS than we not see connection between landmarks
    
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    
    cv2.putText(img, text=str(int(fps)), org=(10, 70), color=(255, 0 , 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, thickness=3)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF==ord('x'):
        break
