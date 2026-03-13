import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils
pTime=0
cTime=0



while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=hands.process(imgRGB)
    #print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
             for id,lm in enumerate(handLms.landmark):
                 h,w,c=img.shape
                 cx,cy=int(lm.x*w),int(lm.y*h)
                 print(id,cx,cy)
                 if id ==4:
                    cv2.circle(img,(cx,cy),15,(240,0,240),cv2.FILLED)

             mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)


    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)),(10,40),cv2.FONT_HERSHEY_DUPLEX,1,(225,0,225),2)

    cv2.imshow("image",img)
    cv2.waitKey(1)

