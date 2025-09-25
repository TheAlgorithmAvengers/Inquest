import cv2

import mediapipe as mp

import os

import time

cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,  
    max_num_hands=1,           
    min_detection_confidence=0.5  
)

char = ""

pm = True

while True:
    ret, frame = cam.read()
    if not ret:
        break

    if pm:
        print(" \n \n \n Camera is on. Press 'Esc' to exit. Press any albhabet to save an image in that folder. \n \n \n ")
        pm = False

    cv2.imshow("Camera To Collect Data", frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    results = hands.process(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        
        if key != 255:

            char = chr(key)
            char = char.upper()

            folder_path = os.path.join("resources", char)

            os.makedirs(folder_path, exist_ok=True)

            count = len(os.listdir(folder_path))

            file_path = os.path.join(folder_path, f"{count+1}.jpg")

            cv2.imwrite(file_path, frame)

            print(f"Image saved at: {file_path}")

            time.sleep(0.5)
 
cam.release()

cv2.destroyAllWindows()






