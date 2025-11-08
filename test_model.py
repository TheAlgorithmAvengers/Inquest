from tensorflow.keras.models import load_model

import cv2

import mediapipe as mp

import numpy as np

import time

import os

model = load_model("trained_model.keras")

cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,  
    max_num_hands=2,           
    min_detection_confidence=0.5  
)

observation = np.array(["Space"])

for i in range (65,91):
    observation = np.append(observation, chr(i))

observation = np.append(observation, "Delete")

po = ""
no = ""

wd = ""

while True:

    observation = np.array(["Space"])

    for i in range (65,91):
        observation = np.append(observation, chr(i))

    observation = np.append(observation, "Delete")

    ret, frame = cam.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.flip(frame, 1)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    results = hands.process(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    """
    if results.multi_hand_landmarks :
                
        for hand_landmarks in results.multi_hand_landmarks:
                
            hand_array = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            ).flatten()

            hand_array = hand_array.reshape(1, -1)

            prediction = model.predict(hand_array)

            predicted_class = np.argmax(prediction, axis=1)[0]

            charec = observation[predicted_class]

            if po == charec:
                no = charec
                print(f"Predicted Class: {no}")
                po = ""

                if no == "Space":
                    wd += " "
                    print(f"Current Text: {wd}")
                elif no == "Delete":
                    wd = wd[:-2]
                else:
                    wd += no

            else:
                po = charec

            time.sleep(1)
    """

    if results.multi_hand_landmarks and results.multi_handedness:
                
        for hand_landmarks, handness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
            hand_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            if (handness.classification[0].label == "Right"):
                
                hand_array = hand_array.reshape(1, -1)

                prediction = model.predict(hand_array)

                predicted_class = np.argmax(prediction, axis=1)[0]

                charec = observation[predicted_class]

                if po == charec:
                    no = charec
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(f"Predicted Class: {no}")
                    po = ""

                    if no == "Space":
                        wd += " "
                        print(f"Current Text: {wd}")
                    elif no == "Delete":
                        wd = wd[:-2]
                    else:
                        wd += no

                else:
                    po = charec

                time.sleep(1)

cam.release()

cv2.destroyAllWindows()

