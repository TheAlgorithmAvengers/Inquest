import numpy as np  

import pandas as pd

import cv2

import os

import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,  
    max_num_hands=1,           
    min_detection_confidence=0.5  
)

coordinates = np.array([[0]*64])

for i in range(65, 91):

    folder = chr(i)

    folder_path = os.path.join("resources", folder)

    for filename in os.listdir(folder_path):

        if filename.lower().endswith((".png", ".jpg", ".jpeg")):

            image_path = os.path.join(folder_path, filename)

            image = cv2.imread(image_path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = hands.process(image)

            if results.multi_hand_landmarks:
                
                for hand_landmarks in results.multi_hand_landmarks:
                
                    hand_array = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    ).flatten()

                    hand_array = np.append(hand_array, (i - 64))

                    coordinates = np.append(coordinates, [hand_array], axis=0)

                    del hand_array
                     
            del image, image_path, results


del hands, mp_hands,  folder, folder_path, cv2, mp

header = np.array([])

for i in range (21):
    
    header = np.append(header, f"x{i}")

    header = np.append(header, f"y{i}")

    header = np.append(header, f"z{i}")

del i

header = np.append(header, "label")

csv_path = os.path.join("resources", "resources.csv")

hd = pd.DataFrame(columns=header)

hd.to_csv(csv_path, index=False, mode='w', header=True)

coordinates = np.delete(coordinates, 0, 0)

cord = pd.DataFrame(coordinates, columns=header)

cord.to_csv(csv_path, index=False, mode='w', header=True)

del coordinates, header, csv_path, hd, cord
