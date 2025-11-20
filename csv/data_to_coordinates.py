import numpy as np  

import pandas as pd

import cv2

import os

import mediapipe as mp

from tqdm import tqdm

import random

coordinates = np.array([[0]*64])

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,  
    max_num_hands=1,           
    min_detection_confidence=0.5  
)

def extract_coordinates(i):

    global coordinates

    folder = chr(i)

    i_num = i - 64

    if(i == 0):
        folder = "Space"
        i_num = 0

    if (i == 27):
        folder = "Delete"
        i_num = 27

    if (i == 28):
        folder = "FullStop"
        i_num = 28

    if (i == 29):
        folder = "Empty"
        i_num = 29

    folder_path = os.path.join("resources", folder)

    print("\n")

    for filename in tqdm(os.listdir(folder_path), desc=f"Processing {folder} folder"):

        if filename.lower().endswith((".png", ".jpg", ".jpeg")):

            image_path = os.path.join(folder_path, filename)

            image = cv2.imread(image_path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if random.random() < 0.5:
                image = cv2.flip(image, 1)

            results = hands.process(image)

            if results.multi_hand_landmarks:
                
                for hand_landmarks in results.multi_hand_landmarks:
                
                    hand_array = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    ).flatten()

                    hand_array = np.append(hand_array, i_num)

                    coordinates = np.append(coordinates, [hand_array], axis=0)
                     
            del image, image_path, results

    del folder, i_num, folder_path, filename
  

for i in tqdm (range(30), desc="Processing Folders"):
    if i == 0 or i == 27 or i == 28 or i == 29:
        extract_coordinates(i)
    else:
        extract_coordinates(i+64)


np.savetxt(os.path.join("resources", "data.txt"), coordinates, delimiter=",", fmt="%f")

np.savetxt(os.path.join("resources", "data.csv"), coordinates, delimiter=",", fmt="%f")

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
