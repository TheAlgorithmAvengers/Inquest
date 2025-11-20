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
  

header = np.array([])

for i in range (21):
    
    header = np.append(header, f"x{i}")

    header = np.append(header, f"y{i}")

    header = np.append(header, f"z{i}")


del i

header = np.append(header, "label")

csv_path = os.path.join("resources", "resources1.csv")

csv_path1 = os.path.join("resources", "resources.csv")

hd = pd.DataFrame(columns=header)

hd.to_csv(csv_path1, index=False, mode='w', header=True)

df = pd.read_csv(csv_path)

df = df[(df.iloc[:, -1] != 1.0) & (df.iloc[:, -1] != 24.0)]

df.to_csv(csv_path1, index=False, mode='a', header=True)

extract_coordinates(1+64)

extract_coordinates(23+64)

coordinates = np.delete(coordinates, 0, 0)

cord = pd.DataFrame(coordinates, columns=header)

cord.to_csv(csv_path, index=False, mode='a', header=False)

del coordinates, header, csv_path, hd, cord
