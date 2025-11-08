from tensorflow.keras.models import load_model

import cv2

import mediapipe as mp

import os

import numpy as np

import random

from tqdm import tqdm

model = load_model("trained_model.keras")

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,  
    max_num_hands=2,           
    min_detection_confidence=0.5  
)

observation = np.array(["Space"])

for i in range (65,91):
    observation = np.append(observation, chr(i))

observation = np.append(observation, "Delete")

wrong = np.array([])
wrong_no = np.array([])

def predict_alphabet(i):
    global wrong

    global wrong_no

    folder = ""

    if(i == 0):
        folder = "Space"
    elif (i == 27):
        folder = "Delete"
    else:
        folder = chr(i + 64)

    no_wrong = 0

    print(f"Testing for alphabet: {folder}, Index: {i}")

    print("---------------------------------------------------")

    folder_path = os.path.join("resources", folder)

    images = [filename for filename in os.listdir(folder_path) if filename.lower().endswith((".png", ".jpg", ".jpeg"))]

    random_files = random.sample(images, 100)

    for file in random_files:
        
        image_path = os.path.join(folder_path, file)

        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image)

        if results.multi_hand_landmarks :
                
            for hand_landmarks in results.multi_hand_landmarks:
                
                hand_array = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                ).flatten()

                hand_array = hand_array.reshape(1, -1)

                prediction = model.predict(hand_array)

                predicted_class = np.argmax(prediction, axis=1)[0]

                charec = observation[predicted_class]

                if charec != folder:
                    print(f" WRONG - Folder: {folder}, Image: {file}, Predicted Class: {predicted_class}")

                    wrong = np.append(wrong, f"Folder: {folder}, Image: {file}, Predicted: {charec}")

                    no_wrong += 1

    wrong_no = np.append(wrong_no, folder + " " + str(no_wrong))

    print(f"Completed testing for alphabet: {folder}, Number of Wrong Predictions: {no_wrong}")

    print("---------------------------------------------------")


for i in tqdm(range(0, 28), desc="Testing Alphabets"):
    predict_alphabet(i)

print("\n")

print("---------------------------------------------------")

print("Wrong Predictions Summary:")

for alp in wrong:
    print(alp)

print("---------------------------------------------------")

for num in wrong_no:
    print(num)