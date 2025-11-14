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
observation = np.append(observation, "FullStop")

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
    elif (i == 28):
        folder = "FullStop"
    else:
        folder = chr(i + 64)

    no_wrong = 0

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

                prediction = model.predict(hand_array, verbose=0)

                predicted_class = np.argmax(prediction, axis=1)[0]

                charec = observation[predicted_class]

                if charec != folder:
                    print(f" WRONG - Folder: {folder}, Image: {file}, Predicted Class: {predicted_class}")

                    wrong = np.append(wrong, f"Folder: {folder}, Image: {file}, Predicted: {charec}")

                    no_wrong += 1

    wrong_no = np.append(wrong_no, no_wrong)



for i in tqdm(range(0, 29), desc="Testing Alphabets"):
    predict_alphabet(i)

print("\n")

print("---------------------------------------------------")

print("Wrong Predictions Summary:")

for alp in wrong:
    print(alp)

print("---------------------------------------------------")

s = 0
for num in wrong_no:
    s+=1
print("Total Wrong Predictions: "+str(s))
accuracy = ((2900 - s)/2900)*100
print(f"Overall Accuracy: {accuracy}%")
    