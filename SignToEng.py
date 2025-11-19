import os

from tensorflow.keras.models import load_model

import cv2

import mediapipe as mp

import numpy as np

import time

from collections import deque

import pyttsx3

from spellchecker import SpellChecker

from textblob import TextBlob

speech_engine = pyttsx3.init()

spell = SpellChecker(language='en')

model = load_model(os.path.join("model","trained_model.keras",))

# ---------------------------------
cam = cv2.VideoCapture(0) 
# ---------------------------------

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

observation = np.append(observation, "FullStop")

observation = np.append(observation, "Clear")

po = ""
no = ""

sentence = ""

hand = "Right"

prediction_buffer = deque(maxlen=30)

prediction_no = 0

frames = 0

start_time = time.time()

po = ""

def calculate_fps():
    global frames, start_time

    frames += 1

    elapsed_time = time.time() - start_time

    if elapsed_time >= 1.0:

        fps = frames / elapsed_time

        frames = 0

        start_time = time.time()

        return fps
        
def correct_spelling(sentence):

    corrected_text = " ".join(spell.correction(w) or w for w in sentence.split())

    corrected_text = TextBlob(corrected_text).correct()

    return str(corrected_text)


def predict(hand_array,sentence):

    global prediction_no, prediction_buffer, s1, po

    hand_array = hand_array.reshape(1, -1)

    prediction = model.predict(hand_array, verbose = 0)

    predicted_class = np.argmax(prediction, axis=1)[0]

    prediction_buffer.append(predicted_class)

    prediction_no += 1

    if prediction_no < 30:
        return sentence

    time.sleep(1)

    prediction_no = 0

    counts = np.bincount(prediction_buffer)

    prediction_buffer.clear()

    predicted_char = np.argmax(counts)

    charec = observation[predicted_char]

    
    print("Predicted Character: ", charec)
   
    if charec == "Space":
        charec = " " 

        if po == " ":
            charec = "FullStop"

    if charec == "FullStop":
        charec = "."

        sentence = correct_spelling(sentence)

        print("Corrected Sentence: ", sentence)

        displaySentence(sentence)


    elif charec == "Clear":
        return sentence

    if charec != "Delete":
        sentence = sentence + charec

    else :
        sentence = sentence[:-1]

    po = charec

    return sentence

def displaySentence(sentence):

    global speech_engine

    if sentence != "" and po == " ":
        speech_engine.say(sentence)
        speech_engine.runAndWait()
        sentence = ""


while True:

    ret, frame = cam.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.flip(frame, 1)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if key == ord('1'):
        hand = "Left"
    if key == ord('2'):
        hand = "Right"

    if key == ord('3'):
        sentence = ""

    if key == ord('4'):
        print("FPS:", fps)

    if key == ord('5'):
        
        sentence = ""

    results = hands.process(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    fps = calculate_fps()

    if results.multi_hand_landmarks and results.multi_handedness:
                
        for hand_landmarks, handness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
            hand_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            if (handness.classification[0].label == hand):

                sentence = predict(hand_array,sentence)
    
                
                

                

                

cam.release()

cv2.destroyAllWindows()

