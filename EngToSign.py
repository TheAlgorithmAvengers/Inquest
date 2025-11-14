import speech_recognition as sr
import os
import cv2
import string

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Listening...")
    recognizer.adjust_for_ambient_noise(source, duration=1)
    audio_data = recognizer.listen(source)
    print("Audio captured. Recognizing...")
    try:
        text = recognizer.recognize_google(audio_data)
        print("You said:", text)
        detected_audio = text
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        detected_audio = ""
    except sr.RequestError:
        print("Request error. Check internet connection.")
        detected_audio = ""

RESOURCE_PATH = "Resources"

def display_character_image(char):
    if char == " ":
        folder_name = "Space"
    elif char == ".":
        folder_name = "FullStop"
    elif char.upper() in string.ascii_uppercase:
        folder_name = char.upper()
    else:
        return
    folder_path = os.path.join(RESOURCE_PATH, folder_name)
    if not os.path.exists(folder_path):
        return
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        return
    img_path = os.path.join(folder_path, image_files[0])
    img = cv2.imread(img_path)
    if img is None:
        return
    cv2.imshow(f"Character: {folder_name}", img)
    cv2.waitKey(700)
    cv2.destroyAllWindows()

input_text = detected_audio.lower()
print("Input text:", input_text)
for char in input_text:
    display_character_image(char)