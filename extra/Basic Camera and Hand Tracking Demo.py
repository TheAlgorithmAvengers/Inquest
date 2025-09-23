# Use "source venv/bin/activate" in terminal to activate the virtual environment

# Use "deactivate" to exit the virtual environment

# Run "pip install -r requirements.txt" to install dependencies before running this script

# Run "python -m pip freeze > requirements.txt" to update dependencies after installing new packages

import cv2; # OpenCV for video capture and image processing

import mediapipe as mp; # MediaPipe for hand detection and tracking


# Set up video capture from the default camera (0)

# Change to 1 later for external camera

cap = cv2.VideoCapture(0);

# Initialize MediaPipe Hands solution - for hand detection and tracking
mp_hands = mp.solutions.hands;

hands = mp_hands.Hands()

# Initialize MediaPipe Drawing utility - for drawing landmarks on the image
mp_drawing = mp.solutions.drawing_utils;

while True:
    # Capture frame-by-frame
    ret, frame = cap.read();

    # If frame is not captured correctly, break the loop
    if not ret:
        break;

    # Draw the hand annotations on the image
    results = hands.process(frame);

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the camera feed
    cv2.imshow('Webcam', frame);
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;


