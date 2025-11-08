
import cv2  
import mediapipe as mp  
import numpy as np  


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)  # 0 = Default webcam


with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        
        frame = cv2.flip(frame, 1)

        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        results = hands.process(rgb_frame)

        
        landmark_list = []

        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                
                for lm in hand_landmarks.landmark:
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([cx, cy])

        
        if landmark_list:
            landmarks_array = np.array(landmark_list)
            print("Detected Hand Landmarks:\n", landmarks_array)

        
        cv2.imshow("Sign Language Detection", frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()


