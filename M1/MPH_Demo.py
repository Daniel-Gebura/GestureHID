################################################################
# MPH_Demo.py
#
# Description: MediaPipe Hands Demo. Simply uses MediaPipe Hands to overlay
# landmarks on the user's hand in real-time.
#
# Author: Daniel Gebura, djg2170
################################################################
import cv2
import mediapipe as mp

# Bring in the MediaPipe Hands modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the MediaPipe Hands model
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize the webcam
cap = cv2.VideoCapture(1)

# Loop to process the webcam frames
while cap.isOpened():
    # 1. Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # 2. Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # 3. Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 4. Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # 5. If hands are detected, overlay the landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 6. Display the frame
    cv2.imshow('MediaPipe Hands Demo', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release created objects and destroy all openCV windows
cap.release()
hands.close()
cv2.destroyAllWindows()

# End of MPH_Demo.py
################################################################
