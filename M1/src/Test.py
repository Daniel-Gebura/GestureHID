################################################################
# Test.py
#
# Description: Uses MediaPipe Hands to detect hand landmarks and classifies
# the gesture in real-time using a trained PyTorch model.
#
# Author: Daniel Gebura, djg2170
################################################################
import cv2
import mediapipe as mp
import torch
import numpy as np
from Model import GestureClassifier

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Load the trained PyTorch model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureClassifier().to(DEVICE)
model.load_state_dict(torch.load("../models/binary_gesture_classifier.pth"))
model.eval()  # Set model to evaluation mode

# Start the webcam
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for selfie-view
    frame = cv2.flip(frame, 1)
    
    # Convert the frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Default prediction
    gesture = "None"

    # If hands are detected, process landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Flatten into a 63-dimensional vector

            # Convert to PyTorch tensor
            landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Make prediction
            with torch.no_grad():
                output = model(landmarks_tensor)
                prediction = (output > 0.5).float().item()  # Binary classification

            # Map prediction to gesture label
            gesture = "Closed Fist" if prediction == 1 else "None"

    # Display classification result
    cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Hand Gesture Classification", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
