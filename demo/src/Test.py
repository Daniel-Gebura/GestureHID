################################################################
# Test.py
#
# Description: Uses MediaPipe Hands to detect hand landmarks and classifies
# the gesture in real-time using a trained PyTorch model.
# Applies the same normalization used during training.
#
# Author: Daniel Gebura, djg2170
################################################################

import cv2
import mediapipe as mp
import torch
import numpy as np
from Model import GestureClassifier

# Define gesture labels (matching the dataset)
GESTURE_LABELS = ["closed_fist", "open_hand", "thumbs_up", "index_thumb", "pinky_thumb", "thumbs_down"]

# Confidence threshold for classification
CONFIDENCE_THRESHOLD = 0.3  # HYPERPARAMETER TO ADJUST

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Load the trained PyTorch model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureClassifier(output_size=len(GESTURE_LABELS)).to(DEVICE)
model.load_state_dict(torch.load("../models/right_multiclass_gesture_classifier.pth", map_location=DEVICE))
model.eval()  # Set model to evaluation mode

def normalize_landmarks(landmarks):
    """
    Normalizes hand landmarks relative to the wrist (landmark 0).
    Ensures location-invariant gesture classification.

    Args:
        landmarks (list): List of 63 values (21 landmarks * XYZ).

    Returns:
        numpy.ndarray: Normalized landmark coordinates.
    """
    # Convert to NumPy array
    landmarks = np.array(landmarks)

    # Extract wrist coordinates
    wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]

    # Translate all landmarks relative to wrist
    for i in range(21):
        landmarks[i * 3] -= wrist_x  # x
        landmarks[i * 3 + 1] -= wrist_y  # y
        landmarks[i * 3 + 2] -= wrist_z  # z

    # Compute hand scale (max distance from wrist)
    max_distance = max(np.linalg.norm(landmarks[i * 3: i * 3 + 3]) for i in range(21))

    # Scale all coordinates
    landmarks /= (max_distance + 1e-8)

    return landmarks

# Start the webcam
cap = cv2.VideoCapture(1)

# Main Loop
while cap.isOpened():
    # 1. Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # 2. Preprocess the frame (flip horizontally and convert to RGB)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 3. Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # 4. Set default prediction
    gesture = "None"

    # 5. If hands are detected, process landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:  # For each hand detected
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Flatten into a 63-dimensional vector

            # Normalize using the same approach as training
            landmarks = normalize_landmarks(landmarks)

            # Convert to PyTorch tensor
            landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Make prediction
            with torch.no_grad():
                output = model(landmarks_tensor)  # Get class probabilities
                probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]  # Convert to NumPy array
                predicted_class = np.argmax(probabilities)  # Get class with highest probability
                confidence = probabilities[predicted_class]  # Get confidence score

            # Apply confidence threshold
            if confidence >= CONFIDENCE_THRESHOLD:
                gesture = GESTURE_LABELS[predicted_class]  # Assign predicted gesture
            else:
                gesture = "None"  # Low confidence, default to "None"

    # 6. Display classification result
    cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Classification", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()