################################################################
# Test.py
#
# Description: Uses MediaPipe Hands to detect hand landmarks and classifies
# the gesture in real-time using trained PyTorch models for both left and right hands.
#
# Author: Daniel Gebura, djg2170
################################################################

import cv2
import mediapipe as mp
import torch
import numpy as np
from Model import GestureClassifier

# Define gesture labels for right/left hand separately
R_GESTURE_LABELS = ["closed_fist", "open_hand", "thumbs_up", "index_thumb", "pinky_thumb", "thumbs_down"]
L_GESTURE_LABELS = ["forward_point", "back_point", "left_point", "right_point", "open_hand", "index_thumb"]

# Confidence threshold for classification
CONFIDENCE_THRESHOLD = 0.3  # HYPERPARAMETER TO ADJUST

def normalize_landmarks(landmarks):
    """
    Vectorized normalization of hand landmarks relative to the wrist (landmark 0).
    Ensures location-invariant gesture classification.

    Args:
        landmarks (list): List of 63 values (21 landmarks * XYZ).

    Returns:
        numpy.ndarray: Normalized landmark coordinates.
    """
    landmarks = np.array(landmarks).reshape(21, 3)  # Convert to 21x3 array
    wrist = landmarks[0]  # Extract wrist coordinates
    landmarks -= wrist  # Translate all landmarks relative to wrist
    max_distance = np.max(np.linalg.norm(landmarks, axis=1)) + 1e-8  # Compute scaling factor
    return (landmarks / max_distance).flatten()  # Normalize and flatten

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)  # Detect up to 2 hands

# Define custom colors for the landmarks
right_hand_color = (0, 255, 0)  # Green
left_hand_color = (255, 0, 0)  # Blue

# Define drawing specs for landmarks and connections
right_hand_spec = mp_drawing.DrawingSpec(color=right_hand_color, thickness=2, circle_radius=3)
left_hand_spec = mp_drawing.DrawingSpec(color=left_hand_color, thickness=2, circle_radius=3)

# Load the trained PyTorch models for right and left hand
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

right_model = GestureClassifier(output_size=len(R_GESTURE_LABELS)).to(DEVICE)
right_model.load_state_dict(torch.load("../models/right_multiclass_gesture_classifier.pth", map_location=DEVICE))
right_model.eval()  # Set model to evaluation mode

left_model = GestureClassifier(output_size=len(L_GESTURE_LABELS)).to(DEVICE)
left_model.load_state_dict(torch.load("../models/left_multiclass_gesture_classifier.pth", map_location=DEVICE))
left_model.eval()  # Set model to evaluation mode

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

    # 4. If no hands are detected, reset both gestures and continue
    if not results.multi_hand_landmarks:
        right_gesture, left_gesture = "None", "None"

    # 5. Else, process the detected hands
    else:
        detected_hands = {"Right": False, "Left": False}  # Track detected hands

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):  # For each hand detected
            handedness = results.multi_handedness[idx].classification[0].label  # check "Left" or "Right" hand

            # Select the correct color specs
            if handedness == "Right":
                landmark_spec = right_hand_spec
                connection_spec = right_hand_spec
            else:
                landmark_spec = left_hand_spec
                connection_spec = left_hand_spec

            detected_hands[handedness] = True  # Mark the detected hand as found

            # Draw landmarks with matching colors
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec
            )

            # Extract & normalize landmark coordinates
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            landmarks = normalize_landmarks(landmarks)

            # Convert to PyTorch tensor
            landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Make prediction based on hand type
            with torch.no_grad():
                if handedness == "Right":
                    output = right_model(landmarks_tensor)  # Get class probabilities
                    probabilities = torch.softmax(output, dim=1)[0]  # Directly get tensor
                    predicted_class = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_class].item()

                    # Apply confidence threshold
                    if confidence >= CONFIDENCE_THRESHOLD:
                        right_gesture = R_GESTURE_LABELS[predicted_class]

                elif handedness == "Left":
                    output = left_model(landmarks_tensor)  # Get class probabilities
                    probabilities = torch.softmax(output, dim=1)[0]  # Directly get tensor
                    predicted_class = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_class].item()

                    # Apply confidence threshold
                    if confidence >= CONFIDENCE_THRESHOLD:
                        left_gesture = L_GESTURE_LABELS[predicted_class]

        # 6. If a hand was not detected, reset only that hand's gesture
        if not detected_hands["Right"]:
            right_gesture = "None"
        if not detected_hands["Left"]:
            left_gesture = "None"

    # 6. Display classification result for both hands
    frame_height, frame_width, _ = frame.shape
    cv2.putText(frame, f"Left: {left_gesture}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Right: {right_gesture}", (frame_width - 325, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Classification", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()