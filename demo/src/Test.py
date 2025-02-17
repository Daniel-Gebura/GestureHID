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

# Define gesture labels for right and left hands separately
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


def initialize_mediapipe_hands():
    """
    Initialize MediaPipe Hands for hand detection.

    Returns:
        hands: MediaPipe Hands model instance.
        mp_drawing: Drawing utility for visualization.
        drawing_specs: Dictionary containing drawing specifications for both hands.
    """
    # Get the MediaPipe Hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Define the settings and create the model instance
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    # Define drawing specs for landmarks and connections
    drawing_specs = {
        "Right": mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  # Green for right hand
        "Left": mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3)   # Blue for left hand
    }

    return hands, mp_drawing, drawing_specs


def load_model(model_path, num_classes):
    """
    Load a trained PyTorch model for hand gesture classification.

    Args:
        model_path (str): Path to the saved model file.
        num_classes (int): Number of output classes.

    Returns:
        model (torch.nn.Module): Loaded PyTorch model in evaluation mode.
    """
    model = GestureClassifier(output_size=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()  # Set model to evaluation mode
    return model


def main():
    """Main function to initialize models, process webcam feed, and classify gestures."""

    # Specify the device being used (GPU or CPU)
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize MediaPipe Hands and drawing specs
    hands, mp_drawing, drawing_specs = initialize_mediapipe_hands()

    # Load trained PyTorch models for gesture classification
    right_model = load_model("../models/right_multiclass_gesture_classifier.pth", len(R_GESTURE_LABELS))
    left_model = load_model("../models/left_multiclass_gesture_classifier.pth", len(L_GESTURE_LABELS))

    # Start webcam capture
    cap = cv2.VideoCapture(1)

    # Perform processing loop in a try block to ensure graceful exit
    try:
        # Main processing loop
        while cap.isOpened():
            # 1. Capture frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # 2. Preprocess frame (flip horizontally and convert to RGB)
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 3. Process frame using MediaPipe Hands
            results = hands.process(rgb_frame)

            # 4. If hands are detected, process landmarks
            if results.multi_hand_landmarks:
                detected_hands = {"Right": False, "Left": False}  # Track which hands are detected

                # Iterate over detected hands
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Determine if Left or Right hand
                    handedness = results.multi_handedness[idx].classification[0].label

                    # Select corresponding color specs for drawing
                    landmark_spec = drawing_specs[handedness]
                    connection_spec = drawing_specs[handedness]

                    detected_hands[handedness] = True  # Mark detected hand as found

                    # Draw hand landmarks on the frame
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=landmark_spec,
                        connection_drawing_spec=connection_spec
                    )

                    # Extract and normalize landmark coordinates
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    landmarks = normalize_landmarks(landmarks)

                    # Convert landmarks to PyTorch tensor
                    landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                    # Gesture classification based on hand type
                    with torch.no_grad():
                        # Classify right hand gestures
                        if handedness == "Right":
                            output = right_model(landmarks_tensor)  # Get class probabilities from model
                            probabilities = torch.softmax(output, dim=1)[0]  # Format probabilities
                            predicted_class = torch.argmax(probabilities).item()  # Get most likely gesture
                            confidence = probabilities[predicted_class].item()  # Get confidence score of predicted gesture

                            # Apply confidence threshold
                            if confidence >= CONFIDENCE_THRESHOLD:
                                right_gesture = R_GESTURE_LABELS[predicted_class]
                            else:
                                right_gesture = "None"

                        # Classify left hand gestures
                        elif handedness == "Left":
                            output = left_model(landmarks_tensor)  # Get class probabilities
                            probabilities = torch.softmax(output, dim=1)[0]  # Format probabilities
                            predicted_class = torch.argmax(probabilities).item()  # Get most likely gesture
                            confidence = probabilities[predicted_class].item()  # Get confidence score of predicted gesture

                            # Apply confidence threshold
                            if confidence >= CONFIDENCE_THRESHOLD:
                                left_gesture = L_GESTURE_LABELS[predicted_class]
                            else:
                                left_gesture = "None"

                # Reset gesture classification for any undetected hands
                if not detected_hands["Right"]:
                    right_gesture = "None"
                if not detected_hands["Left"]:
                    left_gesture = "None"

            else:  # No hands were detected, set both gestures to none
                right_gesture, left_gesture = "None", "None"

            # 5. Display classification results on the frame
            frame_height, frame_width, _ = frame.shape
            cv2.putText(frame, f"Left: {left_gesture}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Right: {right_gesture}", (frame_width - 325, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Hand Gesture Classification", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Catch any exception from main processing loop
    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted! Closing resources...")

    # Ensure all resources are always released even on unexpected exits
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("[INFO] Resources released successfully. Exiting.")

if __name__ == "__main__":
    main()