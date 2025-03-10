################################################################
# GestureDemo.py
#
# Description: Uses MediaPipe Hands to detect hand landmarks and classify
# gestures in real-time using trained PyTorch models for left and right hands.
#
# Author: Daniel Gebura
################################################################

import cv2
import mediapipe as mp
import torch
import numpy as np
from Model import GestureClassifier  # Import the trained PyTorch model class

# Optimize PyTorch for Jetson Nano
torch.backends.cudnn.benchmark = True  # Enable faster CUDA optimizations
torch.backends.cudnn.deterministic = False  # Avoid strict determinism for speed
torch.set_grad_enabled(False)  # Disable autograd to save computation

# Define gesture labels for classification
R_GESTURE_LABELS = ["closed_fist", "open_hand", "thumbs_up", "index_thumb", "pinky_thumb", "thumbs_down"]
L_GESTURE_LABELS = ["forward_point", "back_point", "left_point", "right_point", "open_hand", "index_thumb"]

# Confidence threshold for classification
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence required to classify a gesture

# Set GPU or CPU based on availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_landmarks(landmarks):
    """
    Vectorized normalization of hand landmarks relative to the wrist (landmark 0).
    Ensures location-invariant gesture classification.

    Args:
        landmarks (list): List of 63 values (21 landmarks * XYZ coordinates).

    Returns:
        numpy.ndarray: Normalized hand landmark coordinates.
    """
    landmarks = np.array(landmarks).reshape(21, 3)  # Convert to a 21x3 NumPy array
    wrist = landmarks[0]  # Extract wrist coordinates (reference point)
    landmarks -= wrist  # Translate all landmarks relative to the wrist
    max_distance = np.max(np.linalg.norm(landmarks, axis=1)) + 1e-8  # Normalize to the largest distance
    return (landmarks / max_distance).flatten()  # Flatten and normalize the coordinates

def initialize_mediapipe_hands():
    """
    Initialize MediaPipe Hands with tracking for stable detection.

    Returns:
        hands: MediaPipe Hands model instance.
        mp_drawing: Drawing utility for visualizing landmarks.
    """
    # Get the MediaPipe Hands model and define settings
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,  # Enable real-time video processing
        max_num_hands=2,  # Allow detection of up to 2 hands
        min_detection_confidence=0.9,  # Require high detection confidence
        min_tracking_confidence=0.6  # Improve hand tracking accuracy across frames
    )
    return hands, mp.solutions.drawing_utils  # Return the hands model and drawing utils

def load_model(model_path, num_classes):
    """
    Load a trained PyTorch model for hand gesture classification.

    Args:
        model_path (str): Path to the saved PyTorch model.
        num_classes (int): Number of output gesture classes.

    Returns:
        model: Loaded PyTorch model in evaluation mode.
    """
    model = GestureClassifier(hidden_size=16, output_size=num_classes).to(DEVICE)  # Move model to CPU or GPU
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))  # Load model weights
    model.half()  # Convert model to FP16 (Half Precision) to reduce memory usage
    model.eval()  # Set the model to evaluation mode
    return model

def get_video_capture():
    """
    Initialize the webcam for capturing video frames.

    Returns:
        cap: OpenCV video capture object.
    """
    cap = cv2.VideoCapture(1)  # Try USB webcam first
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Fallback to another camera index
    if not cap.isOpened():
        print("Error: Could not open any webcam!")  # Display an error if no camera is found
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduce resolution to 320x240 for efficiency
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Maintain a stable 30 FPS
    return cap

def main():
    """
    Main function to process webcam feed and classify hand gestures in real time.
    """
    hands, mp_drawing = initialize_mediapipe_hands()  # Initialize MediaPipe Hands

    # Load the trained gesture classification models
    right_model = load_model("../models/mini_right_multiclass_gesture_classifier.pth", len(R_GESTURE_LABELS))
    left_model = load_model("../models/mini_left_multiclass_gesture_classifier.pth", len(L_GESTURE_LABELS))

    cap = get_video_capture()  # Start video capture

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
            results = hands.process(rgb_frame)  # Process frame using MediaPipe Hands

            # 4. Reset gesture labels for each hand
            right_gesture, left_gesture = "None", "None"  # Default values when no hands are detected

            # 5. If hands are detected, process landmarks
            if results.multi_hand_landmarks:
                hands_data = []  # List to store landmark data
                hand_sides = []  # Track which hand is left/right

                # Iterate over detected hands, extract landmarks, and normalize them
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0].label  # Get hand type
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    hands_data.append(normalize_landmarks(landmarks))
                    hand_sides.append(handedness)

                # Classify the gesture for each hand
                if hands_data:
                    hands_tensor = torch.tensor(hands_data, dtype=torch.float16).to(DEVICE)  # Convert to FP16 tensor

                    with torch.no_grad():
                        for idx, hand_tensor in enumerate(hands_tensor):
                            if hand_sides[idx] == "Right":  # Process right hand separately
                                right_output = right_model(hand_tensor.unsqueeze(0))  # Get class probabilities
                                right_probs = torch.softmax(right_output, dim=1)[0]  # Format probabilities
                                predicted_class = torch.argmax(right_probs).item()  # Get most likely gesture
                                confidence = right_probs[predicted_class].item()  # Get confidence score of predicted gesture
                                if confidence >= CONFIDENCE_THRESHOLD:  # Apply confidence threshold
                                    right_gesture = R_GESTURE_LABELS[predicted_class]

                            elif hand_sides[idx] == "Left":  # Process left hand separately
                                left_output = left_model(hand_tensor.unsqueeze(0))  # Get class probabilities
                                left_probs = torch.softmax(left_output, dim=1)[0]  # Format probabilities
                                predicted_class = torch.argmax(left_probs).item()  # Get most likely gesture
                                confidence = left_probs[predicted_class].item()  # Get confidence score of predicted gesture
                                if confidence >= CONFIDENCE_THRESHOLD:  # Apply confidence threshold
                                    left_gesture = L_GESTURE_LABELS[predicted_class]

                # Efficiently Draw Landmarks on the Frame
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                # Display classification results
                cv2.putText(frame, f"Left: {left_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, f"Right: {right_gesture}", (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Free unused GPU memory after each frame to prevent memory overload
            torch.cuda.empty_cache()

            cv2.imshow("Hand Gesture Classification", frame)  # Display the processed frame

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
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
    main()  # Run the main function
