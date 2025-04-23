#!/usr/bin/env python3
################################################################
# GestureHID.py
#
# Description: Uses MediaPipe and trained PyTorch models to classify
# left and right hand gestures/position in real time. This information
# is used to control mouse and keyboard input via USB HID commands.
#
# Author: Daniel Gebura
################################################################

import cv2
import mediapipe as mp
import torch
import numpy as np
from Model import GestureClassifier
from GestureFSM import HIDToggleFSM, MouseFSM, KeyboardFSM

# Device Optimization -----------------------------------------------

torch.backends.cudnn.benchmark = True  # Enable faster CUDA optimizations
torch.backends.cudnn.deterministic = False  # Avoid strict determinism for speed
torch.set_grad_enabled(False)  # Disable autograd to save computation

# Configuration Constants -------------------------------------------

# Gesture Labels
R_GESTURE_LABELS = ["closed_fist", "open_hand", "thumbs_up", "index_thumb", "pinky_thumb", "thumbs_down"]
L_GESTURE_LABELS = ["forward_point", "back_point", "left_point", "right_point", "open_hand", "index_thumb"]

# Gesture Classification Model paths
R_MODEL_PATH = "../models/mini_right_multiclass_gesture_classifier.pth"
L_MODEL_PATH = "../models/mini_left_multiclass_gesture_classifier.pth"

# MediaPipe Hands Settings
MPH_DETECTION_CONFIDENCE = 0.85
MPH_TRACKING_CONFIDENCE = 0.6

# Gesture Classification Settings
GESTURE_CONFIDENCE = 0.3

# Mouse Control Settings
MOUSE_SENSITIVITY = 1500
MOUSE_TRACKING_LANDMARK = 0  # Wrist

# Camera Settings
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 30

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Utility Functions -------------------------------------------------

def initialize_mediapipe_hands():
    """
    Initialize MediaPipe Hands model with detection and tracking config.

    Returns:
        mp.solutions.Hands: MediaPipe Hands object
    """
    # Get the MediaPipe Hands model and define settings
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,  # Enable real-time video processing
        max_num_hands=2,  # Allow detection of up to 2 hands
        min_detection_confidence=MPH_DETECTION_CONFIDENCE,
        min_tracking_confidence=MPH_TRACKING_CONFIDENCE
    )

def load_gesture_model(path, num_classes):
    """
    Load a trained PyTorch model for hand gesture classification from file and set to eval mode.

    Args:
        path (str): Path to model file
        num_classes (int): Number of output classes for classification

    Returns:
        model (torch.nn.Module): Loaded gesture classifier model
    """
    model = GestureClassifier(hidden_size=16, output_size=num_classes).to(DEVICE)  # Move model to CPU or GPU
    model.load_state_dict(torch.load(path, map_location=DEVICE))  # Load model weights
    model.half()  # Convert model to FP16 (Half Precision) to reduce memory usage
    model.eval()  # Set the model to evaluation mode
    return model

def get_camera():
    """
    Attempt to open camera capture.

    Returns:
        cv2.VideoCapture: Video capture object
    """
    cap = cv2.VideoCapture(1)  # Try USB webcam first
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Fallback to another camera index
    if not cap.isOpened():
        print("Error: Could not open any webcam!")  # Display an error if no camera is found
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    return cap

def normalize_landmarks(landmarks):
    """
    Vectorized normalization of hand landmarks relative to the wrist (landmark 0).
    Ensures location-invariant gesture classification.

    Args:
        landmarks (list): List of 63 values (21 landmarks * XYZ coordinates).

    Returns:
        numpy.ndarray: Normalized hand landmark coordinates.
    """
    # Check for faulty landmarks
    if len(landmarks) != 63:
        return np.zeros(63, dtype=np.float32)

    landmarks = np.array(landmarks).reshape(21, 3)  # Convert to a 21x3 NumPy array
    wrist = landmarks[0]  # Extract wrist coordinates (reference point)
    landmarks -= wrist  # Translate all landmarks relative to the wrist
    max_distance = np.max(np.linalg.norm(landmarks, axis=1)) + 1e-8  # Normalize to the largest distance
    return (landmarks / max_distance).flatten()  # Flatten and normalize the coordinates

# Main Gesture HID Application --------------------------------------

def main():
    """
    Main processing loop that reads webcam frames, detects hand landmarks,
    classifies gestures, and delegates control to FSM-based HID abstraction.
    """
    # Load gesture models
    right_model = load_gesture_model(R_MODEL_PATH, len(R_GESTURE_LABELS))
    left_model = load_gesture_model(L_MODEL_PATH, len(L_GESTURE_LABELS))

    # Initialize HID FSMs
    toggle_fsm = HIDToggleFSM()  # State machine to toggle HID control on/off
    mouse_fsm = MouseFSM(sensitivity=MOUSE_SENSITIVITY)  # State machine to control mouse
    keyboard_fsm = KeyboardFSM()  # State machine to control keyboard press/release
    hid_enabled = False  # HID control is initially disabled
    prev_track_coords = None  # Track the previous frame's tracking coordinates 

    # Initialize webcam and mediapipe hands model
    cap = get_camera()
    hands = initialize_mediapipe_hands()

    # Perform processing loop in a try block to ensure graceful exit
    try:
        # Main processing loop
        while cap.isOpened():
            # 1. Read frame
            ret, frame = cap.read()
            if not ret:
                continue

            # 2. Preprocess frame (flip horizontally and convert to RGB)
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 3. Process frame using MediaPipe Hands
            results = hands.process(rgb)

            # 4. Reset gesture labels for each hand and tracking coordinates
            right_gesture = "None"
            left_gesture = "None"
            curr_track_coords = None

            # 5. If hands are detected, process landmarks
            if results.multi_hand_landmarks:
                # Iterate over detected hands
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get the handedness ("Right" or "Left") of this hand
                    handedness = results.multi_handedness[hand_idx].classification[0].label

                    # Extract hand landmarks, normalize them, and convert to tensor
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    normalized = normalize_landmarks(landmarks)
                    landmark_tensor = torch.tensor(normalized, dtype=torch.float16).unsqueeze(0).to(DEVICE)

                    # Classify this gesture based on the handedness
                    if handedness == "Right":
                        output = right_model(landmark_tensor)  # Forward pass through the model
                        probs = torch.softmax(output, dim=1)[0]  # Get classification probabilities
                        # Update gesture label if most likely class exceeds confidence threshold
                        if probs.max().item() >= GESTURE_CONFIDENCE:
                            right_gesture = R_GESTURE_LABELS[torch.argmax(probs).item()]
                        # Update tracking coordinates for mouse movement
                        curr_track_coords = (
                            hand_landmarks.landmark[MOUSE_TRACKING_LANDMARK].x,
                            hand_landmarks.landmark[MOUSE_TRACKING_LANDMARK].y
                        )

                    elif handedness == "Left":
                        output = left_model(landmark_tensor)  # Forward pass through the model
                        probs = torch.softmax(output, dim=1)[0]  # Get classification probabilities
                        # Update gesture label if most likely class exceeds confidence threshold
                        if probs.max().item() >= GESTURE_CONFIDENCE:
                            left_gesture = L_GESTURE_LABELS[torch.argmax(probs).item()]

            # 6. FSM HID Toggle Logic
            if toggle_fsm.update(right_gesture):
                # Gesture sequence completed, toggle the HID control state
                hid_enabled = not hid_enabled
                print(f"[INFO] HID {'ENABLED' if hid_enabled else 'DISABLED'}")

            # 7. FSM HID Action Logic
            if hid_enabled:
                # HID control is enabled, update the keyboard and mouse state machines with gestures
                mouse_fsm.update(right_gesture, prev_track_coords, curr_track_coords)
                keyboard_fsm.update(left_gesture)

            # 8. Save previous wrist position for next frame
            prev_track_coords = curr_track_coords

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Catch user interupt exception from main processing loop
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")

    # Ensure all resources are always released even on unexpected exits
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown complete")

if __name__ == "__main__":
    main()