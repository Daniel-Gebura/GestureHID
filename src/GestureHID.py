################################################################
# GestureHID_Controller.py
#
# Description: Uses MediaPipe Hands to detect hand landmarks and classify
# gestures in real-time using trained PyTorch models. Sends HID inputs
# (mouse + keyboard) to a host system using HIDController on Jetson Nano.
#
# Author: Daniel Gebura (extended for live HID control)
################################################################

import cv2
import mediapipe as mp
import torch
import numpy as np
from Model import GestureClassifier
from hid_controller import HIDController, MOUSE_LEFT, MOUSE_RIGHT

# Constants and Configuration ---------------------------------------

# Labels used by each trained model
R_GESTURE_LABELS = ["closed_fist", "open_hand", "thumbs_up", "index_thumb", "pinky_thumb", "thumbs_down"]
L_GESTURE_LABELS = ["forward_point", "back_point", "left_point", "right_point", "open_hand", "index_thumb"]

CONFIDENCE_THRESHOLD = 0.3              # Minimum model confidence to accept prediction
MOUSE_TRACKING_LANDMARK = 0             # Wrist as reference for mouse movement
SENSITIVITY = 1000                      # Scales delta hand movement to pixel movement

# MediaPipe hand detection thresholds
MPH_DETECTION_CONFIDENCE = 0.9              # Confidence for detecting hands
MPH_TRACKING_CONFIDENCE = 0.6               # Confidence for tracking hand motion

# Webcam resolution configuration
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 30

# Gesture → action mapping
KEY_MAP = {
    "forward_point": "w",
    "back_point": "s",
    "left_point": "a",
    "right_point": "d",
    "index_thumb": "esc"
}
MOUSE_MAP = {
    "index_thumb": MOUSE_LEFT,
    "pinky_thumb": MOUSE_RIGHT
}

# PyTorch device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_grad_enabled(False)

# Utility Functions -------------------------------------------------

def normalize_landmarks(landmarks):
    """
    Normalize 3D hand landmarks relative to the wrist.

    Args:
        landmarks (list): Flat list of 63 values (21 points × 3 coordinates)

    Returns:
        np.ndarray: Normalized and flattened landmarks
    """
    landmarks = np.array(landmarks).reshape(21, 3)
    wrist = landmarks[0]
    landmarks -= wrist
    max_distance = np.max(np.linalg.norm(landmarks, axis=1)) + 1e-8
    return (landmarks / max_distance).flatten()

def initialize_mediapipe_hands():
    """
    Initialize MediaPipe Hands with provided confidence thresholds.

    Args:
        detection_confidence (float): Minimum detection confidence.
        tracking_confidence (float): Minimum tracking confidence.

    Returns:
        hands: Initialized MediaPipe Hands model
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MPH_DETECTION_CONFIDENCE,
        min_tracking_confidence=MPH_TRACKING_CONFIDENCE
    )
    return hands

def load_model(path, num_classes):
    """
    Load a trained PyTorch model from disk.

    Args:
        path (str): Path to .pth file
        num_classes (int): Output class size

    Returns:
        model: Loaded and ready PyTorch model
    """
    model = GestureClassifier(hidden_size=16, output_size=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.half()
    model.eval()
    return model

def get_webcam_capture():
    """
    Initialize and return a webcam capture object.

    Returns:
        cap: OpenCV VideoCapture object
    """
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open any webcam!")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    return cap

def calculate_mouse_delta(prev_coords, current_coords):
    """
    Compute pixel delta from hand movement.

    Args:
        prev_coords (tuple): Previous (x, y)
        current_coords (tuple): Current (x, y)

    Returns:
        tuple: (dx, dy) scaled by sensitivity
    """
    if not prev_coords:
        return 0, 0
    dx = (current_coords[0] - prev_coords[0]) * SENSITIVITY
    dy = (current_coords[1] - prev_coords[1]) * SENSITIVITY
    return int(dx), int(dy)

# Main Processing ---------------------------------------------------

def main():
    """
    Main loop for video capture, gesture recognition, and HID control.
    """
    hands = initialize_mediapipe_hands()
    right_model = load_model("../models/mini_right_multiclass_gesture_classifier.pth", len(R_GESTURE_LABELS))
    left_model = load_model("../models/mini_left_multiclass_gesture_classifier.pth", len(L_GESTURE_LABELS))
    cap = get_webcam_capture()                                              # Start webcam
    hid = HIDController()                                                   # Initialize HID writer

    prev_coords = None                                                      # Store previous mouse position for delta calculation

    try:
        while cap.isOpened():
            ret, frame = cap.read()                                         # Read frame
            if not ret:
                continue

            frame = cv2.flip(frame, 1)                                      # Mirror image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                   # Convert to RGB
            results = hands.process(rgb)                                    # Run hand detection

            right_gesture, left_gesture = "None", "None"                   # Reset state
            current_coords = None

            if results.multi_hand_landmarks:
                hands_data = []
                hand_sides = []
                raw_landmarks = []

                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0].label  # Get Left/Right
                    raw_landmarks.append(hand_landmarks)
                    hand_sides.append(handedness)

                    # Normalize for classifier
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    hands_data.append(normalize_landmarks(landmarks))

                hands_tensor = torch.tensor(hands_data, dtype=torch.float16).to(DEVICE)

                with torch.no_grad():
                    for i, hand_tensor in enumerate(hands_tensor):
                        if hand_sides[i] == "Right":
                            out = right_model(hand_tensor.unsqueeze(0))
                            probs = torch.softmax(out, dim=1)[0]
                            pred = torch.argmax(probs).item()
                            if probs[pred] > CONFIDENCE_THRESHOLD:
                                right_gesture = R_GESTURE_LABELS[pred]

                            # Track right-hand motion landmark
                            current_coords = (
                                raw_landmarks[i].landmark[MOUSE_TRACKING_LANDMARK].x,
                                raw_landmarks[i].landmark[MOUSE_TRACKING_LANDMARK].y
                            )

                        elif hand_sides[i] == "Left":
                            out = left_model(hand_tensor.unsqueeze(0))
                            probs = torch.softmax(out, dim=1)[0]
                            pred = torch.argmax(probs).item()
                            if probs[pred] > CONFIDENCE_THRESHOLD:
                                left_gesture = L_GESTURE_LABELS[pred]

            # ---------------- Perform HID Actions ----------------

            if current_coords and right_gesture != "closed_fist":
                dx, dy = calculate_mouse_delta(prev_coords, current_coords)
                hid.move_mouse(dx, dy)                                       # Move mouse
                prev_coords = current_coords
            else:
                prev_coords = None                                           # Stop tracking if hand is gone or closed

            if right_gesture in MOUSE_MAP:
                hid.press_mouse(MOUSE_MAP[right_gesture])                   # Click action
                hid.release_mouse()

            if left_gesture in KEY_MAP:
                hid.tap_key(KEY_MAP[left_gesture])                          # Send key

            torch.cuda.empty_cache()                                        # Clean GPU memory

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")

    finally:
        cap.release()
        hands.close()
        print("[INFO] Clean exit.")

if __name__ == "__main__":
    main()
