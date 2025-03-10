################################################################
# SimHID.py
#
# Description: Uses MediaPipe Hands to detect hand landmarks and classifies
# the gesture in real-time using trained PyTorch models for both left and right hands.
# Additionally, simulates a mouse pointer (controlled by right-hand gestures) and 
# displays key presses (controlled by left-hand gestures).
#
# Author: Daniel Gebura, djg2170
################################################################

import cv2
import mediapipe as mp
import torch
import numpy as np
from Model import GestureClassifier

# ---------------- Constants and Mappings ----------------

# Define gesture labels for right and left hands separately
R_GESTURE_LABELS = ["closed_fist", "open_hand", "thumbs_up", "index_thumb", "pinky_thumb", "thumbs_down"]
L_GESTURE_LABELS = ["forward_point", "back_point", "left_point", "right_point", "open_hand", "index_thumb"]

# Specify the paths to the trained PyTorch models
R_MODEL_PATH = "../models/mini_right_multiclass_gesture_classifier.pth"
L_MODEL_PATH = "../models/mini_left_multiclass_gesture_classifier.pth"

# Model confidence thresholds
MIN_DETECTION_CONFIDENCE = 0.5   # Confidence threshold for hand detection
MIN_RECOGNITION_CONFIDENCE = 0.3  # Confidence threshold for gesture recognition

# Constant for selecting the landmark used for mouse tracking (0 = wrist)
MOUSE_TRACKING_LANDMARK = 0

# Sensitivity factor to convert normalized landmark differences to pixel movements
SENSITIVITY = 1000

# Simulated window size for the mouse and key press display
SIM_WINDOW_WIDTH = 640
SIM_WINDOW_HEIGHT = 480

# Left-hand gesture mapping to key presses
left_key_map = {
    "forward_point": "W",
    "back_point": "S",
    "left_point": "A",
    "right_point": "D",
    "index_thumb": "ESC"
}

# Right-hand gesture mapping to mouse button actions
right_mouse_map = {
    "thumbs_up": "scroll_up",
    "index_thumb": "left_click",
    "pinky_thumb": "right_click",
    "thumbs_down": "scroll_down"
}

# ---------------- Utility Functions ----------------

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
        min_detection_confidence=MIN_DETECTION_CONFIDENCE
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
    model = GestureClassifier(hidden_size=32, output_size=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()  # Set model to evaluation mode
    return model

def calculate_mouse_movement(prev_coords, current_coords, sensitivity):
    """
    Calculate the change in mouse x and y coordinates based on the previous and current normalized coordinates.
    
    Args:
         prev_coords (tuple or None): (prev_x, prev_y) if available, otherwise None.
         current_coords (tuple): (current_x, current_y)
         sensitivity (float): Scaling factor to convert normalized differences to pixel movement.
         
    Returns:
         tuple: (delta_x, delta_y) computed as pixel differences.
    """
    if prev_coords is None:
        return 0, 0
    prev_x, prev_y = prev_coords
    current_x, current_y = current_coords
    delta_x = (current_x - prev_x) * sensitivity
    delta_y = (current_y - prev_y) * sensitivity
    return delta_x, delta_y

def update_mouse_position(current_coords, right_gesture, sim_mouse_dx, sim_mouse_dy, sim_mouse_x, sim_mouse_y, window_width, window_height):
    """
    Update the simulated mouse pointer based on current right-hand coordinates,
    potential mouse movement, and gesture.

    Args:
         current_coords (tuple or None): Current right-hand raw coordinates (x, y)
         right_gesture (str): The recognized gesture for the right hand.
         sim_mouse_dx (float): Calculated change in x.
         sim_mouse_dy (float): Calculated change in y.
         sim_mouse_x (int): Current simulated mouse x position.
         sim_mouse_y (int): Current simulated mouse y position.
         window_width (int): Width of simulated window.
         window_height (int): Height of simulated window.

    Returns:
         tuple: (sim_mouse_x, sim_mouse_y, prev_right_x, prev_right_y) after updating.
    """
    # Update the mouse location ONLY if the right hand is not a closed fist
    if current_coords is not None and right_gesture != "closed_fist":
        # Send mouse position update 
        sim_mouse_x += int(sim_mouse_dx)
        sim_mouse_y += int(sim_mouse_dy)
        # Clamp pointer position to remain within simulated window bounds
        sim_mouse_x = max(0, min(window_width, sim_mouse_x))
        sim_mouse_y = max(0, min(window_height, sim_mouse_y))
        # Save the current coordinates for the next frame
        prev_right_x, prev_right_y = current_coords
    else:  # Hand was lost or closed_fist, so do not update the mouse position and do not track
        prev_right_x, prev_right_y = None, None
    return sim_mouse_x, sim_mouse_y, prev_right_x, prev_right_y

def show_mouse_action(mouse_action):
    """
    Determine the pointer color and shape based on the mouse action.

    Args:
        mouse_action (str): The current mouse action.

    Returns:
        tuple: (pointer_color, pointer_shape) for the simulated mouse pointer.
    """
    match mouse_action:
        case "left_click":
            pointer_color = (0, 0, 255)  # Red for left click
            pointer_shape = "circle"
        case "right_click":
            pointer_color = (255, 0, 0)  # Blue for right click
            pointer_shape = "circle"
        case "scroll_up":
            pointer_color = (255, 255, 255)  # Default: white
            pointer_shape = "arrow_up"
        case "scroll_down":
            pointer_color = (255, 255, 255)  # Default: white
            pointer_shape = "arrow_down"
        case _:
            pointer_color = (255, 255, 255)  # Default: white
            pointer_shape = "circle"         # Default shape

    return pointer_color, pointer_shape

def display_simulated_window(sim_mouse_x, sim_mouse_y, pointer_shape, pointer_color, key_press, mouse_action):
    """
    Display the simulated mouse pointer and key press on a blank window.

    Args:
        sim_mouse_x (int): Simulated mouse x position.
        sim_mouse_y (int): Simulated mouse y position.
        pointer_shape (str): Shape of the pointer ("circle", "arrow_up", "arrow_down").
        pointer_color (tuple): Color of the pointer (B, G, R).
        key_press (str): Key press to display.
        mouse_action (str): Mouse action to display.
    """
    # Create a blank simulated window image
    sim_window = np.zeros((SIM_WINDOW_HEIGHT, SIM_WINDOW_WIDTH, 3), dtype=np.uint8)

    # Draw the simulated mouse pointer
    if pointer_shape == "circle":
        cv2.circle(sim_window, (sim_mouse_x, sim_mouse_y), 10, pointer_color, -1)
    elif pointer_shape == "arrow_up":
        # Draw an upward pointing arrow using arrowedLine
        cv2.arrowedLine(sim_window, (sim_mouse_x, sim_mouse_y + 15),
                        (sim_mouse_x, sim_mouse_y - 15), pointer_color, 3, tipLength=0.5)
    elif pointer_shape == "arrow_down":
        # Draw a downward pointing arrow
        cv2.arrowedLine(sim_window, (sim_mouse_x, sim_mouse_y - 15),
                        (sim_mouse_x, sim_mouse_y + 15), pointer_color, 3, tipLength=0.5)

    # Display the key press text on the simulated window
    if key_press is not None:
        cv2.putText(sim_window, f"Key: {key_press}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Optionally, display the current mouse action (for debugging/feedback)
    if mouse_action is not None:
        cv2.putText(sim_window, f"Mouse: {mouse_action}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Simulated Mouse and Key Presses", sim_window)

# ---------------- Main Function ----------------

def main():
    """Main function to initialize models, process webcam feed, classify gestures,
       and update the simulated mouse/key press window."""
    
    # Specify the device being used (GPU or CPU)
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize MediaPipe Hands and drawing specs
    hands, mp_drawing, drawing_specs = initialize_mediapipe_hands()

    # Load trained PyTorch models for gesture classification
    right_model = load_model(R_MODEL_PATH, len(R_GESTURE_LABELS))
    left_model = load_model(L_MODEL_PATH, len(L_GESTURE_LABELS))

    # Initialize variables for mouse tracking
    sim_mouse_x = SIM_WINDOW_WIDTH // 2  # Start in the middle of the window
    sim_mouse_y = SIM_WINDOW_HEIGHT // 2  # Start in the middle of the window
    prev_right_x, prev_right_y = None, None  # Save, the previous frame hand coordinates
    current_right_coords = None  # Track the current right-hand raw x,y coordinates
    sim_mouse_dx, sim_mouse_dy = 0, 0  # Variables to store next mouse movement

    # Start webcam capture
    cap = cv2.VideoCapture(1)

    # Main processing loop inside try block to ensure graceful exit
    try:
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

            # 4. Reset gesture variables for this frame
            right_gesture = "None"
            left_gesture = "None"
            current_right_coords = None  # Reset current right-hand raw coordinate each frame
            sim_mouse_dx, sim_mouse_dy = 0, 0  # Reset potential mouse movement each frame

            # 5. If hands are detected, process landmarks
            if results.multi_hand_landmarks:
                detected_hands = {"Right": False, "Left": False}  # Track which hands are in frame

                # Iterate over detected hands (Max 2 guaranteed from detection model)
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Determine handedness ("Right" or "Left")
                    handedness = results.multi_handedness[idx].classification[0].label
                    detected_hands[handedness] = True  # Mark detected hand as found

                    # Select color specs and draw hand landmarks on frame
                    landmark_spec = drawing_specs[handedness]   # Controls color and thickness
                    connection_spec = drawing_specs[handedness]
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=landmark_spec,
                        connection_drawing_spec=connection_spec
                    )

                    # Use the main right-hand coordinate for mouse tracking
                    if handedness == "Right":
                        # Get raw normalized coordinates for the designated landmark (before any normalization)
                        current_right_coords = (
                            hand_landmarks.landmark[MOUSE_TRACKING_LANDMARK].x,
                            hand_landmarks.landmark[MOUSE_TRACKING_LANDMARK].y
                        )
                        # Calculate corresponding mouse movement
                        prev_coords = (prev_right_x, prev_right_y) if prev_right_x is not None and prev_right_y is not None else None
                        sim_mouse_dx, sim_mouse_dy = calculate_mouse_movement(prev_coords, current_right_coords, SENSITIVITY)

                    # Extract and normalize landmark coordinates for classification
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    normalized_landmarks = normalize_landmarks(landmarks)
                    landmarks_tensor = torch.tensor(normalized_landmarks, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # Convert to PyTorch tensor

                    # Classify the gesture
                    with torch.no_grad():
                        # Classify right hand gestures
                        if handedness == "Right":
                            output = right_model(landmarks_tensor)  # Get class probabilities from model
                            probabilities = torch.softmax(output, dim=1)[0]  # Format probabilities
                            predicted_class = torch.argmax(probabilities).item()  # Get most likely gesture
                            confidence = probabilities[predicted_class].item()  # Get confidence score

                            # Apply confidence threshold
                            if confidence >= MIN_RECOGNITION_CONFIDENCE:
                                right_gesture = R_GESTURE_LABELS[predicted_class]
                            else:
                                right_gesture = "None"

                        # Classify left hand gestures
                        elif handedness == "Left":
                            output = left_model(landmarks_tensor)  # Get class probabilities from model
                            probabilities = torch.softmax(output, dim=1)[0]  # Format probabilities
                            predicted_class = torch.argmax(probabilities).item()  # Get most likely gesture
                            confidence = probabilities[predicted_class].item()  # Get confidence score

                            # Apply confidence threshold
                            if confidence >= MIN_RECOGNITION_CONFIDENCE:
                                left_gesture = L_GESTURE_LABELS[predicted_class]
                            else:
                                left_gesture = "None"

                # Reset gesture classification for any undetected hands
                if not detected_hands["Right"]:
                    right_gesture = "None"
                    current_right_coords = None  # No right-hand data available
                if not detected_hands["Left"]:
                    left_gesture = "None"

            else:  # No hands were detected
                # Reset both gestures and right-hand coordinates
                right_gesture, left_gesture = "None", "None"
                current_right_coords = None

            # 6. Display classification results on the main frame
            frame_height, frame_width, _ = frame.shape
            cv2.putText(frame, f"Left: {left_gesture}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Right: {right_gesture}", (frame_width - 325, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Hand Gesture Classification", frame)

            # ---------------- Update Mouse and Key Press ----------------

            # Update simulated mouse pointer position.
            sim_mouse_x, sim_mouse_y, prev_right_x, prev_right_y = update_mouse_position(
                current_right_coords, right_gesture, sim_mouse_dx, sim_mouse_dy,
                sim_mouse_x, sim_mouse_y, SIM_WINDOW_WIDTH, SIM_WINDOW_HEIGHT
            )

            # Determine the mouse button action from the right-hand gesture
            mouse_action = right_mouse_map.get(right_gesture, None)
            pointer_color, pointer_shape = show_mouse_action(mouse_action)

            # Determine the key press from the left-hand gesture mapping
            key_press = left_key_map.get(left_gesture, None)

            # Display the mouse and keypress on a separate window
            display_simulated_window(sim_mouse_x, sim_mouse_y, pointer_shape, pointer_color, key_press, mouse_action)
            
            # -----------------------------------------------------------------------------

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted! Closing resources...")

    finally:
        # Release all resources
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("[INFO] Resources released successfully. Exiting.")

if __name__ == "__main__":
    main()