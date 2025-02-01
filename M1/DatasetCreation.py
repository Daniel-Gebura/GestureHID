################################################################
# DatasetCreation.py
#
# Description: Tool to create a dataset of hand landmarks and the labeled gesture. 
# This script will save the landmarks of the user's hand to a single CSV file.
#
# Author: Daniel Gebura, djg2170
################################################################
import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Directory to save the dataset
DATASET_DIR = "gesture_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)  # Create directory if it does not exist

# Check if dataset file exists; if not, create it with a header
DATASET_FILE = os.path.join(DATASET_DIR, "dataset.csv")
if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ["gesture_label"] + [f"x{i}, y{i}, z{i}" for i in range(1, 22)]
        csv_writer.writerow(header)


def collect_gesture_data(gesture_label, num_samples=500):
    """
    Collects and appends hand landmark data for a given gesture label.

    Args:
        gesture_label (str): The label of the gesture.
        num_samples (int): The number of samples to collect.

    Returns:
        None
    """
    # Open dataset file in append mode
    with open(DATASET_FILE, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Initialize webcam
        cap = cv2.VideoCapture(1)  # Open webcam
        print(f"Recording gesture: {gesture_label}")
        print("Press 's' to start recording. Press 'q' to finish recording and enter new gesture.")

        # Initialize variables for recording
        recording = False
        sample_count = 0

        # Main loop for collecting samples
        while cap.isOpened():
            # 1. Get a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # 2. Pre-process the frame (flip, convert to RGB)
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 3. Process the frame with MediaPipe Hands
            results = hands.process(rgb_frame)

            # 4. If hands are detected, overlay the landmarks and save data
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:  # For each hand detected
                    # Draw landmarks on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Save data if recording
                    if recording and sample_count < num_samples:
                        # Extract landmark coordinates
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])

                        # Write to CSV
                        csv_writer.writerow([gesture_label] + landmarks)
                        sample_count += 1
                        print(f"Sample {sample_count}/{num_samples} collected for {gesture_label}")

            # Display the frame with instructions
            cv2.putText(frame, f"Gesture: {gesture_label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Press 's' to start, 'q' to finish gesture entry.", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Gesture Data Collection", frame)

            # Key press handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not recording:
                recording = True
                sample_count = 0
                print("Recording started...")
            elif key == ord('q') or sample_count >= num_samples:
                print(f"Finished recording gesture: {gesture_label}.")
                break

        cap.release()
        cv2.destroyAllWindows()


# Main loop for continuous dataset collection
if __name__ == "__main__":
    print("Starting gesture dataset collection tool.")

    # Main loop for collecting gestures
    while True:
        # Prompt User for gesture name and number of samples
        gesture_name = input("\nEnter the gesture name (or type 'exit' to finish): ").strip().lower()

        # Exit if user types 'exit'
        if gesture_name == "exit":
            print("Dataset collection finished.")
            break
        
        # Collect gesture data
        num_samples = input("Enter the number of samples to capture (default 500): ").strip()
        num_samples = int(num_samples) if num_samples.isdigit() else 500
        collect_gesture_data(gesture_name, num_samples)