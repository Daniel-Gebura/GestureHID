################################################################
# DatasetCreation.py
#
# Description: Tool ton create a dataset of hand landmarks. This script will
# save the landmarks of the user's hand to a CSV file.
#
# Author: Daniel Gebura, djg2170
################################################################
import cv2
import mediapipe as mp
import csv
import os
import time

# Initialize the MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Directory to save the dataset
DATASET_DIR = "gesture_dataset"
os.makedirs(DATASET_DIR, exist_ok=True) # Create the directory if it does not exist

def collect_gesture_data(gesture_label, num_samples=500):
    """
    Collects and saves hand landmaerk data for a given gesture label.

    Args:
        gesture_label (str): The label of the gesture.
        num_samples (int): The number of samples to collect.

    Returns:
        None
    """
    # Create the csv file to save the data
    csv_filename = f"{DATASET_DIR}/{gesture_label}.csv"

    # Open the csv file
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the csv file header for formatting
        header = [gesture_label] + [f"x{i}, y{i}, z{i}" for i in range(1, 22)]
        csv_writer.writerow(header)

        # Initialize the webcam
        cap = cv2.VideoCapture(1)  # Open the webcam

        # Set up to start gesture recording
        print(f"Recording gesture: {gesture_label}")
        print("Press 's' to start recording. Press 'q' to stop recording.")
        recording = False
        sample_count = 0

        # Loop to process the webcam frames
        while cap.isOpened():
            # 1. Read the frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # 2. Flip the frame horizontally for selfie-view display
            frame = cv2.flip(frame, 1)

            # 3. Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 4. Process the frame with MediaPipe Hands
            results = hands.process(rgb_frame)

            # 5. If hands are detected, overlay the landmarks on the frame and save the data
            if results.multi_hand_landmarks:
                # For each hand detected
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the landmarks on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # Save the landmarks if recording
                    if recording and sample_count < num_samples:
                        # Get the landmarks
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        # Write the landmarks to the csv file
                        csv_writer.writerow([gesture_label] + landmarks)
                        sample_count += 1  # Increment the sample count
                        print(f"Sample {sample_count}/{num_samples} collected.")

            # 6. Display the frame
            cv2.putText(frame, f"Gesture: {gesture_label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Display the gesture label
            cv2.putText(frame, "Press 's' to start, 'q' to quit.", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Display the instructions
            cv2.imshow('Gesture Data Collection', frame)  # Display

            #7. Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not recording:
                recording = True
                sample_count = 0
            elif key == ord('q'):
                break
        
        # Release the webcam and close the csv file
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print(f"Dataset saved to {csv_filename}")


# Main function
if __name__ == "__main__":
    gesture_name = input("Enter the gesture name (e.g., closed_fist): ").strip().lower()
    collect_gesture_data(gesture_name, num_samples=500)

