################################################################
# ReviewAndReviseLabels.py
#
# Description: Interactive tool to visually review and revise gesture labels
# frame-by-frame from a CSV file. Arrow keys allow navigation and editing.
#
# Author: Daniel Gebura, djg2170
################################################################

import cv2
import csv
import pandas as pd
from pynput import keyboard

# === CONFIGURATION ===
DIRECTORY = "../labeled_videos/simple_all_gestures/"
VIDEO_PATH = DIRECTORY + "video.mp4"          # Path to the video
CSV_PATH = DIRECTORY + "right_labels.csv"    # Path to the CSV file
# LABELS = ["forward_point", "back_point", "left_point", "right_point", "open_hand", "index_thumb"]  # Left Hand
LABELS = ["closed_fist", "open_hand", "thumbs_up", "index_thumb", "pinky_thumb", "thumbs_down"]  # Right Hand

# === LOAD LABEL DATA ===
df = pd.read_csv(CSV_PATH, dtype={'frame': int, 'label': str})
df.set_index('frame', inplace=True)

# === LOAD VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / (fps / 2))  # Half-speed playback

# === CONTROL STATE ===
current_frame = 0
advance = False
backtrack = False

def on_press(key):
    """Set control flags based on key press."""
    global advance, backtrack
    if key == keyboard.Key.right:
        advance = True
    elif key == keyboard.Key.left:
        backtrack = True

def on_release(key):
    """Reset control flags on key release."""
    global advance, backtrack
    if key == keyboard.Key.right:
        advance = False
    elif key == keyboard.Key.left:
        backtrack = False

# Start keyboard listener in background
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# === MAIN LOOP ===
while cap.isOpened():
    # Clamp frame index bounds
    current_frame = max(0, min(current_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))

    # Seek to desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        print("End of video or error.")
        break

    # Retrieve label for current frame
    label = df.loc[current_frame, 'label'] if current_frame in df.index else "MISSING"

    # Overlay text on frame
    cv2.putText(frame, f"Frame: {current_frame}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Label: {label}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Label Review Tool", frame)

    # Wait with adjustable delay
    key = cv2.waitKey(frame_delay) & 0xFF

    # === LABEL EDITING ===
    if key in [ord(str(i)) for i in range(len(LABELS))]:
        new_label = LABELS[int(chr(key))]
        df.loc[current_frame] = new_label
        print(f"Frame {current_frame} updated to label: {new_label}")

    # === QUIT ===
    elif key == 27:  # ESC
        print("Exiting...")
        break

    # === FRAME NAVIGATION ===
    if advance:
        current_frame += 1
    elif backtrack:
        current_frame -= 1
    # Otherwise stay on the current frame

# === CLEANUP ===
cap.release()
cv2.destroyAllWindows()
listener.stop()

# === SAVE REVISED LABELS ===
df = df.reset_index()
df.to_csv(CSV_PATH, index=False)
print(f"Updated labels saved to {CSV_PATH}")
