################################################################
# GestureVideoLabeler.py
#
# Description: Tool to label left and right hand gestures in a given video 
# using real-time keyboard and mouse input. Labels are saved to CSV files.
#
# Author: Daniel Gebura, djg2170
################################################################
import cv2
import csv
from pynput import keyboard, mouse
from threading import Thread

# Define the path to the video and output csv files
DIRECTORY = "../labeled_videos/simple_all_gestures/"

# Define label mappings for the left hand
LEFT_LABELS = {
    'w': 'forward_point',
    's': 'back_point',
    'a': 'left_point',
    'd': 'right_point',
    'o': 'open_hand',
    'e': 'index_thumb'
}

# Define label mappings for the right hand
RIGHT_LABELS = {
    'c': 'closed_fist',
    'o': 'open_hand',
    'up': 'thumbs_up',
    'left_click': 'index_thumb',
    'right_click': 'pinky_thumb',
    'down': 'thumbs_down'
}


class GestureLabeler:
    """
    A class that handles frame-by-frame gesture labeling using video input.
    Left and right gestures are labeled separately using keys and mouse input.
    """
    def __init__(self, video_path):
        # Open the video file
        self.cap = cv2.VideoCapture(video_path)

        # Retrieve the frame rate of the video
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

        # Get the total number of frames in the video
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Storage lists for frame labels
        self.labels_left = []
        self.labels_right = []

        # Set initial labeling mode (start with left hand)
        self.mode = 'left'

        # Track current label to apply
        self.current_label = None

        # Number of frames to apply a label to (for 10 labels per second)
        self.advance = int(self.frame_rate // 10)

        # Current frame index in the video
        self.frame_idx = 0

        # Flag to exit early if needed
        self.stop = False

        # Initialize listeners for keyboard and mouse input
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key)
        self.mouse_listener = mouse.Listener(on_click=self.on_click)

    def on_key(self, key):
        """
        Callback for keyboard key presses.
        Maps key to a label depending on the current mode (left/right).
        """
        try:
            char = key.char  # Get character for alphanumeric keys
        except AttributeError:
            # Handle arrow key mapping
            if key == keyboard.Key.up:
                char = 'up'
            elif key == keyboard.Key.down:
                char = 'down'
            else:
                return  # Ignore other special keys

        # Map key to left or right label
        if self.mode == 'left' and char in LEFT_LABELS:
            self.current_label = LEFT_LABELS[char]
        elif self.mode == 'right' and char in RIGHT_LABELS:
            self.current_label = RIGHT_LABELS[char]

    def on_click(self, x, y, button, pressed):
        """
        Callback for mouse clicks.
        Maps mouse buttons to right hand labels.
        """
        if not pressed:
            return  # Only handle press events

        if button == mouse.Button.left:
            self.current_label = RIGHT_LABELS['left_click']
        elif button == mouse.Button.right:
            self.current_label = RIGHT_LABELS['right_click']

    def label_video(self):
        """
        Main loop to play video and label frames via user input.
        Stores labels and saves them to CSV files.
        """
        # Start input listeners
        self.keyboard_listener.start()
        self.mouse_listener.start()

        # Loop until both left and right labeling are completed
        while self.mode in ['left', 'right']:
            # Set video to current frame index
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)

            # Read the current frame
            ret, frame = self.cap.read()
            if not ret:
                break  # Stop if frame could not be read

            # Copy frame and display current label overlay
            display_frame = frame.copy()
            label_text = self.current_label if self.current_label else "Press a key"
            cv2.putText(display_frame, f'{self.mode.title()} Hand: {label_text}', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display frame to user
            cv2.imshow("Labeling", display_frame)

            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC key to quit early
                self.stop = True
                break

            # Apply current label to `advance` number of frames
            if self.current_label:
                for _ in range(self.advance):
                    if self.frame_idx >= self.total_frames:
                        break
                    if self.mode == 'left':
                        self.labels_left.append((self.frame_idx, self.current_label))
                    else:
                        self.labels_right.append((self.frame_idx, self.current_label))
                    self.frame_idx += 1
                self.current_label = None  # Reset label after applying
            else:
                self.frame_idx += 1  # Proceed to next frame

            # Switch to right hand labeling after left hand is done
            if self.frame_idx >= self.total_frames:
                if self.mode == 'left':
                    self.mode = 'right'
                    self.frame_idx = 0  # Reset for right hand
                else:
                    break  # Both hands labeled, exit loop

        # Cleanup resources
        self.cap.release()
        cv2.destroyAllWindows()
        self.keyboard_listener.stop()
        self.mouse_listener.stop()

        # Write labeled frames to CSV files
        with open(DIRECTORY + "left_labels.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'label'])
            writer.writerows(self.labels_left)

        with open(DIRECTORY + "right_labels.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'label'])
            writer.writerows(self.labels_right)

        # Confirmation message
        print("Labeling complete. CSV files saved.")


# Entry point of the script
if __name__ == "__main__":
    # Initialize and run the labeling tool
    labeler = GestureLabeler(DIRECTORY + "video.mp4")
    labeler.label_video()