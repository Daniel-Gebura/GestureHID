################################################################
# EvaluateModelSize.py
#
# Description:
# A complete evaluation script to analyze gesture classification
# performance and latency across:
# Varying Model Sizes
#
# Metrics:
# - Per-class and overall accuracy
# - Latency breakdown (MediaPipe + Classifier)
# - Accuracy vs Latency
# - Raw data CSV for future analysis
#
# Author: Daniel Gebura
################################################################

import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import mediapipe as mp
from collections import defaultdict
from Model import GestureClassifier

# Configuration -----------------------------------------------------
VIDEO_PATH = "../labeled_videos/simple_all_gestures/video.mp4"
LEFT_LABEL_CSV = "../labeled_videos/simple_all_gestures/left_labels.csv"
RIGHT_LABEL_CSV = "../labeled_videos/simple_all_gestures/right_labels.csv"
RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

LEFT_LABELS = ["forward_point", "back_point", "left_point", "right_point", "open_hand", "index_thumb"]
RIGHT_LABELS = ["closed_fist", "open_hand", "thumbs_up", "index_thumb", "pinky_thumb", "thumbs_down"]

RIGHT_MODELS = {
    64: "../models/right_multiclass_gesture_classifier.pth",
    32: "../models/half_right_multiclass_gesture_classifier.pth",
    16: "../models/mini_right_multiclass_gesture_classifier.pth",
}

LEFT_MODELS = {
    64: "../models/left_multiclass_gesture_classifier.pth",
    32: "../models/half_left_multiclass_gesture_classifier.pth",
    16: "../models/mini_left_multiclass_gesture_classifier.pth",
}

# MediaPipe Hands Settings
MPH_TRACKING_CONFIDENCE = 0.5

# Gesture Classification Settings
GESTURE_CONFIDENCE = 0.3

# Camera Settings
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Device Optimization -----------------------------------------------
torch.backends.cudnn.benchmark = True  # Enable faster CUDA optimizations
torch.backends.cudnn.deterministic = False  # Avoid strict determinism for speed
torch.set_grad_enabled(False)  # Disable autograd to save computation

# Utility Functions -------------------------------------------------
def load_model(path, hidden_size, output_size):
    """
    Load a trained PyTorch model for hand gesture classification from file and set to eval mode.

    Args:
        path (str): Path to model file
        hidden_size (int): Size of the hidden layer in the model
        output_size (int): Number of output classes for classification

    Returns:
        model (torch.nn.Module): Loaded gesture classifier model
    """
    model = GestureClassifier(hidden_size=hidden_size, output_size=output_size).to(DEVICE)  # Move model to CPU or GPU
    model.load_state_dict(torch.load(path, map_location=DEVICE))  # Load model weights
    model.half()  # Convert model to FP16 (Half Precision) to reduce memory usage
    model.eval()  # Set the model to evaluation mode
    return model

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

def evaluate(video_path, left_csv, right_csv, left_model_path, right_model_path,
             hidden_size, track_conf, gesture_conf, experiment_type, experiment_value):
    """
    Evaluate classification performance and latency for one configuration.

    Args:
        video_path (str): Path to video file
        left_csv (str): Path to left hand labels CSV file
        right_csv (str): Path to right hand labels CSV file
        left_model_path (str): Path to left hand model file
        right_model_path (str): Path to right hand model file
        hidden_size (int): Hidden size of the model
        track_conf (float): Tracking confidence for MediaPipe Hands
        gesture_conf (float): Gesture confidence for classification
        experiment_type (str): Type of experiment being run
        experiment_value (str): Value of the experiment
    """
    # Load MediaPipe hands with specified configuration
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.85,
                           min_tracking_confidence=track_conf)

    # Load the labeled video at a specific resolution
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    left_labels = pd.read_csv(left_csv).set_index('frame')['label'].to_dict()
    right_labels = pd.read_csv(right_csv).set_index('frame')['label'].to_dict()

    # Load the gesture classification model
    model_left = load_model(left_model_path, hidden_size, len(LEFT_LABELS))
    model_right = load_model(right_model_path, hidden_size, len(RIGHT_LABELS))

    # Initialize variables to store statistics
    total_latency, mp_latencies, model_latencies = [], [], []
    correct_left, correct_right, total_left, total_right = 0, 0, 0, 0
    class_correct, class_total = defaultdict(int), defaultdict(int)

    # Iterate through the video frames and process them
    for frame_idx in tqdm(range(total_frames)):
        # 1. Read the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: 
            continue

        # 2. Resize, flip, and recolor the frame for MediaPipe
        frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))  # Ensure resolution
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3. Start timers
        total_start = time.time()
        mp_start = time.time()

        # 4. Process the frame with MediaPipe Hands
        results = hands.process(rgb)
        mp_time = (time.time() - mp_start) * 1000  # Record MediaPipe processing time (ms)

        # 5. Initialzie gesture prediction ("none" is treated the same as "open_hand")
        pred_left, pred_right = "open_hand", "open_hand"

        # 6. Predict gesture if hands are detected
        model_start = time.time()
        if results.multi_hand_landmarks:
            # Iterate through the detected hands and classify them
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get the handedness ("Right" or "Left") of this hand
                handedness = results.multi_handedness[hand_idx].classification[0].label

                # Extract hand landmarks, normalize them, and convert to tensor
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                normalized = normalize_landmarks(landmarks)
                landmark_tensor = torch.tensor(normalized, dtype=torch.float16).unsqueeze(0).to(DEVICE)

                # Classify this gesture based on handedness
                with torch.no_grad():
                    if handedness == 'Left':
                        out = model_left(landmark_tensor)  # Forward pass through the model
                        probs = torch.softmax(out, dim=1)[0]  # Get classification probabilities
                        # Update gesture label if most likely class exceeds confidence threshold
                        if probs.max().item() >= gesture_conf:
                            pred_left = LEFT_LABELS[torch.argmax(probs).item()]
                    else:
                        out = model_right(landmark_tensor)
                        probs = torch.softmax(out, dim=1)[0]
                        if probs.max().item() >= gesture_conf:
                            pred_right = RIGHT_LABELS[torch.argmax(probs).item()]

        # 7. Record the gesture prediction time
        model_time = (time.time() - model_start) * 1000
        total_time = (time.time() - total_start) * 1000

        # 8. Update accuracy metrics
        if frame_idx in left_labels:
            total_left += 1
            true = left_labels[frame_idx]
            class_total[true] += 1
            if pred_left == true:
                correct_left += 1
                class_correct[true] += 1

        if frame_idx in right_labels:
            total_right += 1
            true = right_labels[frame_idx]
            class_total[true] += 1
            if pred_right == true:
                correct_right += 1
                class_correct[true] += 1

        # 9. Update latency metrics
        mp_latencies.append(mp_time)
        model_latencies.append(model_time)
        total_latency.append(total_time)

    # Release resources
    cap.release()
    hands.close()

    return {
        "experiment": experiment_type,
        "value": experiment_value,
        "hidden_size": hidden_size,
        "tracking_conf": track_conf,
        "gesture_conf": gesture_conf,
        "left_acc": correct_left / total_left if total_left else 0,
        "right_acc": correct_right / total_right if total_right else 0,
        "overall_acc": (correct_left + correct_right) / (total_left + total_right),
        "mp_latency": np.mean(mp_latencies),
        "model_latency": np.mean(model_latencies),
        "total_latency": np.mean(total_latency),
        "mp_var": np.var(mp_latencies),
        "model_var": np.var(model_latencies),
        "classes": dict({k: class_correct[k] / class_total[k] if class_total[k] else 0 for k in class_total})
    }

# Plotting and main logic -------------------------------------------
def run_experiments():
    results = []
    # Model size sweep
    for size in [64, 32, 16]:
        results.append(evaluate(VIDEO_PATH, LEFT_LABEL_CSV, RIGHT_LABEL_CSV,
                                LEFT_MODELS[size], RIGHT_MODELS[size],
                                size, MPH_TRACKING_CONFIDENCE, GESTURE_CONFIDENCE, "model_size", size))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "evaluation_data.csv"), index=False)

    # -- Plot 1: Per-class + overall accuracy --
    class_acc = []
    for row in results:
        for cls, acc in row["classes"].items():
            class_acc.append({"class": cls, "accuracy": acc, "type": "Per-Class", "group": row["experiment"], "value": row["value"]})
        class_acc.append({"class": "left_overall", "accuracy": row["left_acc"], "type": "Overall", "group": row["experiment"], "value": row["value"]})
        class_acc.append({"class": "right_overall", "accuracy": row["right_acc"], "type": "Overall", "group": row["experiment"], "value": row["value"]})

    # Save data
    df.to_csv(os.path.join(RESULTS_DIR, "model_size_full_results.csv"), index=False)
    acc_df = pd.DataFrame(class_acc)
    acc_df.to_csv(os.path.join(RESULTS_DIR, "model_size_per_class_accuracy.csv"), index=False)


    # Insert plots here
    sns.set(style="whitegrid")
    
    # === PLOT 1: Latency Breakdown per Model Size ===
    plt.figure(figsize=(10, 6))
    latency_df = pd.DataFrame({
        "Model Size": df["hidden_size"],
        "MediaPipe Latency (ms)": df["mp_latency"],
        "Model Latency (ms)": df["model_latency"],
        "Total Latency (ms)": df["total_latency"],
        "MediaPipe Var": df["mp_var"],
        "Model Var": df["model_var"]
    })

    latency_df = latency_df.melt(id_vars="Model Size",
                                  value_vars=["MediaPipe Latency (ms)", "Model Latency (ms)", "Total Latency (ms)"],
                                  var_name="Component", value_name="Latency (ms)")

    plt.title("Latency Breakdown by Model Size")
    sns.barplot(data=latency_df, x="Model Size", y="Latency (ms)", hue="Component", ci=None)
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Average Latency (ms)")
    plt.legend(title="Latency Component")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "latency_breakdown_by_model_size.png"))

    # === PLOT 2: Per-Class and Overall Accuracy per Model Size ===
    plt.figure(figsize=(12, 6))
    acc_df = pd.DataFrame(class_acc)
    acc_df["Model Size"] = acc_df["value"]

    sns.barplot(data=acc_df, x="class", y="accuracy", hue="Model Size")
    plt.title("Gesture Class Accuracy Across Model Sizes")
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")
    plt.xlabel("Gesture Class (including overall)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "class_accuracy_by_model_size.png"))

    # === PLOT 3: Total Accuracy vs Total Latency ===
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="total_latency", y="overall_acc", hue="hidden_size", s=150)
    plt.title("Overall Accuracy vs Total Latency")
    plt.xlabel("Total Latency (ms)")
    plt.ylabel("Average Accuracy (L+R Hands)")
    plt.legend(title="Model Size")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "overall_accuracy_vs_latency.png"))

# Run Everything ----------------------------------------------------
if __name__ == "__main__":
    run_experiments()