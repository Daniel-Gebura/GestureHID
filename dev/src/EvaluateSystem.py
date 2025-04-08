################################################################
# EvaluateSystem.py
#
# Description: This script evaluates gesture classification accuracy
# and latency across multiple parameters:
# - MediaPipe tracking confidence levels
# - MLP hidden layer sizes
# - Gesture confidence thresholds
#
# Results are saved as plots in ../results/
#
# Author: Daniel Gebura
################################################################

import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import mediapipe as mp
from collections import defaultdict
from Model import GestureClassifier

# Device Optimization -----------------------------------------------

torch.backends.cudnn.benchmark = True  # Enable faster CUDA optimizations
torch.backends.cudnn.deterministic = False  # Avoid strict determinism for speed
torch.set_grad_enabled(False)  # Disable autograd to save computation

# Configuration Constants -------------------------------------------
VIDEO_PATH = "../labeled_videos/simple_all_gestures/video.mp4"
LEFT_LABEL_CSV = "../labeled_videos/simple_all_gestures/left_labels.csv"
RIGHT_LABEL_CSV = "../labeled_videos/simple_all_gestures/right_labels.csv"
RESULTS_DIR = "../results"

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

# MediaPipe tracking confidence settings
TRACKING_CONFIDENCES = [0.4, 0.5, 0.6, 0.7]

# Gesture classifier output probability thresholds
GESTURE_CONF_THRESHOLDS = [0.1, 0.2, 0.3]

LEFT_LABELS = ["forward_point", "back_point", "left_point", "right_point", "open_hand", "index_thumb"]
RIGHT_LABELS = ["closed_fist", "open_hand", "thumbs_up", "index_thumb", "pinky_thumb", "thumbs_down"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
def normalize_landmarks(landmarks):
    """
    Normalize hand landmark coordinates relative to the wrist.

    Args:
        landmarks (list or np.ndarray): List of 63 float values representing 21 (x, y, z) coordinates.

    Returns:
        np.ndarray: Flattened 63-value array normalized by the wrist and largest distance.
    """
    landmarks = np.array(landmarks).reshape(21, 3)
    wrist = landmarks[0]
    landmarks -= wrist
    max_distance = np.max(np.linalg.norm(landmarks, axis=1)) + 1e-8
    return (landmarks / max_distance).flatten()

def load_model(path, hidden_size, output_size):
    """
    Load a pre-trained gesture classifier model.

    Args:
        path (str): Path to the saved .pth model.
        hidden_size (int): Size of the model's hidden layer.
        output_size (int): Number of gesture classes.

    Returns:
        model (torch.nn.Module): Loaded and ready PyTorch model.
    """
    model = GestureClassifier(hidden_size=hidden_size, output_size=output_size).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    model.half()
    return model

# ------------------------------------------------------------------
def evaluate_combination(tracking_conf, hidden_size, left_model_path, right_model_path, gesture_conf):
    """
    Evaluate gesture classification accuracy and latency on labeled video frames.

    Args:
        tracking_conf (float): Tracking confidence for MediaPipe Hands.
        hidden_size (int): Hidden layer size of the MLP model.
        left_model_path (str): Path to the left hand gesture classifier.
        right_model_path (str): Path to the right hand gesture classifier.
        gesture_conf (float): Confidence threshold for gesture predictions.

    Returns:
        dict: A dictionary containing the following keys:
            - "left_acc" (float): Left hand classification accuracy.
            - "right_acc" (float): Right hand classification accuracy.
            - "mp_latency" (float): Average MediaPipe processing latency (in ms).
            - "model_latency" (float): Average model inference latency (in ms).
            - "total_latency" (float): Average total latency (in ms).
            - "per_class_acc" (dict): Per-class accuracy for each gesture class.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.9,
        min_tracking_confidence=tracking_conf
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    left_labels = pd.read_csv(LEFT_LABEL_CSV).set_index('frame')['label'].to_dict()
    right_labels = pd.read_csv(RIGHT_LABEL_CSV).set_index('frame')['label'].to_dict()

    model_left = load_model(left_model_path, hidden_size, len(LEFT_LABELS))
    model_right = load_model(right_model_path, hidden_size, len(RIGHT_LABELS))

    correct_left, correct_right = 0, 0
    total_left, total_right = 0, 0
    mediapipe_latencies, model_latencies, total_latencies = [], [], []

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for frame_idx in tqdm(range(frame_count)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Start total timing
        total_start = time.time()

        # MediaPipe timing
        mp_start = time.time()
        results = hands.process(rgb_frame)
        mp_latency = (time.time() - mp_start) * 1000

        pred_left, pred_right = None, None

        # Gesture classifier timing
        model_start = time.time()
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                norm = normalize_landmarks(landmarks)
                x = torch.tensor(norm, dtype=torch.float16).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    if handedness == 'Left':
                        out = model_left(x)
                        probs = torch.softmax(out, dim=1)[0]
                        conf = probs.max().item()
                        if conf >= gesture_conf:
                            pred_left = LEFT_LABELS[torch.argmax(probs).item()]
                    else:
                        out = model_right(x)
                        probs = torch.softmax(out, dim=1)[0]
                        conf = probs.max().item()
                        if conf >= gesture_conf:
                            pred_right = RIGHT_LABELS[torch.argmax(probs).item()]
        model_latency = (time.time() - model_start) * 1000
        total_latency = (time.time() - total_start) * 1000

        # Accuracy checks and logging
        if frame_idx in left_labels:
            total_left += 1
            true_label = left_labels[frame_idx]
            class_total[true_label] += 1
            if pred_left == true_label:
                correct_left += 1
                class_correct[true_label] += 1

        if frame_idx in right_labels:
            total_right += 1
            true_label = right_labels[frame_idx]
            class_total[true_label] += 1
            if pred_right == true_label:
                correct_right += 1
                class_correct[true_label] += 1

        mediapipe_latencies.append(mp_latency)
        model_latencies.append(model_latency)
        total_latencies.append(total_latency)

    cap.release()
    hands.close()

    return {
        "left_acc": correct_left / total_left if total_left else 0,
        "right_acc": correct_right / total_right if total_right else 0,
        "mp_latency": np.mean(mediapipe_latencies),
        "model_latency": np.mean(model_latencies),
        "total_latency": np.mean(total_latencies),
        "per_class_acc": {cls: class_correct[cls] / class_total[cls] if class_total[cls] else 0 for cls in class_total}
    }

# ------------------------------------------------------------------
def main():
    """
    Run all experimental combinations and generate result plots.
    """
    all_results = []
    per_class_accuracy_records = []

    for gesture_conf in GESTURE_CONF_THRESHOLDS:
        for tracking_conf in TRACKING_CONFIDENCES:
            for hidden_size in LEFT_MODELS:
                result = evaluate_combination(
                    tracking_conf,
                    hidden_size,
                    LEFT_MODELS[hidden_size],
                    RIGHT_MODELS[hidden_size],
                    gesture_conf
                )
                result.update({
                    "tracking_conf": tracking_conf,
                    "hidden_size": hidden_size,
                    "gesture_conf": gesture_conf
                })
                all_results.append(result)

                # Flatten per-class accuracies
                for cls, acc in result["per_class_acc"].items():
                    per_class_accuracy_records.append({
                        "gesture": cls,
                        "accuracy": acc,
                        "tracking_conf": tracking_conf,
                        "hidden_size": hidden_size,
                        "gesture_conf": gesture_conf
                    })

    df = pd.DataFrame(all_results)
    df_classes = pd.DataFrame(per_class_accuracy_records)

    # === Plots ===
    sns.set(style="whitegrid")

    # Left Accuracy vs Latency
    plt.figure()
    sns.lineplot(data=df, x="total_latency", y="left_acc", hue="hidden_size", style="gesture_conf")
    plt.title("Left Hand Accuracy vs Total Latency")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy")
    plt.legend(title="Hidden Layer")
    plt.savefig(os.path.join(RESULTS_DIR, "left_accuracy_vs_latency.png"))

    # Right Accuracy vs Latency
    plt.figure()
    sns.lineplot(data=df, x="total_latency", y="right_acc", hue="hidden_size", style="gesture_conf")
    plt.title("Right Hand Accuracy vs Total Latency")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy")
    plt.legend(title="Hidden Layer")
    plt.savefig(os.path.join(RESULTS_DIR, "right_accuracy_vs_latency.png"))

    # Per-Class Accuracy Bar Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_classes, x="gesture", y="accuracy")
    plt.title("Average Accuracy per Gesture Class")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "per_class_accuracy.png"))

    # Latency breakdown
    plt.figure()
    df_melt = df.melt(id_vars=["hidden_size", "gesture_conf"], value_vars=["mp_latency", "model_latency", "total_latency"],
                      var_name="Latency Component", value_name="Milliseconds")
    sns.barplot(data=df_melt, x="Latency Component", y="Milliseconds", hue="hidden_size")
    plt.title("Average Latency Components by Model Size")
    plt.savefig(os.path.join(RESULTS_DIR, "latency_breakdown.png"))

    print("[INFO] Evaluation complete. Plots saved to ../results/")

    # Save raw results to CSV
    df.to_csv(os.path.join(RESULTS_DIR, "full_evaluation_results.csv"), index=False)
    df_classes.to_csv(os.path.join(RESULTS_DIR, "per_class_accuracy_results.csv"), index=False)

    print("[INFO] Data saved to CSV for future analysis.")


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()