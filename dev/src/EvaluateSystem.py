################################################################
# ComprehensiveEvaluation.py
#
# Description:
# A complete evaluation script to analyze gesture classification
# performance and latency across:
# 1. Varying Model Sizes
# 2. Varying Tracking Confidences
# 3. Varying Gesture Confidence Thresholds
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

# ---------------- Configuration ----------------
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

TRACKING_CONFIDENCES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
GESTURE_CONF_THRESHOLDS = [0.1, 0.2, 0.3, 0.4]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Device Optimization -----------------------------------------------

torch.backends.cudnn.benchmark = True  # Enable faster CUDA optimizations
torch.backends.cudnn.deterministic = False  # Avoid strict determinism for speed
torch.set_grad_enabled(False)  # Disable autograd to save computation

# ---------------- Utility Functions ----------------
def normalize_landmarks(landmarks):
    """
    Normalize hand landmark coordinates relative to the wrist.
    """
    landmarks = np.array(landmarks).reshape(21, 3)
    wrist = landmarks[0]
    landmarks -= wrist
    max_distance = np.max(np.linalg.norm(landmarks, axis=1)) + 1e-8
    return (landmarks / max_distance).flatten()

def load_model(path, hidden_size, output_size):
    """
    Load a pre-trained PyTorch model for gesture classification.
    """
    model = GestureClassifier(hidden_size=hidden_size, output_size=output_size).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    model.half()
    return model

def evaluate(video_path, left_csv, right_csv, left_model_path, right_model_path,
             hidden_size, track_conf, gesture_conf, experiment_type, experiment_value):
    """
    Evaluate classification performance and latency for one configuration.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.9,
                           min_tracking_confidence=track_conf)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    left_labels = pd.read_csv(left_csv).set_index('frame')['label'].to_dict()
    right_labels = pd.read_csv(right_csv).set_index('frame')['label'].to_dict()

    model_left = load_model(left_model_path, hidden_size, len(LEFT_LABELS))
    model_right = load_model(right_model_path, hidden_size, len(RIGHT_LABELS))

    total_latency, mp_latencies, model_latencies = [], [], []
    correct_left, correct_right, total_left, total_right = 0, 0, 0, 0
    class_correct, class_total = defaultdict(int), defaultdict(int)

    for idx in tqdm(range(frame_count)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.resize(frame, (320, 240))
        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        total_start = time.time()
        mp_start = time.time()
        results = hands.process(rgb)
        mp_time = (time.time() - mp_start) * 1000

        pred_left, pred_right = "open_hand", "open_hand"
        model_start = time.time()

        if results.multi_hand_landmarks:
            for i, lm in enumerate(results.multi_hand_landmarks):
                side = results.multi_handedness[i].classification[0].label
                norm = normalize_landmarks([l.x for l in lm.landmark] +
                                           [l.y for l in lm.landmark] +
                                           [l.z for l in lm.landmark])
                x = torch.tensor(norm, dtype=torch.float16).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    if side == 'Left':
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

        model_time = (time.time() - model_start) * 1000
        total_time = (time.time() - total_start) * 1000

        # Accuracy
        if idx in left_labels:
            total_left += 1
            true = left_labels[idx]
            class_total[true] += 1
            if pred_left == true:
                correct_left += 1
                class_correct[true] += 1

        if idx in right_labels:
            total_right += 1
            true = right_labels[idx]
            class_total[true] += 1
            if pred_right == true:
                correct_right += 1
                class_correct[true] += 1

        mp_latencies.append(mp_time)
        model_latencies.append(model_time)
        total_latency.append(total_time)

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

# ---------------- Plotting and Main Logic ----------------
def run_experiments():
    results = []
    # 1. Model size sweep
    for size in [64, 32, 16]:
        results.append(evaluate(VIDEO_PATH, LEFT_LABEL_CSV, RIGHT_LABEL_CSV,
                                LEFT_MODELS[size], RIGHT_MODELS[size],
                                size, 0.5, 0.3, "model_size", size))
    # 2. Tracking confidence sweep
    for conf in TRACKING_CONFIDENCES:
        results.append(evaluate(VIDEO_PATH, LEFT_LABEL_CSV, RIGHT_LABEL_CSV,
                                LEFT_MODELS[16], RIGHT_MODELS[16],
                                16, conf, 0.3, "tracking_conf", conf))
    # 3. Gesture confidence sweep
    for gconf in GESTURE_CONF_THRESHOLDS:
        results.append(evaluate(VIDEO_PATH, LEFT_LABEL_CSV, RIGHT_LABEL_CSV,
                                LEFT_MODELS[16], RIGHT_MODELS[16],
                                16, 0.6, gconf, "gesture_conf", gconf))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "evaluation_data.csv"), index=False)

    # -- Plot 1: Per-class + overall accuracy --
    class_acc = []
    for row in results:
        for cls, acc in row["classes"].items():
            class_acc.append({"class": cls, "accuracy": acc, "type": "Per-Class", "group": row["experiment"], "value": row["value"]})
        class_acc.append({"class": "left_overall", "accuracy": row["left_acc"], "type": "Overall", "group": row["experiment"], "value": row["value"]})
        class_acc.append({"class": "right_overall", "accuracy": row["right_acc"], "type": "Overall", "group": row["experiment"], "value": row["value"]})

    df_classes = pd.DataFrame(class_acc)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_classes, x="class", y="accuracy", hue="type")
    plt.title("Gesture Accuracy (Class + Overall)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "per_class_and_overall_accuracy.png"))

    # -- Plot 2: Latency variance bar plot --
    plt.figure(figsize=(10, 6))
    df["config"] = df["experiment"] + "_" + df["value"].astype(str)
    sns.barplot(data=df, x="config", y="mp_latency", yerr=df["mp_var"]**0.5, label="MediaPipe")
    sns.barplot(data=df, x="config", y="model_latency", yerr=df["model_var"]**0.5, label="Classifier", bottom=df["mp_latency"])
    plt.xticks(rotation=45)
    plt.legend()
    plt.ylabel("Latency (ms)")
    plt.title("Latency Breakdown with Variance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "latency_variance_stacked_bar.png"))

    # -- Plot 3: Overall Accuracy vs Total Latency --
    plt.figure()
    sns.scatterplot(data=df, x="total_latency", y="overall_acc", hue="experiment", style="value", s=150)
    plt.title("Overall Accuracy vs Total Latency")
    plt.xlabel("Total Latency (ms)")
    plt.ylabel("Overall Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "overall_accuracy_vs_latency.png"))

    print("[INFO] All experiments completed. Results saved to ../results/")

# ---------------- Run Everything ----------------
if __name__ == "__main__":
    run_experiments()