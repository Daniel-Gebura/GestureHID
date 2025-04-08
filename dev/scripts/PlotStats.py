################################################################
# PlotStats.py
#
# Description: This script reads full_evaluation_results.csv and
# creates 3 visualizations showing accuracy vs total latency 
# under controlled variable sweeps:
# - Hidden layer size
# - Tracking confidence
# - Gesture confidence threshold
#
# Author: Daniel Gebura
################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
CSV_PATH = "../results/full_evaluation_results.csv"
RESULTS_DIR = "../results"
PLOT_NAME = "accuracy_vs_latency_config_sweep.png"

# --- LOAD DATA ---
df = pd.read_csv(CSV_PATH)

# --- PREPARE PLOT STYLING ---
sns.set(style="whitegrid")
plt.figure(figsize=(14, 6))

# --- PLOT 1: Hidden Size Sweep (tracking = 0.6, gesture = 0.3) ---
plt.subplot(1, 3, 1)
subset1 = df[(df['tracking_conf'] == 0.6) & (df['gesture_conf'] == 0.3)]
sns.scatterplot(
    data=subset1,
    x="total_latency", y="left_acc",
    hue="hidden_size", style="hidden_size", s=120, palette="deep"
)
plt.title("Varying Hidden Size\n(tracking=0.6, gesture=0.3)")
plt.xlabel("Latency (ms)")
plt.ylabel("Left Accuracy")
plt.legend(title="Hidden Size", loc="best")

# --- PLOT 2: Tracking Confidence Sweep (hidden = 16, gesture = 0.3) ---
plt.subplot(1, 3, 2)
subset2 = df[(df['hidden_size'] == 16) & (df['gesture_conf'] == 0.3)]
sns.scatterplot(
    data=subset2,
    x="total_latency", y="left_acc",
    hue="tracking_conf", style="tracking_conf", s=120, palette="tab10"
)
plt.title("Varying Tracking Confidence\n(hidden=16, gesture=0.3)")
plt.xlabel("Latency (ms)")
plt.ylabel("Left Accuracy")
plt.legend(title="Tracking", loc="best")

# --- PLOT 3: Gesture Threshold Sweep (hidden = 16, tracking = 0.6) ---
plt.subplot(1, 3, 3)
subset3 = df[(df['hidden_size'] == 16) & (df['tracking_conf'] == 0.6)]
sns.scatterplot(
    data=subset3,
    x="total_latency", y="left_acc",
    hue="gesture_conf", style="gesture_conf", s=120, palette="Set2"
)
plt.title("Varying Gesture Threshold\n(hidden=16, tracking=0.6)")
plt.xlabel("Latency (ms)")
plt.ylabel("Left Accuracy")
plt.legend(title="Gesture Conf", loc="best")

# --- SAVE FIGURE ---
plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, PLOT_NAME)
plt.savefig(save_path)
print(f"[INFO] Plot saved to {save_path}")
