################################################################
# DataLoader.py
#
# Description: Loads the dataset, normalizes hand landmarks relative to the wrist,
# and prepares it for training in a PyTorch DataLoader.
#
# Author: Daniel Gebura, djg2170
################################################################

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle

class HandGestureDataset(Dataset):
    """
    Custom PyTorch Dataset for hand gesture classification using normalized hand landmarks.
    """

    def __init__(self, X, y):
        """
        Initializes the dataset from preprocessed tensors.

        Args:
            X (numpy.ndarray): Feature array of hand landmarks.
            y (numpy.ndarray): Label array (integer category labels).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # Use long for multi-class classification

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            (torch.Tensor, torch.Tensor): Feature tensor and corresponding label.
        """
        return self.X[idx], self.y[idx]

def normalize_landmarks(landmarks):
    """
    Normalizes hand landmarks relative to the wrist (landmark 0).
    Ensures location-invariant gesture classification.

    Args:
        landmarks (numpy.ndarray): Hand landmark coordinates (shape: [63,]).

    Returns:
        numpy.ndarray: Normalized landmark coordinates.
    """
    # Extract wrist coordinates (landmark 0)
    wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]

    # Translate all landmarks relative to the wrist
    normalized = landmarks.copy()
    for i in range(21):
        normalized[i * 3] -= wrist_x  # x
        normalized[i * 3 + 1] -= wrist_y  # y
        normalized[i * 3 + 2] -= wrist_z  # z

    # Compute hand scale (max distance from wrist)
    max_distance = max(np.linalg.norm(normalized[i * 3: i * 3 + 3]) for i in range(21))

    # Scale all coordinates
    normalized /= (max_distance + 1e-8)

    return normalized

def get_data_loaders(file_path, batch_size=32, test_size=0.2):
    """
    Loads the dataset from a CSV file, preprocesses it, normalizes landmarks,
    and returns PyTorch DataLoaders.

    Args:
        file_path (str): Path to the dataset CSV file.
        batch_size (int): Batch size for training and validation.
        test_size (float): Fraction of the dataset to allocate to validation.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
    """

    # Load CSV file (no headers)
    df = pd.read_csv(file_path, header=None)

    # Define gesture labels mapping (each gesture is mapped to a unique integer)
    label_mapping = {
        "closed_fist": 0,
        "open_hand": 1,
        "thumbs_up": 2,
        "index_thumb": 3,
        "pinky_thumb": 4,
        "thumbs_down": 5
    }

    # Ensure all labels are correctly mapped
    if not set(df[0].unique()).issubset(set(label_mapping.keys())):
        raise ValueError("Dataset contains unexpected gesture labels not found in the mapping.")

    df[0] = df[0].map(label_mapping)  # Convert string labels to integer values

    # Extract features (X) and labels (y)
    X = df.iloc[:, 1:].values  # All columns except the label column
    y = df.iloc[:, 0].values  # First column contains labels

    # **Shuffle before splitting to prevent ordered bias**
    X, y = shuffle(X, y, random_state=42)

    # **Apply wrist-based normalization to each sample**
    X = np.array([normalize_landmarks(row) for row in X])

    # Split into training and validation sets with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Create PyTorch datasets
    train_dataset = HandGestureDataset(X_train, y_train)
    val_dataset = HandGestureDataset(X_val, y_val)

    # Create DataLoaders with shuffling enabled for training
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for validation

    return train_loader, val_loader
