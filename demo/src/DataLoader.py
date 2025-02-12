################################################################
# DataLoader.py
#
# Description: Loads the dataset and prepares it for training.
#
# Author: Daniel Gebura, djg2170
################################################################

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

class HandGestureDataset(Dataset):
    """
    Custom PyTorch Dataset for hand gesture landmarks.
    """
    def __init__(self, X, y):
        """
        Initializes the dataset from preprocessed tensors.

        Args:
            X (numpy.ndarray): Feature array.
            y (numpy.ndarray): Label array.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_data_loaders(file_path, batch_size=32, test_size=0.2):
    """
    Returns DataLoaders for training and validation.

    Args:
        file_path (string): Path to dataset CSV file.
        batch_size (int): Batch size for training.
        test_size (float): Fraction of dataset used for validation.

    Returns:
        train_loader (DataLoader): DataLoader for training.
        val_loader (DataLoader): DataLoader for validation.
    """
    # Load CSV without headers
    df = pd.read_csv(file_path, header=None)

    # Map gesture labels to binary values
    label_mapping = {"closed_fist": 1, "none": 0}
    df[0] = df[0].map(label_mapping)

    # Extract features (X) and labels (y)
    X = df.iloc[:, 1:].values  # Exclude label column
    y = df.iloc[:, 0].values  # Labels

    # Normalize features (avoid modifying labels)
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-8)

    # **Shuffle before splitting to prevent ordered bias**
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # Create PyTorch datasets
    train_dataset = HandGestureDataset(X_train, y_train)
    val_dataset = HandGestureDataset(X_val, y_val)

    # Create DataLoaders with shuffling enabled for both sets
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader
