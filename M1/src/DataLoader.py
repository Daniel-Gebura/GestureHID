################################################################
# DataLoader.py
#
# Description: This script will load the dataset and prepare it for training.
#
# Author: Daniel Gebura, djg2170
################################################################
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class HandGestureDataset(Dataset):
    """
    Custom PyTorch Dataset for hand gesture landmarks.
    """
    def __init__(self, file_path):
        """
        Initializes the dataset by loading the csv file with the data.

        Args:
            file_path (string): Path to the csv file with annotations.
        """
        df = pd.read_csv(file_path)
        
        # Map gesture labels to binary values
        label_mapping = {"closed_fist": 1, "none": 0}
        df["gesture_label"] = df["gesture_label"].map(label_mapping)

        # Extract features and labels
        self.X = df.iloc[:, 1:].values
        self.y = df["gesture_label"].values.reshape(-1, 1)

        # Normalize features
        self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())

        # Convert to tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns the features and label of a specific sample.
        """
        return self.X[idx], self.y[idx]
    
def get_data_loaders(file_path, batch_size=32, test_size=0.2):
    """
    Returns the DataLoader objects for the training and validation sets.

    Args:
        file_path (string): Path to the csv file with annotations.
        batch_size (int): Number of samples in each batch.
        test_size (float): Fraction of the dataset to include in the test split.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
    """
    # Create the dataset
    dataset = HandGestureDataset(file_path)

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(dataset.X, dataset.y, test_size=test_size)

    # Create DataLoaders
    train_loader = DataLoader(dataset=HandGestureDataset(file_path), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=HandGestureDataset(file_path), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader