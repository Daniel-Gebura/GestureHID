################################################################
# Model.py
#
# Description: Defines the neural network model for multi-class hand gesture recognition.
#
# Author: Daniel Gebura, djg2170
################################################################
import torch
import torch.nn as nn

class GestureClassifier(nn.Module):
    """
    Feedforward neural network for multi-class gesture classification.

    Architecture:
        - Input Layer (63 features)
        - Hidden Layer (64 neurons, ReLU activation)
        - Dropout (30%)
        - Output Layer (7 neurons, Softmax activation)
    """

    def __init__(self, input_size=63, hidden_size=64, output_size=7, dropout_rate=0.3):
        """
        Initializes the GestureClassifier.

        Args:
            input_size (int): Number of input features (default: 63 for XYZ of 21 hand landmarks).
            hidden_size (int): Number of neurons in the hidden layer (default: 64 for increased capacity).
            output_size (int): Number of gesture classes (default: 7).
            dropout_rate (float): Dropout rate for regularization (default: 0.3).
        """
        super(GestureClassifier, self).__init__()

        # Define the layers of the neural network
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Fully Connected Layer
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(dropout_rate),  # Dropout for regularization
            nn.Linear(hidden_size, output_size),  # Output Layer
            nn.Softmax(dim=1)  # Softmax activation for multi-class probabilities
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Probability distribution over 7 classes (batch_size, output_size).
        """
        return self.model(x)
