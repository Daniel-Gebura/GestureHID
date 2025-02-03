################################################################
# Model.py
#
# Description: This script defines the neural network model for hand gesture recognition.
#
# Author: Daniel Gebura, djg2170
################################################################
import torch
import torch.nn as nn

class GestureClassifier(nn.Module):
    """
    Simple feedforward neural network for binary gesture classification.

    Architecture:
        - Input Layer (63 features)
        - Hidden Layer (32 neurons, ReLU activation)
        - Dropout (30%)
        - Output Layer (1 neuron, Sigmoid activation)
    """
    def __init__(self, input_size=63, hidden_size=32, dropout_rate=0.3):
        """
        Initializes the GestureClassifier.
        
        Args:
            input_size (int): Number of input features (default: 63 for XYZ of 21 hand landmarks).
            hidden_size (int): Number of neurons in the hidden layer (default: 32).
            dropout_rate (float): Dropout rate for regularization (default: 0.3).

        Returns:
            None
        """
        super(GestureClassifier, self).__init__()

        # Define the layers of the neural network
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Fully Connected Layer
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(dropout_rate),  # Apply dropout for regularization
            nn.Linear(hidden_size, 1),  # Output Layer
            nn.Sigmoid()  # Sigmoid activation function for binary classification
        )
    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output probabilities (between 0 and 1) of shape (batch_size, 1).
        """
        return self.model(x)
