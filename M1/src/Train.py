################################################################
# Train.py
#
# Description: This script will train the neural network model for hand gesture recognition.
#
# Author: Daniel Gebura, djg2170
################################################################
import torch
import torch.optim as optim
import torch.nn as nn
from Model import GestureClassifier
from DataLoader import get_data_loaders
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Training Hyperparameters
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
train_loader, val_loader = get_data_loaders("../data/dataset.csv", batch_size=BATCH_SIZE)

# Initialize the model
model = GestureClassifier().to(DEVICE)
criterion = nn.BCELoss()  # Loss = Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Optimizer = Adam

# Initialize TensorBoard
writer = SummaryWriter("../runs/binary_gesture_classifier")

# Training/Validation loop
for epoch in range(EPOCHS):
    # TRAINING LOOP ----------------------------------------------------------
    model.train()  # Set model to training mode
    running_loss = 0.0  # Initialize running loss
    correct, total = 0, 0  # Initialize correct and total predictions

    # Loop over the training set
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Move data to device

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # 1. Forward pass
        loss = criterion(outputs, labels)  # 2. Calculate loss
        loss.backward()  # 3. Backward pass
        optimizer.step()  # 4. Optimize

        # Calculate running loss
        running_loss += loss.item()
        predictions = (outputs > 0.5).float()  # Convert to binary predictions
        correct += (predictions == labels).sum().item()  # Count correct predictions
        total += labels.size(0)  # Count total predictions

    # Calculate training accuracy and loss
    train_accuracy = correct / total
    train_loss = running_loss / len(train_loader)

    # Write training accuracy and loss to TensorBoard
    writer.add_scalar("Training Loss", train_loss, epoch)
    writer.add_scalar("Training Accuracy", train_accuracy, epoch)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    # VALIDATION LOOP --------------------------------------------------------
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0  # Initialize validation loss
    correct, total = 0, 0  # Initialize correct and total predictions

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Move data to device

            outputs = model(inputs)  # 1. Forward pass
            loss = criterion(outputs, labels)  # 2. Calculate loss

            # Calculate validation loss
            val_loss += loss.item()
            predictions = (outputs > 0.5).float()  # Convert to binary predictions
            correct += (predictions == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Count total predictions

    # Calculate validation accuracy and loss
    val_accuracy = correct / total
    val_loss = val_loss / len(val_loader)

    # Write validation accuracy and loss to TensorBoard
    writer.add_scalar("Validation Loss", val_loss, epoch)
    writer.add_scalar("Validation Accuracy", val_accuracy, epoch)

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Save the model
torch.save(model.state_dict(), "../models/binary_gesture_classifier.pth")
writer.close()