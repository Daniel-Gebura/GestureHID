################################################################
# Train.py
#
# Description: This script trains the neural network model for multi-class hand gesture recognition.
#              It logs training and validation metrics to TensorBoard, including:
#              - Loss and accuracy per epoch
#              - Per-class accuracy for both training and validation data
#
# Author: Daniel Gebura, djg2170
################################################################
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from Model import GestureClassifier
from DataLoader import get_data_loaders
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

# ---------------------------- Hyperparameters ---------------------------- #
EPOCHS = 20                # Number of training epochs
BATCH_SIZE = 32            # Batch size for training and validation
LEARNING_RATE = 0.001      # Learning rate for optimizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Load the dataset
train_loader, val_loader = get_data_loaders("../data/right_dataset_v1.csv", hand="right", batch_size=BATCH_SIZE)

# Initialize the model
model = GestureClassifier(hidden_size=32, output_size=6).to(DEVICE)  # Model for 6 gesture classes
criterion = nn.CrossEntropyLoss()  # Loss function: Cross-Entropy for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Optimizer: Adam

# Initialize Tensorboard for logging
writer = SummaryWriter("../runs/mini_right_multiclass_gesture_classifier")

# Training/Validation loop
for epoch in range(EPOCHS):
    # ---------------------------- TRAINING PHASE ---------------------------- #
    model.train()  # Set model to training mode
    running_loss = 0.0  # Track total loss for epoch
    correct_class_counts = defaultdict(int)  # Track correct predictions per class
    total_class_counts = defaultdict(int)  # Track total samples per class

    # Loop over the training dataset
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Move data to GPU/CPU

        optimizer.zero_grad()  # Zero out previous gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights

        # Accumulate loss for logging
        running_loss += loss.item()

        # Compute accuracy per class
        predictions = torch.argmax(outputs, dim=1)  # Get predicted classes
        for label, prediction in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
            total_class_counts[label] += 1
            if label == prediction:
                correct_class_counts[label] += 1

    # Compute overall training loss
    train_loss = running_loss / len(train_loader)

    # Compute per-class accuracy for training
    train_class_accuracies = {f"Training Accuracy/class_{cls}": correct_class_counts[cls] / total_class_counts[cls] 
                              for cls in total_class_counts}

    # Log training metrics to TensorBoard
    writer.add_scalar("Training Loss", train_loss, epoch)
    for cls, acc in train_class_accuracies.items():
        writer.add_scalar(cls, acc, epoch)  # Log per-class accuracy

    # Print training results for the epoch
    print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {train_loss:.4f}")
    for cls, acc in train_class_accuracies.items():
        print(f"   {cls}: Accuracy = {acc:.4f}")

    # ---------------------------- VALIDATION PHASE ---------------------------- #
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0  # Track total validation loss
    correct, total = 0, 0  # Track correct predictions and total samples
    val_correct_class_counts = defaultdict(int)  # Track correct predictions per class
    val_total_class_counts = defaultdict(int)  # Track total samples per class

    with torch.no_grad():  # Disable gradient calculations for validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Move data to GPU/CPU

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            val_loss += loss.item()

            # Compute accuracy for validation
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Compute per-class validation accuracy
            for label, prediction in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                val_total_class_counts[label] += 1
                if label == prediction:
                    val_correct_class_counts[label] += 1

    # Compute overall validation accuracy
    val_accuracy = correct / total
    val_loss = val_loss / len(val_loader)

    # Compute per-class accuracy for validation
    val_class_accuracies = {f"Validation Accuracy/class_{cls}": val_correct_class_counts[cls] / val_total_class_counts[cls] 
                            for cls in val_total_class_counts}

    # Log validation metrics to TensorBoard
    writer.add_scalar("Validation Loss", val_loss, epoch)
    writer.add_scalar("Validation Accuracy", val_accuracy, epoch)
    for cls, acc in val_class_accuracies.items():
        writer.add_scalar(cls, acc, epoch)  # Log per-class accuracy

    # Print validation results for the epoch
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    for cls, acc in val_class_accuracies.items():
        print(f"   {cls}: Accuracy = {acc:.4f}")

# Save the Trained Model
torch.save(model.state_dict(), "../models/mini_right_multiclass_gesture_classifier.pth")
writer.close()