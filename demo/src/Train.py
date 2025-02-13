################################################################
# Train.py
#
# Description: This script trains the neural network model for multi-class hand gesture recognition.
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

# Training Hyperparameters
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
train_loader, val_loader = get_data_loaders("../data/right_dataset_v1.csv", batch_size=BATCH_SIZE)

# Initialize the model
model = GestureClassifier(output_size=6).to(DEVICE)  # 6 defined classes
criterion = nn.CrossEntropyLoss()  # Loss = Cross-Entropy for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Optimizer = Adam

# Initialize TensorBoard
writer = SummaryWriter("../runs/right_multiclass_gesture_classifier")

# Training/Validation loop
for epoch in range(EPOCHS):
    # TRAINING LOOP ----------------------------------------------------------
    model.train()  # Set model to training mode
    running_loss = 0.0  # Initialize running loss
    correct_class_counts = defaultdict(int)
    total_class_counts = defaultdict(int)

    # Loop over the training set
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Move data to device

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # 1. Forward pass
        loss = criterion(outputs, labels)  # 2. Calculate loss
        loss.backward()  # 3. Backward pass
        optimizer.step()  # 4. Optimize

        # 5. Gather logging information
        running_loss += loss.item()  # Accumulate loss
        predictions = torch.argmax(outputs, dim=1)  # Predicted class is index with highest probability (This dataset guarantees a defined gesture for each sample)
        for label, prediction in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
            total_class_counts[label] += 1
            if label == prediction:
                correct_class_counts[label] += 1

    # Calculate training accuracy and loss for this epoch
    train_loss = running_loss / len(train_loader)
    class_accuracies = {f"class_{cls}": correct_class_counts[cls] / total_class_counts[cls] for cls in total_class_counts}

    # Log metrics
    writer.add_scalar("Training Loss", train_loss, epoch)
    for cls, acc in class_accuracies.items():
        writer.add_scalar(f"Training Accuracy/{cls}", acc, epoch)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}")
    for cls, acc in class_accuracies.items():
        print(f"   {cls}: Accuracy = {acc:.4f}")

    # VALIDATION LOOP --------------------------------------------------------
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0  # Initialize validation loss
    correct, total = 0, 0  # Initialize correct and total predictions

    # Disable gradient computation for validation
    with torch.no_grad():
        # Loop over the validation set
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Move data to device

            outputs = model(inputs)  # 1. Forward pass
            loss = criterion(outputs, labels)  # 2. Calculate loss

            # 3. Gather logging information
            val_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Calculate validation accuracy and loss
    val_accuracy = correct / total
    val_loss = val_loss / len(val_loader)

    # Log validation metrics
    writer.add_scalar("Validation Loss", val_loss, epoch)
    writer.add_scalar("Validation Accuracy", val_accuracy, epoch)

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), "../models/right_multiclass_gesture_classifier.pth")
writer.close()