import torch
import torch.optim as optim
import torch.nn as nn
from model import FakeImageModel
from data_loader import get_data_loaders
import torch.nn.functional as F  # For applying softmax

# set up device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# load dataset
train_loader, val_loader = get_data_loaders('C:/Users/d.Brown/Desktop/d.brown/portfolio0/data')

# initialize model, loss function, optimizer
model = FakeImageModel().to(device)
criterion = nn.CrossEntropyLoss()  # calculate error
optimizer = optim.Adam(model.parameters(), lr=0.001)  # adapt model's weights

# training loop
num_epochs = 10
best_val_accuracy = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # compute model's predictions
    for inputs, labels in train_loader:
        print(f"Labels: {labels}")  # Prints the labels for the batch
    
        # Move data to GPU if available
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Apply softmax to get probabilities (confidence) for each class
        probabilities = F.softmax(outputs, dim=1)

        # Get the predicted class (index of the highest probability)
        _, predicted = torch.max(probabilities, 1)

        # Get the confidence (probability of the predicted class)
        confidence = probabilities[range(len(probabilities)), predicted]

        # Print out the predicted class and its confidence
        print(f"Predicted: {predicted.item()}, Confidence: {confidence.item():.4f}")

        # Calculate statistics
        running_loss += criterion(outputs, labels).item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total * 100
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            # forward pass
            val_outputs = model(val_inputs)

            # calc loss
            loss = criterion(val_outputs, val_labels)

            # accumulate loss
            val_loss += loss.item()

            # calc accuracy
            _, predicted = torch.max(val_outputs, 1)
            val_total += val_labels.size(0)
            val_correct += (predicted == val_labels).sum().item()

    val_accuracy = val_correct / val_total * 100
    val_loss = val_loss / len(val_loader)
    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.2f}%")

    # save model if validation accuracy improves
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'model.pth')

print(f"Training complete. Best Validation accuracy: {best_val_accuracy:.2f}%")
