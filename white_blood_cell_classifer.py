# white_blood_cell_classifier.py

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Determine whether to use GPU or CPU.
# Using GPU (cuda) speeds up training significantly if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Define dataset paths relative to the project folder.
# This makes the project portable so anyone can run it after placing data in the same structure.
train_data_dir = os.path.join("data", "TRAIN")
test_data_dir = os.path.join("data", "TEST")

# Check that the training data exists before continuing.
# This prevents confusing runtime errors later and clearly tells the user what is missing.
if not os.path.exists(train_data_dir):
    raise FileNotFoundError(
        "Training data not found. Please download dataset and place in data/TRAIN"
    )


# Define how each image is preprocessed before being fed into the model.
# - Resize ensures all images are the same size for batching
# - ToTensor converts images into numerical tensors
# - Normalize helps stabilize training by keeping pixel values centered
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# Load datasets using ImageFolder.
# This automatically assigns labels based on folder names (each folder = cell type).
train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_data_dir, transform=transform)

print("Classes:", train_dataset.classes)
print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))


# DataLoaders split the dataset into batches and shuffle training data.
# Shuffling helps the model generalize better instead of memorizing order.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define a simple Convolutional Neural Network.
# Convolutional layers extract spatial features (edges, shapes, textures).
# Fully connected layers use those features to classify the image.
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            # First layer learns basic features like edges
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second layer captures more complex patterns
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third layer extracts higher-level features
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),

            # Convert extracted features into a vector for classification
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),

            # Dropout helps prevent overfitting by randomly turning off neurons
            nn.Dropout(0.3),

            # Final layer outputs probabilities for each class
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# Initialize model and move it to the selected device (CPU/GPU)
model = SimpleCNN(num_classes=len(train_dataset.classes)).to(device)
print(model)


# CrossEntropyLoss is used for multi-class classification problems
loss_fn = nn.CrossEntropyLoss()

# Adam optimizer adjusts model weights to minimize loss efficiently
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train the model over multiple passes through the dataset (epochs)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()  # enables training behaviors like dropout
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Reset gradients from previous step
        optimizer.zero_grad()

        # Forward pass: get predictions
        outputs = model(images)

        # Compute how wrong the predictions are
        loss = loss_fn(outputs, labels)

        # Backpropagation: compute gradients
        loss.backward()

        # Update weights based on gradients
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


# Evaluate model performance on unseen test data
model.eval()
correct = 0
total = 0

with torch.no_grad():  # disables gradient calculation for efficiency
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        # Choose the class with the highest predicted score
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


# Visualize a few predictions to qualitatively assess model performance
test_loader_viz = DataLoader(test_dataset, batch_size=5, shuffle=True)

images, labels = next(iter(test_loader_viz))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# Move tensors back to CPU for plotting
images = images.cpu()
labels = labels.cpu()
predicted = predicted.cpu()

for i in range(5):
    img = images[i].clone()

    # Reverse normalization for display
    img = img * 0.5 + 0.5
    img = img.permute(1, 2, 0)

    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Predicted: {test_dataset.classes[predicted[i]]} | "
        f"True: {test_dataset.classes[labels[i]]}"
    )
    plt.axis("off")
    plt.show()