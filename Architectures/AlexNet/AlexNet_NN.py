# %% Adding the system path to import the dataset module
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from data.MNIST_1D.dataset_MNIST import trainloader, testloader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torch
import os
os.chdir("/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/Deep learning")

# %% Imports

# %%


class AlexNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            # Output size: (224 - 11) / 4 + 1 = 54, so output shape [96, 54, 54]
            in_channels=3, out_channels=96, kernel_size=11, stride=4)
        # Output size: (54 - 3) / 2 + 1 = 26, so output shape [96, 26, 26]
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(
            # Output size: (26 - 5 + 2*2) / 1 + 1 = 26, so output shape [256, 26, 26]
            in_channels=96, out_channels=256, kernel_size=5, padding=2)
        # Output size: (26 - 3) / 2 + 1 = 12, so output shape [256, 12, 12]
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(
            # Output size: (12 - 3 + 2*1) / 1 + 1 = 12, so output shape [384, 12, 12]
            in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            # Output size: (12 - 3 + 2*1) / 1 + 1 = 12, so output shape [384, 12, 12]
            in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            # Output size: (12 - 3 + 2*1) / 1 + 1 = 12, so output shape [256, 12, 12]
            in_channels=384, out_channels=256, kernel_size=3, padding=1)
        # Output size: (12 - 3) / 2 + 1 = 5, so output shape [256, 5, 5]
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        # In_features 256 * 5 * 5
        # We can also inspect the size of the output of the flatten by
        # using print(x.shape) in the forward method after flattening, which should give us [batch_size, 256*5*5]
        # Output shape [4096]
        self.fc1 = nn.Linear(in_features=256*5*5, out_features=4096)
        self.dropout1 = nn.Dropout(p=0.5)
        # Output shape [4096]
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.dropout2 = nn.Dropout(p=0.5)
        # Output shape [10] - MNIST has 10 classes
        self.fc3 = nn.Linear(in_features=4096, out_features=10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        # Remove if using CrossEntropyLoss which applies softmax internally
        # x = self.softmax(x)
        return x


# %% Training the model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AlexNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
train_losses = []
test_losses = []
NUM_EPOCHS = 20
for epoch in range(NUM_EPOCHS):
    # Train the model
    loss_epochs = []
    for i, batch in enumerate(trainloader, 0):
        inputs, labels = batch['image'], batch['label']
        inputs, labels = inputs.to(device), labels.to(device)
        # zero gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(inputs)
        # calculate loss
        loss = loss_fn(outputs, labels)
        # backward pass
        loss.backward()
        # update weights
        optimizer.step()
        loss_epochs.append(loss.item())
        # print(
        #     f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}/{len(trainloader)}, Loss: {loss.item():.4f}")
    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {np.mean(loss_epochs):.4f}")
    train_losses.append(np.mean(loss_epochs))
    # Evaluate on the test set
    model.eval()
    test_loss_epochs = []
    y_test = []
    y_test_hat = []
    for batch in testloader:
        inputs, labels = batch['image'], batch['label']
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
            test_loss_epochs.append(loss.item())
        y_test.extend(labels.cpu().numpy())
        y_test_hat.extend(predicted.cpu().numpy())
    print(
        f"Test Loss: {np.mean(test_loss_epochs):.4f}, Test Accuracy: {accuracy_score(y_test, y_test_hat):.4f}")
    test_losses.append(np.mean(test_loss_epochs))
    # Set the model back to train mode for the next epoch
    model.train()

# %% Visualize the training loss
plt.figure(figsize=(10, 7))
sns.lineplot(x=range(NUM_EPOCHS), y=train_losses, label='Train Loss')
sns.lineplot(x=range(NUM_EPOCHS), y=test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()
# %% Confusion matrix
cm = confusion_matrix(y_test, y_test_hat)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# %%
# Plot some test images with their predicted and true labels
plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i+1)
    img = testloader.dataset[i]['image'].permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    plt.title(
        f"True: {y_test[i]}, Pred: {y_test_hat[i]}")
    plt.axis('off')
plt.suptitle('Sample Test Images with True and Predicted Labels')
plt.tight_layout()
plt.show()
# %%
