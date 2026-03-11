# %% Adding the system path to import the dataset module
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from data.CIFAR100.dataset_CIFAR100 import trainloader, testloader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torch
import os
os.chdir("/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/Deep learning")


# %%


class VGG19(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(
            # Output shape [64, 224, 224]
            in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(
            # Output shape [64, 224, 224]
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # Output shape [64, 112, 112]
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(
            # Output shape [128, 112, 112]
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(
            # Output shape [128, 112, 112]
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # Output shape [128, 56, 56]
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(
            # Output shape [256, 56, 56]
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(
            # Output shape [256, 56, 56]
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(
            # Output shape [256, 56, 56]
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(
            # Output shape [256, 56, 56]
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # Output shape [256, 28, 28]
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1 = nn.Conv2d(
            # Output shape [512, 28, 28]
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(
            # Output shape [512, 28, 28]
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(
            # Output shape [512, 28, 28]
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(
            # Output shape [512, 28, 28]
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # Output shape [512, 14, 14]
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(
            # Output shape [512, 14, 14]
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(
            # Output shape [512, 14, 14]
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(
            # Output shape [512, 14, 14]
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(
            # Output shape [512, 14, 14]
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # Output shape [512, 7, 7]
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()  # Output shape [512*7*7]
        self.fc1 = nn.Linear(in_features=512*7*7, out_features=4096)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=4096, out_features=100)
        self.relu = nn.ReLU()
        self.batchnorm1_1 = nn.BatchNorm2d(64)
        self.batchnorm1_2 = nn.BatchNorm2d(64)
        self.batchnorm2_1 = nn.BatchNorm2d(128)
        self.batchnorm2_2 = nn.BatchNorm2d(128)
        self.batchnorm3_1 = nn.BatchNorm2d(256)
        self.batchnorm3_2 = nn.BatchNorm2d(256)
        self.batchnorm3_3 = nn.BatchNorm2d(256)
        self.batchnorm3_4 = nn.BatchNorm2d(256)
        self.batchnorm4_1 = nn.BatchNorm2d(512)
        self.batchnorm4_2 = nn.BatchNorm2d(512)
        self.batchnorm4_3 = nn.BatchNorm2d(512)
        self.batchnorm4_4 = nn.BatchNorm2d(512)
        self.batchnorm5_1 = nn.BatchNorm2d(512)
        self.batchnorm5_2 = nn.BatchNorm2d(512)
        self.batchnorm5_3 = nn.BatchNorm2d(512)
        self.batchnorm5_4 = nn.BatchNorm2d(512)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.batchnorm1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.batchnorm1_2(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2_1(x)
        x = self.batchnorm2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.batchnorm2_2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.conv3_1(x)
        x = self.batchnorm3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.batchnorm3_2(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.batchnorm3_3(x)
        x = self.relu(x)
        x = self.conv3_4(x)
        x = self.batchnorm3_4(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        x = self.conv4_1(x)
        x = self.batchnorm4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.batchnorm4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.batchnorm4_3(x)
        x = self.relu(x)
        x = self.conv4_4(x)
        x = self.batchnorm4_4(x)
        x = self.relu(x)
        x = self.maxpool4(x)
        x = self.conv5_1(x)
        x = self.batchnorm5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.batchnorm5_2(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x = self.batchnorm5_3(x)
        x = self.relu(x)
        x = self.conv5_4(x)
        x = self.batchnorm5_4(x)
        x = self.relu(x)
        x = self.maxpool5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        # x = self.softmax(x)
        return x


# %% Training the model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = VGG19().to(device)
loss_fn = nn.CrossEntropyLoss()
# L2 regularization is implemented by adding a weight decay in the optimizer
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.001, weight_decay=5*1e-4, momentum=0.9)
model.train()
train_losses = []
test_losses = []
NUM_EPOCHS = 50
for epoch in range(NUM_EPOCHS):
    # Train the model
    loss_epochs = []
    for i, batch in enumerate(trainloader, 0):
        inputs, labels = batch[0], batch[1]
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
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}/{len(trainloader)}, Loss: {loss.item():.4f}")
    # print(
    #     f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {np.mean(loss_epochs):.4f}")
    train_losses.append(np.mean(loss_epochs))
    # Evaluate on the test set
    model.eval()
    test_loss_epochs = []
    y_test = []
    y_test_hat = []
    for batch in testloader:
        inputs, labels = batch[0], batch[1]
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
# %% Accuracy score
print(f"Final Test Accuracy: {accuracy_score(y_test, y_test_hat):.4f}")
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
