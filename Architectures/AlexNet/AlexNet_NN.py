# %% Adding the system path to import the dataset module
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from data.ImageNet.dataset_ImageNet import trainloader, testloader, validationloader
import seaborn as sns
import os
os.chdir("/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/Deep learning")


# %%


class AlexNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Out_channels = (in_channels - kernel_size) / stride + 1
        # First layer out_channel is a design choice
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(
            # Out_channels = 96 - 5 + 2*2 +1 = 96
            in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=256*5*5, out_features=4096)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)
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

losses = []
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
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
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}/{len(trainloader)}, Loss: {loss.item():.4f}")
    losses.append(float(loss.item()))

# %% Visualize the training loss
sns.lineplot(x=range(NUM_EPOCHS), y=losses)

# %%
