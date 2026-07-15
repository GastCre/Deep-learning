# %% Adding the system path to import the dataset module
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from Modules.trainer_ImageNet100 import NN_Trainer_ImageNet100
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from data.ImageNet.dataset_ImageNet100 import trainloader, testloader, validationloader
os.chdir("/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/Deep learning")


# Cap MPS allocation below physical RAM so an oversized batch raises a catchable
# OOM error instead of exhausting unified memory and restarting the machine.
# Must be set before torch initializes the MPS backend. MPS requires
# low <= high, so lower both (defaults are 1.4 / 1.7).
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.6"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"
# %%


class VGG19(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(
            # Output shape [64, 256, 256]
            in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(
            # Output shape [64, 256, 256]
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # Output shape [64, 128, 128]
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(
            # Output shape [128, 128, 128]
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(
            # Output shape [128, 128, 128]
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # Output shape [128, 64, 64]
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(
            # Output shape [256, 64, 64]
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(
            # Output shape [256, 64, 64]
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(
            # Output shape [256, 64, 64]
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(
            # Output shape [256, 64, 64]
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # Output shape [256, 32, 32]
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1 = nn.Conv2d(
            # Output shape [512, 32, 32]
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(
            # Output shape [512, 32, 32]
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(
            # Output shape [512, 32, 32]
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(
            # Output shape [512, 32, 32]
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # Output shape [512, 16, 16]
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(
            # Output shape [512, 16, 16]
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(
            # Output shape [512, 16, 16]
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(
            # Output shape [512, 16, 16]
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(
            # Output shape [512, 16, 16]
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # Output shape [512, 8, 8]
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape [512*7*7] for 224x224 ImageNet input (5 maxpools)
        self.flatten = nn.Flatten()
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
# This guard is REQUIRED when running as a script (`python VGG19_NN.py`):
# the DataLoader uses `spawn` workers, which re-import this file. Without the
# guard, every worker would re-execute the training below (a fork bomb). It is
# also True in a Jupyter kernel, so cell-by-cell use still works.
if __name__ == "__main__":
    SCRIPT_DIR = "/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/Deep learning/Architectures/VGG-19/"

    # --- Memory smoke test: run ONE batch to confirm BATCH_SIZE fits in MPS
    # memory before the full run. Peak memory is reached within a single
    # forward -> backward -> step. Throwaway model, doesn't touch trainer state.
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    smoke_model = VGG19().to(device)
    smoke_model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        smoke_model.parameters(), lr=0.01, weight_decay=5*1e-4, momentum=0.9)

    batch = next(iter(trainloader))
    inputs, labels = batch['image'].to(device), batch['label'].to(device)
    optimizer.zero_grad()
    loss = loss_fn(smoke_model(inputs), labels)
    loss.backward()
    optimizer.step()
    print(
        f"one batch OK — inputs {tuple(inputs.shape)}, loss {loss.item():.3f}")
    if device.type == "mps":
        print(
            f"MPS peak allocated: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")

    # Free the smoke-test tensors/model before running real training.
    del smoke_model, optimizer, loss, inputs, labels
    if device.type == "mps":
        torch.mps.empty_cache()

    # --- Full training ---
    trainer = NN_Trainer_ImageNet100(
        model=VGG19(), NUM_EPOCHS=50, save_dir=os.path.join(SCRIPT_DIR, "train_progress"))
    trainer.train()
    # trainer.get_scores()
# %%
