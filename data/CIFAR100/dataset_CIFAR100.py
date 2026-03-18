# %% Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

# %%

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
])

# %% Dataset
Path = "/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/Deep learning/data/CIFAR100/data"
train_dataset = torchvision.datasets.CIFAR100(
    root=Path,
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR100(
    root=Path,
    train=False,
    download=True,
    transform=transform
)
# %% Dataloaders
BATCH_SIZE = 256

trainloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

testloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# %%
