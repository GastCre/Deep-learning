# %% Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset

import torchvision
import torchvision.transforms as transforms

# %%

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(34, padding=8),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
])

# %% Dataset
Path = "/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/Deep learning/data/CIFAR100/data"
train_dataset_aug = torchvision.datasets.CIFAR100(
    root=Path,
    train=True,
    download=True,
    transform=transform_train
)

train_dataset_eval = torchvision.datasets.CIFAR100(
    root=Path,
    train=True,
    download=True,
    transform=transform_test
)
test_dataset = torchvision.datasets.CIFAR100(
    root=Path,
    train=False,
    download=True,
    transform=transform_test
)
# %% Dataloaders
BATCH_SIZE = 256
num_samples = len(train_dataset_aug)
train_size = int(0.8 * num_samples)
validation_size = num_samples - train_size
generator = torch.Generator().manual_seed(42)
train_subset_idx, validation_subset_idx = random_split(
    range(num_samples), [train_size, validation_size], generator=generator)

train_indices = train_subset_idx.indices
val_indices = validation_subset_idx.indices

train_subset = Subset(train_dataset_aug, train_indices)
validation_subset = Subset(train_dataset_eval, val_indices)

trainloader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validationloader = DataLoader(
    validation_subset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

testloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# %%
