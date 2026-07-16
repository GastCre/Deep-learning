# %% Adding the system path to import the dataset module
from Modules.trainer_segmentation import NN_Trainer_Segmentation, SegmentationDataset
from torch.utils.data import DataLoader, Subset, random_split
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
os.chdir("/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/Deep learning")


# Cap MPS allocation below physical RAM so an oversized batch raises a catchable
# OOM error instead of exhausting unified memory and restarting the machine.
# Must be set before torch initializes the MPS backend. MPS requires
# low <= high, so lower both (defaults are 1.4 / 1.7).
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.6"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"

# %% Encoder and Decoder Blocks for UNet Architecture


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def encoder_block(in_channels, out_channels, maxpool=True):
    return nn.Sequential(
        double_conv(in_channels, out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2) if maxpool else nn.Identity()
    )


def encoder_to_decoder_block(in_channels, out_channels):
    return nn.Sequential(
        double_conv(in_channels, out_channels)
    )


def decoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        double_conv(out_channels, out_channels)
    )

# %% UNet architecture


class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = encoder_block(3, 64)
        self.encoder2 = encoder_block(64, 128)
        self.encoder3 = encoder_block(128, 256)
        self.encoder4 = encoder_block(256, 512, maxpool=False)
        self.encoder4_to_decoder1 = encoder_to_decoder_block(512, 256)
        self.decoder1 = decoder_block(512, 256)
        self.decoder2 = decoder_block(256+128, 128)
        self.decoder3 = decoder_block(128+64, 64)
        self.decoder4 = double_conv(64, 3)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Bottleneck: Transition from encoder to decoder
        enc4_to_dec1 = self.encoder4_to_decoder1(enc4)

        # Decoder
        dec1 = self.decoder1(torch.cat((enc4_to_dec1, enc3), dim=1))
        dec2 = self.decoder2(torch.cat((dec1, enc2), dim=1))
        dec3 = self.decoder3(torch.cat((dec2, enc1), dim=1))
        dec4 = self.decoder4(dec3)
        return dec4


# %% Quick test on Oxford-IIIT Pet segmentation

PET_ROOT = "data/OxfordPet"
SIZE = 128            # divisible by 8 (this UNet pools 3x)
N_SUBSET = 300        # keep the smoke test quick; raise for a fuller run

# Download once: images (.jpg) + trimap masks (.png, labels {1,2,3})
torchvision.datasets.OxfordIIITPet(
    root=PET_ROOT, split="trainval", target_types="segmentation", download=True)
images_dir = os.path.join(PET_ROOT, "oxford-iiit-pet", "images")
masks_dir = os.path.join(PET_ROOT, "oxford-iiit-pet", "annotations", "trimaps")

# Fixed normalization to keep this quick — the real fingerprint would read all
# ~7k images. Swap for trainer.make_dataloaders(...) when you want dataset stats.
full = SegmentationDataset(
    images_dir=images_dir, masks_dir=masks_dir, size=SIZE,
    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
    # trimap -> {0,1,2}
    intensity_max=[255, 255, 255], label_values=[1, 2, 3])

subset = Subset(full, list(range(N_SUBSET)))
n_train = int(0.8 * len(subset))
train_ds, test_ds = random_split(
    subset, [n_train, len(subset) - n_train],
    generator=torch.Generator().manual_seed(42))

model = UNet()        # out_channels = 3 == trimap classes
trainer = NN_Trainer_Segmentation(
    model, NUM_EPOCHS=100, BATCH_SIZE=32, LEARNING_RATE=1e-3,
    save_dir="train_progress_pet")
trainer.trainloader = DataLoader(
    train_ds, batch_size=trainer.BATCH_SIZE, shuffle=True)
trainer.testloader = DataLoader(
    test_ds, batch_size=trainer.BATCH_SIZE, shuffle=False)
trainer.validationloader = trainer.testloader

# %%
trainer.train()

# %%
trainer.get_scores()


# %%
