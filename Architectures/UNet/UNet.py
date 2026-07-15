# %% Adding the system path to import the dataset module
from data.CIFAR100.dataset_CIFAR100 import trainloader, testloader, validationloader
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from Modules.trainer_CIFAR100 import NN_Trainer_CIFAR100
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torch
import os
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

        # Transition from encoder to decoder
        enc4_to_dec1 = self.encoder4_to_decoder1(enc4)

        # Decoder
        dec1 = self.decoder1(torch.cat((enc4_to_dec1, enc3), dim=1))
        dec2 = self.decoder2(torch.cat((dec1, enc2), dim=1))
        dec3 = self.decoder3(torch.cat((dec2, enc1), dim=1))
        dec4 = self.decoder4(dec3)
        return dec4

# %%
