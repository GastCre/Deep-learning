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


def encoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


def encoder_to_decoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def decoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = encoder_block(3, 64)
        self.encoder2 = encoder_block(64, 128)
        self.encoder3 = encoder_block(128, 256)
        self.encoder4 = encoder_block(256, 512)
        self.encoder4_to_decoder1 = encoder_to_decoder_block(512, 256)
        self.decoder1 = decoder_block(512, 256)
        self.decoder2 = decoder_block(256, 128)
        self.decoder3 = decoder_block(128, 64)
        self.decoder4 = decoder_block(64, 3)
