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


class UNet(nn.Module):
    def __init__()
