# %% Imports
from datasets import get_dataset_split_names, load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# %% Transforms
# Faithful VGG preprocessing: rescale the shortest side to 256, then take a
# 224x224 crop (random + horizontal flip for training, center crop for eval).
# This keeps the VGG-19 conv/FC geometry intact (224 input -> 7x7 feature map).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

transform_eval = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class ApplyTransform:
    """Top-level (picklable) transform wrapper.

    DataLoader workers use `spawn` on macOS and must pickle the dataset — and
    with it, its transform. A closure (`_make_transform.<locals>.apply_transform`)
    is not picklable and fails worker startup; a module-level class is.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch):
        # Some ImageNet images are grayscale/CMYK; force 3-channel RGB.
        batch['image'] = [self.transform(img.convert("RGB"))
                          for img in batch['image']]
        return batch


# %% Dataset
# ImageNet-100: 100-class subset of ImageNet-1k (~130k train / 5k val images),
# full resolution. Feasible to train VGG-19 from scratch on modest hardware
# while staying faithful to the original 224x224 / real-photo design point.
DATASET_NAME = "clane9/imagenet-100"
split = get_dataset_split_names(DATASET_NAME)  # ['train', 'validation']

ds = load_dataset(DATASET_NAME)
ds = ds.with_format("torch")

ds_train = ds['train'].with_transform(ApplyTransform(transform_train))
ds_val = ds['validation'].with_transform(ApplyTransform(transform_eval))

# %% Dataloaders
# 224x224 VGG-19 activations are memory-heavy; 64 is a safe default on an MPS
# Mac. Bump it if you have headroom.
BATCH_SIZE = 16
# JPEG decode + resize/crop is the likely bottleneck; parallelize it across
# worker processes so the GPU isn't waiting on the main thread. On macOS these
# use `spawn`, so the training entry point should run as a real script (guarded
# by `if __name__ == "__main__"`), not cell-by-cell, or workers may misbehave.
NUM_WORKERS = 6

trainloader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=NUM_WORKERS, persistent_workers=True,
                         prefetch_factor=4)
validationloader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, persistent_workers=True,
                              prefetch_factor=4)
# ImageNet-100 ships no separate test split; the validation set is the held-out
# eval set. Alias kept so trainers importing `testloader` still work.
testloader = validationloader
# %%
