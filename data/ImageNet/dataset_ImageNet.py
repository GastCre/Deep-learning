# %% Imports
from datasets import get_dataset_split_names, load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# %% Transforms

transform = transforms.Compose([
    # We use 227x227 instead of 224x224 as noticed by Karpathy (see Wikipedia entry of AlexNet)
    transforms.Resize((227, 227)),
    transforms.ToTensor(),    # Back to tensor, now shape [1, 227, 227]
])


def apply_transform(batch):
    batch['image'] = [transform(img) for img in batch['image']]
    return batch


# %%
ds = load_dataset("evanarlian/imagenet_1k_resized_256")
ds = ds.with_format("torch")
ds = ds.with_transform(apply_transform)
split = get_dataset_split_names("evanarlian/imagenet_1k_resized_256")
ds_train = ds[split[0]]
ds_val = ds[split[1]]
ds_test = ds[split[2]]
# %%
BATCH_SIZE = 4
trainloader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
validationloader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)
# %%
