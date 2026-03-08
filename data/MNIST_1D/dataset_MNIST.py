# %% Imports
from datasets import get_dataset_split_names, load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# %% Transforms

transform = transforms.Compose([
    # transforms.ToPILImage(),  # Convert tensor to PIL Image
    # We use 227x227 instead of 224x224 as noticed by Karpathy (see Wikipedia entry of AlexNet)
    transforms.Resize((224, 224)),
    transforms.ToTensor(),    # Back to tensor, now shape [1, 227, 227]
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1 channel to 3
])


def apply_transform(batch):
    batch['image'] = [transform(img) for img in batch['image']]
    return batch


# %%
ds = load_dataset("ylecun/mnist")
ds = ds.with_format("torch")
ds = ds.with_transform(apply_transform)
split = get_dataset_split_names("ylecun/mnist")
ds_train = ds[split[0]]
ds_test = ds[split[1]]
# %%
BATCH_SIZE = 32
trainloader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
# %%
