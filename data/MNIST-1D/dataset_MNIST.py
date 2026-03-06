# %% Imports
from datasets import get_dataset_split_names, load_dataset
from torch.utils.data import DataLoader
# %%
ds = load_dataset("ylecun/mnist")
ds = ds.with_format("torch")
split = get_dataset_split_names("ylecun/mnist")
ds_train = ds[split[0]]
ds_test = ds[split[1]]
# %%
BATCH_SIZE = 4
trainloader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
# %%
