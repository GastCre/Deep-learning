import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from Modules.Data_fingerprint import fingerprint

# Custom dataset class for segmentation tasks


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Pixels carrying this index are excluded from both the loss and the metrics —
# used for regions the ground truth itself leaves unclassified.
IGNORE_INDEX = 255


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, size, mean, std, intensity_max, label_values=None, augment=False, ignore_values=None, ignore_index=IGNORE_INDEX):
        images_dir, masks_dir = Path(images_dir), Path(masks_dir)
        # Pair each image with the mask of the same stem (extensions may differ,
        # e.g. image .jpg / mask .png); sorted for determinism.
        mask_by_stem = {p.stem: p for p in masks_dir.iterdir()
                        if p.suffix.lower() in IMAGE_EXTS}
        self.pairs = []
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            mask_path = mask_by_stem.get(img_path.stem)
            if mask_path is not None:
                self.pairs.append((img_path, mask_path))
        if not self.pairs:
            raise ValueError(
                f"No image/mask pairs found in {images_dir} / {masks_dir}")
        self.size, self.mean, self.std = size, mean, std
        self.intensity_max = np.asarray(intensity_max, dtype=np.float32)

        # Optional remap of raw mask values -> contiguous class indices {0..C-1}.
        # e.g. label_values=[1,2,3] maps Oxford-Pet trimap to {0,1,2};
        # [0,255] maps a binary mask to {0,1}. Built as an 8-bit lookup table.
        # Values listed in ignore_values map to ignore_index instead of a class,
        # so the loss and the metrics skip those pixels entirely (e.g. the
        # Oxford-Pet "unclassified" boundary band).
        self.ignore_index = ignore_index
        if label_values is None:
            self.lut = None
        else:
            self.lut = np.full(256, -1, dtype=np.int64)
            for idx, value in enumerate(label_values):
                self.lut[value] = idx
            for value in (ignore_values or []):
                self.lut[value] = ignore_index

        # Train-time augmentation. Geometric transforms are applied identically
        # to image and mask; photometric jitter is applied to the image only.
        self.augment = augment
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02) if augment else None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        # keep as-is: single-channel label map
        mask = Image.open(mask_path)

        # Resize first: image bilinear, mask nearest (never blend class ids)
        image = TF.resize(image, [self.size, self.size])   # bilinear default
        mask = TF.resize(mask, [self.size, self.size],
                         interpolation=TF.InterpolationMode.NEAREST)

        # --- augmentation (train only) ---
        if self.augment:
            # geometric: identical flip on image AND mask (no fill needed, stays aligned)
            if torch.rand(1).item() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            # photometric: image only (a mask has no colour)
            image = self.color_jitter(image)

        # --- image: scale by intensity_max -> normalize ---
        # [H,W,3], same divisor as the stats
        arr = np.asarray(image, dtype=np.float32) / self.intensity_max
        image = torch.from_numpy(arr).permute(
            2, 0, 1).contiguous()     # [3,H,W]
        image = TF.normalize(image, self.mean, self.std)

        # --- mask: (remap) -> Long tensor of class indices ---
        mask = np.array(mask)                      # [H,W], raw label values
        if self.lut is not None:
            mask = self.lut[mask]                  # raw values -> {0..C-1}
            if (mask < 0).any():
                raise ValueError(
                    f"Mask {mask_path.name} contains a value not in label_values")
        mask = torch.as_tensor(mask, dtype=torch.long)

        return image, mask


class NN_Trainer_Segmentation():
    def __init__(self, model, NUM_EPOCHS=20, BATCH_SIZE=32, LEARNING_RATE=0.001, WEIGHT_DECAY=1e-4, MOMENTUM=0.9, OPT_STEP_SIZE=30, OPT_GAMMA=0.1, save_dir="train_progress", data_folder=None, TRAIN_SPLIT=0.7, VAL_SPLIT=0.15) -> None:
        self.model = model
        self.NUM_EPOCHS = NUM_EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.MOMENTUM = MOMENTUM
        self.OPT_STEP_SIZE = OPT_STEP_SIZE
        self.OPT_GAMMA = OPT_GAMMA
        self.TRAIN_SPLIT = TRAIN_SPLIT
        self.VAL_SPLIT = VAL_SPLIT
        self.train_losses = []
        self.test_losses = []
        self.y_test = []
        self.y_test_hat = []
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # Dataloader from input folder + split into train/test/validation sets
    def make_dataloaders(self, data_folder, images_subdir="images", masks_subdir="masks", size=256, label_values=None):
        data_folder = Path(data_folder)
        images_dir = data_folder / images_subdir
        masks_dir = data_folder / masks_subdir

        # Fingerprint the images for per-channel normalization stats. Divide by
        # intensity_max so the stats and the pixels share the same scale (Option B).
        dataset_fingerprint = fingerprint(images_dir)
        normalization_mean = dataset_fingerprint.mean / dataset_fingerprint.intensity_max
        normalization_std = dataset_fingerprint.std / dataset_fingerprint.intensity_max

        # Load image/mask pairs from the images/ and masks/ subfolders
        dataset = SegmentationDataset(
            images_dir=images_dir, masks_dir=masks_dir, size=size,
            mean=normalization_mean, std=normalization_std,
            intensity_max=dataset_fingerprint.intensity_max,
            label_values=label_values)

        # Split the dataset into train, validation, and test sets
        total_size = len(dataset)
        train_size = int(self.TRAIN_SPLIT * total_size)
        val_size = int(self.VAL_SPLIT * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size])

        # Create DataLoaders for each set
        self.trainloader = DataLoader(
            train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.validationloader = DataLoader(
            val_dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        self.testloader = DataLoader(
            test_dataset, batch_size=self.BATCH_SIZE, shuffle=False)

    def train(self):
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
        model = self.model.to(device)
        loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.LEARNING_RATE, weight_decay=self.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.NUM_EPOCHS, eta_min=self.OPT_GAMMA)
        model.train()
        for epoch in range(self.NUM_EPOCHS):
            # Train the model
            loss_epochs = []
            for i, batch in enumerate(self.trainloader, 0):
                inputs, labels = batch[0], batch[1]
                inputs, labels = inputs.to(device), labels.to(device)
                # zero gradients
                optimizer.zero_grad()
                # forward pass
                outputs = model(inputs)
                # calculate loss
                loss = loss_fn(outputs, labels)
                # backward pass
                loss.backward()
                # update weights
                optimizer.step()
                loss_epochs.append(loss.item())
                # print(
                #     f"Epoch {epoch+1}/{self.NUM_EPOCHS}, Batch {i+1}/{len(trainloader)}, Loss: {loss.item():.4f}")
            print(
                f"Epoch {epoch+1}/{self.NUM_EPOCHS}, Average Loss: {np.mean(loss_epochs):.4f}")
            self.train_losses.append(np.mean(loss_epochs))
            # Evaluate on the test set
            model.eval()
            test_loss_epochs = []
            self.confusion = None          # C x C pixel confusion, built up per batch
            for batch in self.testloader:
                inputs, masks = batch[0], batch[1]
                inputs, masks = inputs.to(device), masks.to(device)
                with torch.no_grad():
                    outputs = model(inputs)             # [N, C, H, W]
                    loss = loss_fn(outputs, masks)      # masks: [N, H, W] Long
                    predicted = outputs.argmax(dim=1)   # [N, H, W]
                    test_loss_epochs.append(loss.item())
                n_classes = outputs.shape[1]
                if self.confusion is None:
                    self.confusion = np.zeros(
                        (n_classes, n_classes), dtype=np.int64)
                self.confusion += self._pixel_confusion(
                    masks, predicted, n_classes)
            pixel_acc = np.trace(self.confusion) / self.confusion.sum()
            dice = self._mean_dice(self.confusion)
            print(
                f"Test Loss: {np.mean(test_loss_epochs):.4f}, Pixel Acc: {pixel_acc:.4f}, mean Dice: {dice:.4f}")
            self.test_losses.append(np.mean(test_loss_epochs))
            # Save loss plot after each epoch
            self.plot_train_test()
            plt.savefig(os.path.join(self.save_dir,
                        f"loss_plot.png"))
            plt.close()
            # Set the model back to train mode for the next epoch
            model.train()
            # Step the scheduler
            scheduler.step()

    def plot_train_test(self):
        plt.figure(figsize=(10, 7))
        sns.lineplot(x=range(len(self.train_losses)),
                     y=self.train_losses, label='Train Loss')
        sns.lineplot(x=range(len(self.test_losses)),
                     y=self.test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()

    def visualize(self):
        self.plot_train_test()
        plt.show()

    @staticmethod
    def _pixel_confusion(target, pred, n_classes):
        # Flatten to 1-D and bin (true, pred) pairs into a C x C matrix
        t = target.reshape(-1).cpu().numpy()
        p = pred.reshape(-1).cpu().numpy()
        # Drop ignored pixels: they have no ground-truth class, and their index
        # lies outside [0, n_classes) so they would overflow the reshape below.
        keep = t != IGNORE_INDEX
        t, p = t[keep], p[keep]
        k = t * n_classes + p
        return np.bincount(k, minlength=n_classes ** 2).reshape(n_classes, n_classes)

    @staticmethod
    def _mean_dice(cm):
        # Per-class Dice = 2*TP / (2*TP + FP + FN) = 2*diag / (row_sum + col_sum),
        # averaged over classes. row_sum = true totals, col_sum = predicted totals.
        tp = np.diag(cm)
        denom = cm.sum(axis=1) + cm.sum(axis=0)
        return np.mean(2 * tp / np.maximum(denom, 1))

    def get_scores(self):
        cm = self.confusion
        pixel_acc = np.trace(cm) / cm.sum()
        print(
            f"Final Pixel Accuracy: {pixel_acc:.4f}, mean Dice: {self._mean_dice(cm):.4f}")
        # Per-pixel confusion matrix (rows = true class, cols = predicted)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
