import os
import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from data.ImageNet.dataset_ImageNet100 import trainloader, testloader, validationloader


class NN_Trainer_ImageNet100():
    def __init__(self, model, NUM_EPOCHS=20, save_dir="train_progress") -> None:
        self.model = model
        self.NUM_EPOCHS = NUM_EPOCHS
        self.train_losses = []
        self.test_losses = []
        self.y_test = []
        self.y_test_hat = []
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
        model = self.model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, weight_decay=5*1e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1)

        # --- Resume from checkpoint if one exists ---
        # A checkpoint restores weights + optimizer momentum + scheduler position
        # so training continues exactly where it stopped. `epoch` in the file is
        # 0-indexed and marks the last COMPLETED epoch, so we resume at epoch+1.
        ckpt_path = os.path.join(self.save_dir, "checkpoint.pt")
        start_epoch = 0
        if os.path.exists(ckpt_path):
            # weights_only=False because the checkpoint holds more than tensors
            # (epoch int, loss-history lists). Safe here: it's our own file.
            ckpt = torch.load(ckpt_path, map_location=device,
                              weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            self.train_losses = ckpt["train_losses"]
            self.test_losses = ckpt["test_losses"]
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from checkpoint: {start_epoch} epoch(s) already done, "
                  f"continuing to {self.NUM_EPOCHS}.")

        model.train()
        for epoch in range(start_epoch, self.NUM_EPOCHS):
            # Train the model
            loss_epochs = []
            num_batches = len(trainloader)
            epoch_start = time.time()
            for i, batch in enumerate(trainloader, 0):
                inputs, labels = batch['image'], batch['label']
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
                # Heartbeat: progress, throughput and ETA every 50 batches so a
                # long epoch isn't silent. mps.synchronize() makes the timing
                # reflect real GPU work rather than async-dispatched queue time.
                if (i + 1) % 50 == 0:
                    if device.type == "mps":
                        torch.mps.synchronize()
                    elapsed = time.time() - epoch_start
                    rate = (i + 1) / elapsed
                    eta = (num_batches - (i + 1)) / rate if rate > 0 else 0
                    print(
                        f"Epoch {epoch+1}/{self.NUM_EPOCHS}, Batch {i+1}/{num_batches}, "
                        f"Loss: {loss.item():.4f}, {rate:.1f} batch/s, "
                        f"epoch ETA: {eta/60:.1f} min")
            print(
                f"Epoch {epoch+1}/{self.NUM_EPOCHS}, Average Loss: {np.mean(loss_epochs):.4f}")
            self.train_losses.append(float(np.mean(loss_epochs)))
            # Evaluate on the test set
            model.eval()
            test_loss_epochs = []
            self.y_test = []
            self.y_test_hat = []
            for batch in testloader:
                inputs, labels = batch['image'], batch['label']
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)
                    test_loss_epochs.append(loss.item())
                self.y_test.extend(labels.cpu().numpy())
                self.y_test_hat.extend(predicted.cpu().numpy())
            print(
                f"Test Loss: {np.mean(test_loss_epochs):.4f}, Test Accuracy: {accuracy_score(self.y_test, self.y_test_hat):.4f}")
            self.test_losses.append(float(np.mean(test_loss_epochs)))
            # Save loss plot after each epoch
            self.plot_train_test()
            plt.savefig(os.path.join(self.save_dir,
                        f"loss_plot.png"))
            plt.close()
            # Set the model back to train mode for the next epoch
            model.train()
            # Step the scheduler
            scheduler.step()
            # --- Save checkpoint AFTER stepping the scheduler, so the saved
            # scheduler state already accounts for this finished epoch. Write to
            # a temp file then rename: an atomic swap that can't leave a
            # half-written checkpoint if the process dies mid-save.
            tmp_path = ckpt_path + ".tmp"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_losses": self.train_losses,
                "test_losses": self.test_losses,
            }, tmp_path)
            os.replace(tmp_path, ckpt_path)

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

    def get_scores(self):
        #  Accuracy score
        print(
            f"Final Test Accuracy: {accuracy_score(self.y_test, self.y_test_hat):.4f}")
        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.y_test_hat)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
