import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
import torchvision.transforms as transforms
import Data_fingerprint import fingerprint


class NN_Trainer_Segmentation():
    def __init__(self, model, NUM_EPOCHS=20, BATCH_SIZE=32, save_dir="train_progress", data_folder=None) -> None:
        self.model = model
        self.NUM_EPOCHS = NUM_EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.train_losses = []
        self.test_losses = []
        self.y_test = []
        self.y_test_hat = []
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # Dataloader from input folder + split into train/test/validation sets
    def make_dataloaders(self, data_folder):

        # Extract dataset fingerprint from the specified folder
        dataset_fingerprint = fingerprint(data_folder)
        normalization_mean, normalization_std = dataset_fingerprint[
            'mean'], dataset_fingerprint['std']

        # Define transformations for the images
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalization_mean, normalization_std)
        ])

        # Load the dataset from the specified folder
        dataset = torchvision.datasets.ImageFolder(
            root=data_folder, transform=transform)

        # Split the dataset into train, validation, and test sets
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
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
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, weight_decay=5*1e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1)
        model.train()
        for epoch in range(self.NUM_EPOCHS):
            # Train the model
            loss_epochs = []
            for i, batch in enumerate(self.trainloader, 0):
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
                # print(
                #     f"Epoch {epoch+1}/{self.NUM_EPOCHS}, Batch {i+1}/{len(trainloader)}, Loss: {loss.item():.4f}")
            print(
                f"Epoch {epoch+1}/{self.NUM_EPOCHS}, Average Loss: {np.mean(loss_epochs):.4f}")
            self.train_losses.append(np.mean(loss_epochs))
            # Evaluate on the test set
            model.eval()
            test_loss_epochs = []
            self.y_test = []
            self.y_test_hat = []
            for batch in self.testloader:
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
            self.test_losses.append(np.mean(test_loss_epochs))
            # Save loss plot after each epoch
            self.plot_train_test()
            plt.savefig(os.path.join(self.save_dir,
                        f"loss_plot_epoch_{epoch+1}.png"))
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
