import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from data.CIFAR100.dataset_CIFAR100 import trainloader, validationloader


class NN_Trainer_CIFAR100():
    def __init__(self, model, NUM_EPOCHS=20, save_dir="train_progress") -> None:
        self.model = model
        self.NUM_EPOCHS = NUM_EPOCHS
        self.train_losses = []
        self.val_losses = []
        self.y_validation = []
        self.y_validation_hat = []
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
        model = self.model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.0075, weight_decay=5*1e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=40, gamma=0.1)
        model.train()
        for epoch in range(self.NUM_EPOCHS):
            # Train the model
            loss_epochs = []
            for i, batch in enumerate(trainloader, 0):
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
            # Evaluate on the validation set
            model.eval()
            validation_loss_epochs = []
            self.y_validation = []
            self.y_validation_hat = []
            for batch in validationloader:
                inputs, labels = batch[0], batch[1]
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)
                    validation_loss_epochs.append(loss.item())
                self.y_validation.extend(labels.cpu().numpy())
                self.y_validation_hat.extend(predicted.cpu().numpy())
            print(
                f"Validation Loss: {np.mean(validation_loss_epochs):.4f}, Validation Accuracy: {accuracy_score(self.y_validation, self.y_validation_hat):.4f}")
            self.val_losses.append(np.mean(validation_loss_epochs))
            # Save loss plot after each epoch
            self.plot_train_test()
            plt.savefig(os.path.join(self.save_dir, f"loss_plot.png"))
            plt.close()
            # Step LR scheduler
            scheduler.step()
            # Set the model back to train mode for the next epoch
            model.train()

    def plot_train_test(self):
        plt.figure(figsize=(10, 7))
        sns.lineplot(x=range(1, len(self.train_losses)+1),
                     y=self.train_losses, label='Train Loss')
        sns.lineplot(x=range(1, len(self.val_losses)+1),
                     y=self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

    def visualize(self):
        self.plot_train_test()
        plt.show()

    def get_scores(self):
        #  Accuracy score
        print(
            f"Final Validation Accuracy: {accuracy_score(self.y_validation, self.y_validation_hat):.4f}")
        # Confusion matrix
        cm = confusion_matrix(self.y_validation, self.y_validation_hat)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
