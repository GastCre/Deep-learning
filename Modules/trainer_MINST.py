import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from data.MNIST_1D.dataset_MNIST import trainloader, testloader


class NN_Trainer_MNIST():
    def __init__(self, model, NUM_EPOCHS=20) -> None:
        self.model = model
        self.NUM_EPOCHS = NUM_EPOCHS

    def train(self):
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
        model = self.model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()
        train_losses = []
        test_losses = []
        for epoch in range(self.NUM_EPOCHS):
            # Train the model
            loss_epochs = []
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
                # print(
                #     f"Epoch {epoch+1}/{self.NUM_EPOCHS}, Batch {i+1}/{len(trainloader)}, Loss: {loss.item():.4f}")
            print(
                f"Epoch {epoch+1}/{self.NUM_EPOCHS}, Average Loss: {np.mean(loss_epochs):.4f}")
            train_losses.append(np.mean(loss_epochs))
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
            test_losses.append(np.mean(test_loss_epochs))
            # Set the model back to train mode for the next epoch
            model.train()
        self.train_losses = train_losses
        self.test_losses = test_losses

    def visualize(self):
        plt.figure(figsize=(10, 7))
        sns.lineplot(x=range(self.NUM_EPOCHS),
                     y=self.train_losses, label='Train Loss')
        sns.lineplot(x=range(self.NUM_EPOCHS),
                     y=self.test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
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

    # def store_results(self):
