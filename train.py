"""
Jenny Nguyen
April 16, 2026

CS5330 - Final Project: Dental X-Ray Cavity Detection
Build pretrained models and train them on the dental X-ray dataset.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from dataset import (build_labels, DentalDataset,
                     train_transforms, val_transforms,
                     TRAIN_DIR, VALID_DIR, BATCH_SIZE, SEED)


# Build model
def build_model(name):
    """Load a pretrained model and replace the final layer for binary classification."""

    if name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1')
        # swap the final fully connected layer for a 2-class output
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1')
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model


# Training loop 
def train_model(model, train_loader, valid_loader, train_labels,
                device, name='model', epochs=15, lr=1e-4, patience=4):
    """Train the model and evaluate on validation set each epoch.

    Uses class weighting to handle the cavity/no-cavity imbalance,
    saves the best checkpoint when val loss improves, and stops early
    if val loss hasn't improved in patience epochs.
    """
    n_neg = (train_labels['label'] == 0).sum()
    n_pos = (train_labels['label'] == 1).sum()

    # cavity images are a minority so we weight that class higher in the loss
    # otherwise the model just predicts "no cavity" for everything
    weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # training pass
        model.train()
        train_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f'epoch {epoch}/{epochs}'):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()  # reset gradients each batch
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()   # backpropagate
            optimizer.step()  # update weights
            train_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)

        # validation pass
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_loss += criterion(out, labels).item()
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        # record history for plotting
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(valid_loader))
        history['train_acc'].append(correct / total)
        history['val_acc'].append(val_correct / val_total)

        print(f'  train loss: {history["train_loss"][-1]:.4f} | train acc: {history["train_acc"][-1]:.4f} '
              f'| val loss: {history["val_loss"][-1]:.4f} | val acc: {history["val_acc"][-1]:.4f}')

        # save checkpoint if val loss improved
        if history['val_loss'][-1] < best_val_loss:
            best_val_loss     = history['val_loss'][-1]
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'results/best_{name}.pth')
            print(f'  -> val loss improved to {best_val_loss:.4f}, checkpoint saved')
        else:
            epochs_no_improve += 1
            print(f'  -> no improvement for {epochs_no_improve}/{patience} epochs')

        # early stopping
        if epochs_no_improve >= patience:
            print(f'  early stopping at epoch {epoch}')
            break

    return history


# Plot training curves 
def plot_history(history, title):
    """Plot loss and accuracy curves for train and validation sets."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], color='blue', label='train')
    ax1.plot(history['val_loss'], color='red',  label='val')
    ax1.set_title(f'{title} - loss')
    ax1.set_xlabel('epoch')
    ax1.legend()

    ax2.plot(history['train_acc'], color='blue', label='train')
    ax2.plot(history['val_acc'], color='red',  label='val')
    ax2.set_title(f'{title} - accuracy')
    ax2.set_xlabel('epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'results/{title.lower().replace("-", "")}_curves.png')
    plt.show()


def main(argv):
    """Load dataset, train ResNet-18 and DenseNet-121, save best checkpoints."""

    os.makedirs('results', exist_ok=True)
    torch.manual_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device: {device}')

    # load labels and build datasets
    train_labels = build_labels(TRAIN_DIR)
    valid_labels = build_labels(VALID_DIR)

    train_set = DentalDataset(train_labels, TRAIN_DIR, train_transforms)
    valid_set = DentalDataset(valid_labels, VALID_DIR, val_transforms)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)

    # train resnet-18
    print('\n=== training resnet-18 ===')
    resnet = build_model('resnet18').to(device)
    resnet_history = train_model(resnet, train_loader, valid_loader,
                                 train_labels, device, name='resnet18')
    plot_history(resnet_history, 'resnet-18')

    # train densenet-121
    print('\n=== training densenet-121 ===')
    densenet = build_model('densenet121').to(device)
    densenet_history = train_model(densenet, train_loader, valid_loader,
                                   train_labels, device, name='densenet121')
    plot_history(densenet_history, 'densenet-121')

    print('\ntraining complete -- best checkpoints saved to results/')


if __name__ == '__main__':
    main(sys.argv)