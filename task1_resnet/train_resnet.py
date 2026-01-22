"""
Task 1.1: Baseline - Train ResNet-152 with frozen backbone on CIFAR-10
Run: python train_resnet.py
Saves: results/baseline_metrics.json, checkpoints/resnet_frozen.pth
"""

import os
import json
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from utils import train_one_epoch, evaluate


def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Device configuration
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using Device: {DEVICE}")

    # Load ResNet-152 pretrained
    print("Loading ResNet-152...")
    weights = models.ResNet152_Weights.IMAGENET1K_V2
    model = models.resnet152(weights=weights)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classification head for CIFAR-10
    model.fc = nn.Linear(2048, 10)
    model = model.to(DEVICE)

    # Data
    print("Preparing Data...")
    preprocess = weights.transforms()
    full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)
    train_data, val_data = random_split(full_train, [45000, 5000], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=2)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    num_epochs = 5

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("Starting Training...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs} Complete")

    # Save results
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    with open('results/baseline_metrics.json', 'w') as f:
        json.dump(history, f)

    torch.save(model.state_dict(), "checkpoints/resnet_frozen.pth")
    print("\nTraining complete. Model and history saved.")


if __name__ == '__main__':
    main()