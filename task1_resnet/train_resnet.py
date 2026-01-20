import random
import json
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

    # Load ResNet-152 from PyTorch
    print("Loading ResNet-152...")
    weights = models.ResNet152_Weights.IMAGENET1K_V2
    resnet_model = models.resnet152(weights=weights)

    # Freeze backbone
    for param in resnet_model.parameters():
        param.requires_grad = False

    # Replace final classification layer to match 10 classes in CIFAR-10
    resnet_model.fc = nn.Linear(2048, 10)
    resnet_model = resnet_model.to(DEVICE)

    # Data Processing
    print("Preparing Data...")
    preprocess = weights.transforms()
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess)

    # create data split
    train_data, val_data = random_split(full_train_dataset, 
        [45000, 5000],
        generator=torch.Generator().manual_seed(42)
    )

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet_model.fc.parameters(), lr=0.001)
    num_epochs = 5

    # Metrics storage
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    print("Starting Training...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(resnet_model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(resnet_model, val_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} Complete")

    # Save history
    with open('task1_training_metrics.json', 'w') as f:
            json.dump(history, f)

    # Save model weights
    torch.save(resnet_model.state_dict(), "resnet152_frozen_cifar10.pth")
    print("\nTraining complete. Model and history saved.")

if __name__ == '__main__':
    main()