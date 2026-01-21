"""
Task 1.2: Train ResNet-152 with disabled skip connections.
Run: python train_broken.py
Saves: results/broken_metrics.json
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


class BottleneckNoSkip(nn.Module):
    """
    Modified Bottleneck block with disabled skip connection.
    Source code for resnet: https://docs.pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
    """
    expansion = 4
    
    def __init__(self, original_block):
        super().__init__()
        self.conv1 = original_block.conv1
        self.bn1 = original_block.bn1
        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2
        self.conv3 = original_block.conv3
        self.bn3 = original_block.bn3
        self.relu = original_block.relu
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        # NO skip connection
        out = self.relu(out)
        return out


def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Device configuration
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using Device: {DEVICE}")

    # Load model and disable skip connections in layer4
    print("Loading ResNet-152 with disabled skip connections...")
    weights = models.ResNet152_Weights.IMAGENET1K_V2
    model = models.resnet152(weights=weights)
    
    # Disable skip connections in layer4
    for idx in [0, 1, 2]:
        model.layer4[idx] = BottleneckNoSkip(model.layer4[idx])
    
    # Freeze all except layer4 and fc
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    
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
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("Starting Training...")
    for epoch in range(5):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1}/5 Complete")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/broken_metrics.json', 'w') as f:
        json.dump(history, f)
    print("\nTraining complete. Metrics saved to results/broken_metrics.json")

    # Save model
    # torch.save(model.state_dict(), "checkpoints/resnet_broken.pth")


if __name__ == '__main__':
    main()
