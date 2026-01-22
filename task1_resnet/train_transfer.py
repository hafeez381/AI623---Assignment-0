"""
Task 1.4: Transfer Learning - The All-in-One Script
Handles 3 modes via command line args:
  1. --mode random    : Random Init (Task 1.4b) - Trains FULL network from scratch
  2. --mode full      : Full Fine-tuning (Task 1.4c) - Trains FULL network from ImageNet weights
  3. --mode lastblock : Last Block Fine-tuning (Task 1.4c) - Trains only last block + FC

Run examples:
  python train_transfer.py --mode random
  python train_transfer.py --mode full
  python train_transfer.py --mode lastblock

Saves: results/<mode>_metrics.json
"""

import os
import json
import random
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from utils import train_one_epoch, evaluate


def get_model(mode, num_classes=10):
    """
    Returns model configured based on mode:
    - random: Random init, TRAIN EVERYTHING.
    - full: Pretrained, TRAIN EVERYTHING (lower LR).
    - lastblock: Pretrained, only layer4 + FC unfrozen.
    """
    # Note: We use 10 classes for CIFAR-10 consistency
    
    if mode == 'random':
        print("Configuring: Random Initialization (Training from Scratch)...")
        # 1. Load Architecture only (No Weights)
        model = models.resnet152(weights=None)
        for p in model.parameters():
            p.requires_grad = True
            
        model.fc = nn.Linear(2048, num_classes)
        lr = 0.0001
        
    elif mode == 'full':
        print("Configuring: Full Fine-Tuning (ImageNet Weights)...")
        # 2. Load ImageNet Weights
        weights = models.ResNet152_Weights.IMAGENET1K_V2
        model = models.resnet152(weights=weights)
        
        # Unfreeze Everything
        for p in model.parameters():
            p.requires_grad = True
            
        model.fc = nn.Linear(2048, num_classes)
        lr = 0.0001
        
    elif mode == 'lastblock':
        print("Configuring: Last Block Fine-Tuning...")
        # 3. Load ImageNet Weights
        weights = models.ResNet152_Weights.IMAGENET1K_V2
        model = models.resnet152(weights=weights)
        
        # Freeze Layers 1-3, Unfreeze Layer 4 + FC
        for name, p in model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
                
        model.fc = nn.Linear(2048, num_classes)
        lr = 0.0001
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return model, lr


def main():
    parser = argparse.ArgumentParser(description='Task 1.4: Transfer Learning')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['random', 'full', 'lastblock'],
                        help='Training mode: random, full, or lastblock')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using Device: {DEVICE}")

    print("Loading CIFAR-10...")
    weights = models.ResNet152_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    
    full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)
    train_data, val_data = random_split(full_train, [45000, 5000], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2)

    # Model Setup
    model, lr = get_model(args.mode, num_classes=10)
    model = model.to(DEVICE)

    # Optimizer (Only optimize parameters that require grad)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nStarting Training ({args.mode} mode)...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1}/{args.epochs} - Val Acc: {val_acc:.2f}%")

    # Save results
    os.makedirs('results', exist_ok=True)
    output_file = f'results/{args.mode}_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(history, f)
    
    print(f"\nTraining complete. Results saved to {output_file}")
    print(f"Final Val Accuracy: {history['val_acc'][-1]:.2f}%")


if __name__ == '__main__':
    main()