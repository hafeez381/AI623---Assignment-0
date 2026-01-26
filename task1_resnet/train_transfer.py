"""
Task 1.4: Transfer Learning
Handles 3 modes via command line args:
  1. --mode random    : Random Init - Trains FULL network from scratch
  2. --mode full      : Full Fine-tuning - Trains FULL network from ImageNet weights
  3. --mode lastblock : Last Block Fine-tuning - Trains only last block + FC

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
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, Subset
from utils_update import train_one_epoch, evaluate
from sklearn.model_selection import train_test_split


def get_model(mode, num_classes=10):
    """
    Returns model configured based on mode:
    - random: Random init, TRAIN EVERYTHING.
    - full: Pretrained, TRAIN EVERYTHING (lower LR).
    - lastblock: Pretrained, only layer4 + FC unfrozen.
    """
    if mode == 'random':
        print("Random Init: Training full ResNet-152 from scratch")
        model = models.resnet152(weights=None)
        model.fc = nn.Linear(2048, num_classes)

        # Enable all gradients
        for p in model.parameters():
            p.requires_grad = True
        lr = 0.001
        
    elif mode == 'full':
        print("Full Fine-tuning: Pretrained ResNet-152")
        weights = models.ResNet152_Weights.IMAGENET1K_V2
        model = models.resnet152(weights=weights)
        model.fc = nn.Linear(2048, num_classes)

        # Enable all gradients
        for p in model.parameters():
            p.requires_grad = True
        lr = 0.0001
        
    elif mode == 'lastblock':
        print("Last Block Fine-tuning: Freeze layers 1-3")
        weights = models.ResNet152_Weights.IMAGENET1K_V2
        model = models.resnet152(weights=weights)
        model.fc = nn.Linear(2048, num_classes)

        # Freeze layers 1-3
        for name, p in model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        lr = 0.001
    
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
    
    train_transform = transforms.Compose([
        transforms.Resize(96),
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ])
    val_transform = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ])


    print("Loading CIFAR-10 (Stratified Subset)...")
    train_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=val_transform)
    
    # perform stratified split
    targets = train_full.targets
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=1000,
        train_size=4000,
        random_state=42,
        stratify=targets
    )
    
    # create subsets
    # train_full (augmented) for training indices
    train_data = Subset(train_full, train_idx)
    # val_full (clean) for validation indices
    val_data = Subset(val_full, val_idx)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2)

    model, lr = get_model(args.mode, num_classes=10)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nStarting Training ({args.mode} mode)...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, accumulation_steps=4)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1}/{args.epochs}")

    os.makedirs('results', exist_ok=True)
    output_file = f'results/{args.mode}_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(history, f)
    print(f"\nTraining complete. Results saved to {output_file}")

if __name__ == '__main__':
    main()