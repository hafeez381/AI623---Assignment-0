import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using Device: {DEVICE}")

# Load ResNet-152 from PyTorch
weights = models.ResNet152_Weights.IMAGENET1K_V2
resnet_model = models.resnet152(weights=weights)

# Freeze backbone
for param in resnet_model.parameters():
    param.requires_grad = False

# Replace final classification layer to match 10 classes in CIFAR-10
resnet_model.fc = nn.Linear(2048, 10)
resnet_model = resnet_model.to(DEVICE)

# Data Processing
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

