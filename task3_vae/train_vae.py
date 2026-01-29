"""
VAE Training Script
Trains the VAE on FashionMNIST with baseline or mitigated (KL annealing) mode.

Usage:
    python train_vae.py --mode baseline --epochs 30
    python train_vae.py --mode mitigated --epochs 30
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from architecture import VAE
from vae_utils import vae_loss_function, save_history


def get_data_loaders(batch_size=128, val_split=0.2, data_dir='./data'):
    """
    Download FashionMNIST and create train/val/test data loaders.
    
    Args:
        batch_size: Number of images per batch
        val_split: Fraction of training data to use for validation
        data_dir: Directory to store the dataset
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Transform: convert image to tensor and normalize to [0, 1]
    transform = transforms.ToTensor()
    
    # Download FashionMNIST
    full_train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    
    # Split training data into train and validation
    train_size = int((1 - val_split) * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # Create Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    print(f"Data Loaded: Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    return train_loader, val_loader, test_loader


def get_beta(epoch, num_epochs, mode='baseline'):
    """
    Get the beta value for KL weighting.
    
    Args:
        epoch: Current training epoch (0-indexed)
        num_epochs: Total number of epochs
        mode (str): 'baseline' (beta=1) or 'mitigated' (KL annealing)
        
    Returns:
        beta (float): Weight for KL divergence term (0.0 to 1.0)
    """
    if mode == 'baseline':
        return 1.0
    elif mode == 'mitigated':
        # KL Annealing: Slowly increase beta from 0 to 1 over first 50% of epochs
        # This allows the encoder to learn features before KL regularization kicks in
        warmup_end = int(num_epochs * 0.5)
        if epoch < warmup_end:
            return epoch / warmup_end
        return 1.0
    else:
        raise ValueError(f"Unknown mode: {mode}")


def train_epoch(model, loader, optimizer, device, beta):
    """
    Runs one full pass of training over the dataset.

    Args:
        model: The model to train (VAE)
        loader: the training data
        optimizer: the optimization algorithm
        device: 'cpu' or 'cuda'/'mps'
        beta: current weight for KL divergence

    Returns:
        tuple: (avg_total_loss, avg_recon_loss, avg_kl_loss)
    """
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for data, _ in loader:
        data = data.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        
        # Calculate Loss
        loss, recon, kl = vae_loss_function(recon_batch, data, mu, logvar, beta)
        
        # Backward pass & Update weights
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        
    n_samples = len(loader.dataset)
    return total_loss/n_samples, total_recon/n_samples, total_kl/n_samples


def evaluate(model, loader, device, beta=1.0):
    """
    Runs evaluation on validation/test set (no gradient updates).
    """
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, recon, kl = vae_loss_function(recon_batch, data, mu, logvar, beta)
            
            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            
    n_samples = len(loader.dataset)
    return total_loss/n_samples, total_recon/n_samples, total_kl/n_samples


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train VAE on FashionMNIST')
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'mitigated'])
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    
    torch.manual_seed(42)
    
    # Device setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs/losses', exist_ok=True)
    
    # Load data and initialize model
    print("Loading FashionMNIST dataset...")
    train_loader, val_loader, _ = get_data_loaders()
    model = VAE().to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    # Training loop
    print(f"Training VAE in {args.mode.upper()} mode")

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_recon_loss': [],
        'val_recon_loss': [],
        'train_kl_loss': [],
        'val_kl_loss': [],
        'beta': []
    }

    for epoch in range(args.epochs):
        # Get beta for this epoch
        beta = get_beta(epoch, args.epochs, args.mode)
        
        # Train and validate
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device, beta)
        val_loss, val_recon, val_kl = evaluate(model, val_loader, device, beta)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_recon_loss'].append(train_recon)
        history['val_recon_loss'].append(val_recon)
        history['train_kl_loss'].append(train_kl)
        history['val_kl_loss'].append(val_kl)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} | β={beta:.2f} | "
              f"Train Loss: {train_loss:.3f} (Recon: {train_recon:.3f}, KL: {train_kl:.3f}) | "
              f"Val Loss: {val_loss:.3f}")
        
    # Save Results
    torch.save(model.state_dict(), f"models/vae_{args.mode}.pth")
    save_history(history, f"outputs/losses/history_{args.mode}.json")
    print("Training Complete.")

if __name__ == '__main__':
    main()