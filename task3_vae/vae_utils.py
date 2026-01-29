"""
VAE Utility Functions
Helper functions for loss computation, sampling, and visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Compute the VAE loss (Evidence Lower Bound - ELBO): 
    Loss = Reconstruction Loss + (β * KL Divergence)
    
    Args:
        recon_x: Reconstructed images. Shape: (Batch, 784) or (Batch, 1, 28, 28)
        x: Original images Shape: (Batch, 784) or (Batch, 1, 28, 28)
        mu: Mean of latent distribution. Shape: (Batch, latent_dim)
        logvar: Log variance of latent distribution. Shape: (Batch, latent_dim)
        beta: Weight for KL divergence term (for KL annealing, latent space regularization)
        
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss only
        kl_loss: KL divergence only
    """
    # Reconstruction loss: MSE
    # We use reduction='sum' to sum errors over all pixels and batch items
    # x.view(-1, 784) ensures input is flattened to match reconstruction
    recon_loss = F.mse_loss(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL Divergence
    # Analytical solution for KL(N(mu, sigma) || N(0, 1))
    # -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def sample_latents(batch_size, latent_dim, dist='normal', device='mps'):
    """
    Samples random vectors from a prior distribution (Gaussian or Laplacian).
    
    Args:
        batch_size: Number of samples to generate
        latent_dim: Dimension of the latent space
        dist: 'normal' (Gaussian) or 'laplace' (Laplacian)
        device: Device to create tensor on ('cpu' or 'cuda'/'mps')
        
    Returns:
        z: Sampled latent vectors (batch_size, latent_dim)
    """
    if dist == 'normal':
        # Standard normal distribution N(0, I)
        z = torch.randn(batch_size, latent_dim, device=device)
    elif dist == 'laplace':
        # Laplacian distribution with mean=0, scale=1
        # Using numpy for Laplace sampling, then converting to tensor
        z_np = np.random.laplace(loc=0, scale=1, size=(batch_size, latent_dim))
        z = torch.tensor(z_np, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unknown distribution: {dist}. Use 'normal' or 'laplace'.")
    
    return z


def save_history(history, save_path):
    """
    Saves the training history dictionary to a JSON file.
    
    Args:
        history (dict): Dictionary containing lists of loss values.
        save_path (str): File path to save the JSON.
    """
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {save_path}")


def plot_reconstructions(model, dataloader, device, save_path=None, num_images=10):
    """
    Visualize original images vs. their VAE reconstructions
    """
    model.eval()
    
    # Get a batch of images
    images, _ = next(iter(dataloader))
    images = images[:num_images].to(device)
    
    with torch.no_grad():
        recon_images, _, _ = model(images)
    
    # Reshape to (Batch, 28, 28) for plotting
    images = images.cpu().view(-1, 28, 28)
    recon_images = recon_images.cpu().view(-1, 28, 28)
    
    # Create figure
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.5, 3))
    
    for i in range(num_images):
        # top row: original
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        # bottom row: reconstructed
        axes[1, i].imshow(recon_images[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reconstructions to {save_path}")
    
    plt.show()
    plt.close()


def plot_generations(model, latent_dim, num_samples, device, save_path=None, dist='normal'):
    """
    Generate and visualize new images by sampling from the latent prior
    """
    model.eval()
    
    # Sample from prior
    z = sample_latents(num_samples, latent_dim, dist=dist, device=device)
    
    with torch.no_grad():
        generated = model.decode(z)
    
    # Reshape for plotting
    generated = generated.cpu().view(-1, 28, 28)
    
    # create grid plot
    nrows = int(np.ceil(np.sqrt(num_samples)))
    ncols = int(np.ceil(num_samples / nrows))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(generated[i], cmap='gray')
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    dist_name = 'Gaussian' if dist == 'normal' else 'Laplacian'
    plt.suptitle(f'Generated Samples ({dist_name} Prior)', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved generations to {save_path}")
    
    plt.show()
    plt.close()


def plot_training_curves(history, save_path=None, title_suffix=''):
    """
    Plot training and validation loss curves
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Helper to plot on a specific axis
    def plot_metric(ax, train_key, val_key, title):
        ax.plot(epochs, history[train_key], 'b-', label='Train')
        ax.plot(epochs, history[val_key], 'r-', label='Validation')
        ax.set_title(title + f" {title_suffix}")
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plot_metric(axes[0], 'train_loss', 'val_loss', 'Total Loss')
    plot_metric(axes[1], 'train_recon_loss', 'val_recon_loss', 'Reconstruction Loss')
    plot_metric(axes[2], 'train_kl_loss', 'val_kl_loss', 'KL Divergence')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved curves to {save_path}")
    plt.show()
    plt.close()