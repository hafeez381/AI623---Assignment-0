import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Standard Variational Autoencoder (VAE) architecture for FashionMNIST.
    
    Structure:
        Encoder: Linear -> ReLU -> Linear (splits into mu and logvar)
        Decoder: Linear -> ReLU -> Linear -> Sigmoid
    """

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        """
        Standard VAE Architecture for FashionMNIST.
        
        Args:
            input_dim (int): Size of input (28x28 = 784 for FashionMNIST)
            hidden_dim (int): Size of hidden layer
            latent_dim (int): Size of the latent space (z)
        """
        super(VAE, self).__init__()

        # Encoder
        # Takes flattened image (Batch, 784) -> Hidden (Batch, 400)
        self.fc1 = nn.Linear(input_dim, hidden_dim) # 
        
        # We output two vectors: Mean (mu) and Log-Variance (logvar)
        # Hidden (Batch, 400) -> (Batch, 20) for mu and logvar
        self.fc21 = nn.Linear(hidden_dim, latent_dim) # mu
        self.fc22 = nn.Linear(hidden_dim, latent_dim) # logvar

        # Decoder
        # Takes latent (Batch, 20) -> Hidden (Batch, 400)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        # Hidden (Batch, 400) -> Reconstructed image (Batch, 784)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        Encodes the input into mu and logvar, the latent distribution parameters.

        Args:
            x: Flattened input image. Shape: (Batch, input_dim)

        Returns:
            mu: Mean of the latent Gaussian. Shape: (Batch, latent_dim)
            logvar: Log variance of the latent Gaussian. Shape: (Batch, latent_dim)
        """
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """
        Performs the Reparameterization Trick: z = mu + std * epsilon
        This allows gradients to flow back through the stochastic sampling node.

        Args:
            mu: Mean vector.
            logvar: Log-variance vector.

        Returns:
            z: Sampled latent vector. Shape: (Batch, latent_dim)
        """
       # Calculate standard deviation from log variance
        std = torch.exp(0.5 * logvar)
        
        # Sample random noise (epsilon) from standard normal distribution
        eps = torch.randn_like(std)
        
        # Transform noise to match our distribution
        return mu + eps * std

    def decode(self, z):
        """
        Decodes the latent vector z back into image space.

        Args:
            z: Latent vector. Shape: (Batch, latent_dim)

        Returns:
            recon_x: Reconstructed image pixels (0 to 1). Shape: (Batch, input_dim)
        """
        h3 = F.relu(self.fc3(z))

        # Sigmoid ensures output is between 0 and 1
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        """
        Performs a full forward pass of the VAE.

        Args:
            x: Input image batch. Shape: (Batch, 1, 28, 28)

        Returns:
            recon_x: Reconstructed image. Shape: (Batch, 784)
            mu: Latent mean.
            logvar: Latent log-variance.
        """
        # Flatten the input: (Batch, 1, 28, 28) -> (Batch, 784)
        x = x.view(-1, 784)
        
        # Encode inputs to distribution parameters
        mu, logvar = self.encode(x)
        
        # Sample z using the reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Decode z back to image
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar