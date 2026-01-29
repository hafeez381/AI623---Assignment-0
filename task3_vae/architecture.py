import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
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
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # We output two vectors: Mean (mu) and Log-Variance (logvar)
        self.fc21 = nn.Linear(hidden_dim, latent_dim) # mu layer
        self.fc22 = nn.Linear(hidden_dim, latent_dim) # logvar layer

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        Encodes the input into mu and logvar.
        """
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """
        The Reparameterization Trick: z = mu + std * epsilon
        Allows gradients to flow back through the stochastic sampling.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decodes the latent vector z back into image space.
        """
        h3 = F.relu(self.fc3(z))
        # Sigmoid ensures output is between 0 and 1 (matching image pixels)
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        """
        Full forward pass.
        Returns: reconstructed_x, mu, logvar
        """
        # Flatten the image (Batch, 1, 28, 28) -> (Batch, 784)
        x = x.view(-1, 784)
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar