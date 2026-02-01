# Task 3: Variational Autoencoder (VAE)

This task implements and trains a Variational Autoencoder on FashionMNIST, investigating posterior collapse and implementing KL annealing as mitigation.

## 📁 Files

| File | Description |
|------|-------------|
| `architecture.py` | VAE model definition (Encoder, Decoder, Reparameterization) |
| `vae_utils.py` | Loss function, sampling, and visualization utilities |
| `train_vae.py` | Training script with baseline and mitigated modes |
| `analysis_vae.ipynb` | Comprehensive analysis notebook |

## 🏗️ Architecture

```
Encoder: Input(784) → Linear(400) → ReLU → [μ(20), σ(20)]
         ↓ Reparameterization: z = μ + σ * ε
Decoder: z(20) → Linear(400) → ReLU → Linear(784) → Sigmoid
```

## 🔬 Training Modes

### Baseline (Standard VAE)
```bash
python train_vae.py --mode baseline --epochs 30
```
Standard VAE training with β=1 for KL divergence.

### Mitigated (KL Annealing)
```bash
python train_vae.py --mode mitigated --epochs 30
```
Implements KL annealing to prevent posterior collapse:
- β linearly increases from 0 to 1 over the first 50% of training
- Allows encoder to learn meaningful features before regularization

## 📊 Outputs

- `models/`: Saved model weights (`vae_baseline.pth`, `vae_mitigated.pth`)
- `outputs/losses/`: Training history JSON files

## 📓 Analysis

Open `analysis_vae.ipynb` to:
- Visualize reconstructions vs. originals
- Generate new samples from the latent space
- Compare latent space distributions (baseline vs. mitigated)
- Investigate posterior collapse indicators
- Plot training curves (reconstruction loss, KL divergence)
