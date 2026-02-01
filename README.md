# AI623 - Deep Vision and Language Models: Assignment 0

This repository contains implementations for Assignment 0 of the Deep Vision-Language Models course (AI623) at LUMS, Pakistan. The assignment explores four fundamental deep learning architectures for computer vision.

## Project Structure

```
AI623 - Assignment-0/
├── task1_resnet/      # ResNet-152
├── task2_vit/         # Vision Transformer
├── task3_vae/         # Variational Autoencoder on FashionMNIST
├── task4_clip/        # CLIP
└── DVLM Assignment-0 Report.pdf
```

## Tasks Overview

### Task 1: Inner Workings of ResNet-152
Explores ResNet-152 architecture through transfer learning experiments and skip connection analysis on CIFAR-10.

### Task 2: Vision Transformer (ViT)
Implements attention visualization, patch masking experiments, and linear probe training using a pretrained ViT model.

### Task 3: Variational Autoencoder (VAE)
Builds and trains a VAE on FashionMNIST, investigating posterior collapse and implementing KL annealing as mitigation.

### Task 4: CLIP Model
Explores zero-shot classification on STL-10 and analyzes the alignment between image and text embeddings.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- transformers (for ViT)
- clip (OpenAI)
- numpy, scikit-learn, matplotlib, tqdm

## Quick Start

Each task folder contains its own README with specific instructions. Navigate to the respective folders for detailed usage.

```bash
# Example: Run ResNet baseline training
cd task1_resnet
python train_resnet.py

# Example: Train VAE
cd task3_vae
python train_vae.py --mode baseline --epochs 30
```

## Results

All training metrics are saved to `results/` or `outputs/` directories within each task folder. Visualizations and analysis are performed in Jupyter notebooks (`analysis_*.ipynb`).
