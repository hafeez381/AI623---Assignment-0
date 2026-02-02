# Task 1: Inner Workings of ResNet-152

This task explores the ResNet-152 architecture through transfer learning experiments and skip connection analysis on CIFAR-10.

## Files

| File | Description |
|------|-------------|
| `train_resnet.py` | **Task 1.1**: Baseline training with frozen backbone |
| `train_broken.py` | **Task 1.2**: Training with disabled skip connections in layer4 |
| `train_transfer.py` | **Task 1.4**: Transfer learning with three modes |
| `utils.py` | Training and evaluation helper functions |
| `utils_update.py` | Updated utilities with gradient accumulation |
| `visualization.ipynb` | Comprehensive analysis notebook |

## Experiments

### Task 1.1: Baseline (Frozen Backbone)
```bash
python train_resnet.py
```
Trains ResNet-152 with frozen ImageNet-pretrained backbone, only training the final FC layer for CIFAR-10.

### Task 1.2: Broken Skip Connections
```bash
python train_broken.py
```
Disables skip connections in `layer4` to analyze their importance for feature propagation.

### Task 1.4: Transfer Learning Comparison
```bash
# Random initialization - train full network from scratch
python train_transfer.py --mode random

# Full fine-tuning - train all layers from ImageNet weights  
python train_transfer.py --mode full

# Last block fine-tuning - only train layer4 + FC
python train_transfer.py --mode lastblock
```

## Outputs

- `results/`: Training metrics (JSON)
- `checkpoints/`: Saved model weights
- `figures/`: Generated visualizations

## Analysis

Open `visualization.ipynb` to:
- Plot training curves
- Compare different training strategies
- Visualize activation maps (Task 1.3)
- Analyze feature representations (Task 1.5)
