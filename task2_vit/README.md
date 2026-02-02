# Task 2: Vision Transformer (ViT)

This task explores the Vision Transformer architecture through attention visualization, patch masking experiments, and linear probe training.

## Files

| File | Description |
|------|-------------|
| `vit_utils.py` | Utility functions for model loading, attention extraction, and masking |
| `run_inference.py` | Basic ViT inference and classification |
| `run_masking.py` | Patch masking experiments (random & center) |
| `train_linear_probe.py` | Train linear classifier on frozen ViT features |
| `analysis_vit.ipynb` | Comprehensive analysis notebook |

## Key Functions

### `vit_utils.py`
- `load_model()` - Load pretrained ViT-Base-Patch16-224
- `extract_cls_attention()` - Extract CLS token attention from a layer
- `extract_attention_rollout()` - Compute attention rollout across layers
- `mask_patches()` - Apply random or center masking to images
- `get_patch_embeddings()` - Extract embeddings for linear probe

## Experiments

### Inference
```bash
python run_inference.py
```

### Patch Masking
```bash
python run_masking.py
```
Tests model robustness by masking random or center patches at various ratios.

### Linear Probe Training
```bash
python train_linear_probe.py
```
Trains a linear classifier on frozen ViT features to evaluate representation quality.

## Outputs

- `outputs/`: Saved embeddings and metrics
- `figures/`: Attention maps and visualizations

## Analysis

Open `analysis_vit.ipynb` to:
- Visualize CLS token attention maps
- Compare attention rollout patterns
- Analyze masking experiment results
- Evaluate linear probe performance
