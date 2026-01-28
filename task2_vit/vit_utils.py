"""
Vision Transformer (ViT) Utility Functions

Helper functions for ViT model loading, attention extraction, and patch masking.

"""

import torch
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


def load_model(model_name: str = "google/vit-base-patch16-224"):
    """
    Load a pretrained ViT model and its corresponding image processor.
    """
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name, attn_implementation="eager")
    model.config.output_attentions = True
    model.eval()
    return model, processor


def extract_cls_attention(model, inputs, layer_idx=-1):
    """
    Runs forward pass and extracts the attention weights of the [CLS] token from a specific layer.
    Forward pass -> Extract Layer -1 -> Avg Heads -> Slice CLS -> Reshape 14x14
    """
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions is a tuple of tensors, one per layer
    # Each tensor has shape (batch_size, num_heads, seq_len, seq_len)
    layer_attentions = outputs.attentions[layer_idx]  # (batch, heads, seq, seq) (1, 12, 197, 197)
    
    # Average across all heads
    attn_mean = layer_attentions[0].mean(dim=0)  # (seq, seq) (197, 197)
    
    # Extract CLS token's attention to patch tokens
    # CLS token is at position 0, patches are at positions 1:
    cls_attention_vector = attn_mean[0, 1:]  # (num_patches,) (196,)

    # Reshape to grid
    # For vit-base-patch16-224, patches = 196 -> 14x14
    seq_len = cls_attention_vector.shape[0]
    grid_size = int(np.sqrt(seq_len))
    attention = cls_attention_vector.detach().cpu().numpy()
    attention_grid = attention.reshape(grid_size, grid_size)

    return attention_grid


def extract_attention_rollout(model, inputs, start_layer=0):
    """
    Compute attention rollout - a better method for visualizing attention flow.
    This method accounts for how attention propagates through all layers.
    
    Based on: "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020)
    """
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    if outputs.attentions is None:
        raise ValueError("Model returned None for attentions")
    
    # Start with identity matrix
    num_tokens = outputs.attentions[0].shape[-1]
    rollout = torch.eye(num_tokens).to(outputs.attentions[0].device)
    
    # Roll out attention through layers
    for layer_attention in outputs.attentions[start_layer:]:
        # Average attention across heads
        attention_heads_fused = layer_attention[0].mean(dim=0)
        
        # Add residual connection
        attention_heads_fused = attention_heads_fused + torch.eye(num_tokens).to(attention_heads_fused.device)
        
        # Normalize
        attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
        
        # Matrix multiply to accumulate attention
        rollout = torch.matmul(attention_heads_fused, rollout)
    
    # Extract CLS token attention to patches
    cls_attention = rollout[0, 1:].detach().cpu().numpy()
    
    # Reshape to grid
    grid_size = int(np.sqrt(len(cls_attention)))
    attention_grid = cls_attention.reshape(grid_size, grid_size)
    
    return attention_grid


def mask_patches(image, mode = "random", ratio = 0.25, patch_size = 16):
    """
    Mask a fraction of image patches.
    
    Args:
        image: PIL Image to mask
        mode: Masking strategy - "random" or "center"
        ratio: Fraction of patches to mask (0.0 to 1.0)
        patch_size: Size of each patch in pixels (default 16)
        
    Returns:
        masked_image: PIL Image with patches masked (set to gray)
    """
    
    if ratio <= 0:
        return image
    
    # Convert to numpy array
    arr = np.array(image) # (H, W, 3) for RGB images
    h, w = arr.shape[:2]
    
    # Calculate grid dimensions
    grid_h = h // patch_size
    grid_w = w // patch_size
    num_patches = grid_h * grid_w
    num_mask = int(num_patches * ratio)
    
    if num_mask == 0:
        return image
    
    # Create mask indices based on mode
    mask_indices = []

    if mode == "random":
        # Randomly pick N unique indices
        mask_indices = np.random.choice(num_patches, num_mask, replace=False)
    
    elif mode == "center":
        # Calculate distance of every patch from the center
        center_y = grid_h // 2
        center_x = grid_w // 2

        distances = []
        for i in range(grid_h):
            for j in range(grid_w):
                # Squared Euclidean distance
                dist = (i - center_y) ** 2 + (j - center_x) ** 2

                # Store (distance, linear_index)
                distances.append((dist, i * grid_w + j))
        
        # Sort by distance (closest to center first)
        distances.sort(key=lambda x: x[0])

        # Take the top N closest patches
        mask_indices = [idx for dist, idx in distances[:num_mask]]
    
    else:
        raise ValueError(f"Unknown masking mode: {mode}. Use 'random' or 'center'.")
    
    # Apply mask (Set pixels to Gray/128)
    masked_arr = arr.copy()

    for idx in mask_indices:
        # Convert linear index back to (row, col)
        row = idx // grid_w
        col = idx % grid_w

        # Convert grid coordinates to pixel coordinates
        y = row * patch_size
        x = col * patch_size

        # Mask the square region
        masked_arr[y : y + patch_size, x : x + patch_size, :] = 128
    
    return Image.fromarray(masked_arr)


def get_patch_embeddings(model, processor, image, pooling = "cls"):
    """
    Extract patch embeddings from ViT for linear probe training.
    
    Args:
        model: ViT model (can be ViTForImageClassification or ViTModel)
        processor: ViT image processor
        image: PIL Image
        pooling: "cls" for CLS token, "mean" for mean of patch tokens
        
    Returns:
        embeddings: torch.Tensor of shape (hidden_size,)
    """
    device = model.device
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Handle both ViTModel and ViTForImageClassification
        if hasattr(model, 'vit'):
            outputs = model.vit(**inputs)
        else:
            outputs = model(**inputs)
        
        # outputs.last_hidden_state has shape (batch, seq_len, hidden_size)
        hidden_states = outputs.last_hidden_state  # (1, seq_len, hidden_size) (1, 197, 768)
        
        if pooling == "cls":
            # Use CLS token (index 0)
            embeddings = hidden_states[0, 0, :]  # (hidden_size,)
        elif pooling == "mean":
            # Mean of all patch tokens (excluding CLS token, indices 1 to end are patches)
            embeddings = hidden_states[0, 1:, :].mean(dim=0)  # (hidden_size,)
        else:
            raise ValueError(f"Unknown pooling: {pooling}. Use 'cls' or 'mean'.")
    
    return embeddings