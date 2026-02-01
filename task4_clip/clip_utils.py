"""
clip_utils.py
Helper functions for OpenAI CLIP: Zero-shot classification, Feature Extraction, and Alignment.
"""

import torch
import clip
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Subset
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm

def load_clip_model(device):
    """
    Load the standard ViT-B/32 CLIP model and its preprocessing pipeline.
    """
    print("Loading CLIP model (ViT-B/32)...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def load_stl10_data(preprocess, batch_size=32, split='test', download=True):
    """
    Load STL-10 dataset with CLIP's specific preprocessing (Resizing/Normalization).
    STL-10 images are 96x96, but CLIP expects 224x224.
    """
    print(f"Loading STL-10 ({split} split)...")
    dataset = torchvision.datasets.STL10(
        root='./data', 
        split=split, 
        download=download, 
        transform=preprocess
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, loader

def get_zeroshot_classifier(model, class_names, templates, device):
    """
    Create the zero-shot classifier weights by averaging text embeddings 
    across multiple templates (e.g., 'a photo of a {label}').
    
    Args:
        model: CLIP model
        class_names: List of strings (STL-10 classes)
        templates: List of strings (e.g., ['a photo of a {}'])
        device: Torch device
        
    Returns:
        zeroshot_weights: Tensor of shape (Hidden_Dim, Num_Classes)
    """
    with torch.no_grad():
        zeroshot_weights = []
        
        for classname in tqdm(class_names, desc="Building prompt embeddings"):
            texts = [template.format(classname) for template in templates]
            # Tokenize: (Num_Templates, 77)
            texts = clip.tokenize(texts).to(device) 
            
            # Embed: (Num_Templates, Hidden_Dim)
            class_embeddings = model.encode_text(texts)
            
            # Normalize embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            
            # Average over templates (Ensembling)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            
            zeroshot_weights.append(class_embedding)
            
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    
    return zeroshot_weights

def evaluate_zero_shot(model, loader, class_names, templates, device):
    """
    Run Zero-Shot classification on the dataset.
    
    Args:
        model: CLIP model
        loader: DataLoader for images
        class_names: List of class labels
        templates: List of prompt templates to use
        
    Returns:
        accuracy (float)
    """
    # 1. Build the Text Classifier (The "Weights")
    classifier = get_zeroshot_classifier(model, class_names, templates, device)
    
    # 2. Run Inference
    correct = 0
    total = 0
    
    print(f"Evaluating Zero-Shot with {len(templates)} template(s)...")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Inference"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Encode Images
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Cosine Similarity (Image Features @ Text Features)
            # Logits: (Batch, Hidden) @ (Hidden, Classes) -> (Batch, Classes)
            logits = 100. * image_features @ classifier
            
            # Accuracy
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    accuracy = correct / total
    return accuracy

def extract_features(model, dataset, num_samples, device):
    """
    Extract Image and Text embeddings for a subset of data (for visualization).
    
    Returns:
        image_embeddings: (N, D) numpy array
        text_embeddings: (N, D) numpy array (corresponding ground truth label embeddings)
        labels: (N,) list of class indices
    """
    # Create a small subset
    subset_indices = range(num_samples)
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    
    classes = dataset.classes
    image_embeds = []
    text_embeds = []
    label_list = []
    
    # Standard prompt for visualization
    template = "a photo of a {}"
    
    print(f"Extracting features for {num_samples} samples...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            
            # Image Embeddings
            img_feat = model.encode_image(images)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            image_embeds.append(img_feat.cpu())
            
            # Text Embeddings (Ground Truth)
            # We create a text embedding for the specific label of this image
            batch_texts = [template.format(classes[l]) for l in labels]
            text_tokens = clip.tokenize(batch_texts).to(device)
            
            txt_feat = model.encode_text(text_tokens)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
            text_embeds.append(txt_feat.cpu())
            
            label_list.extend(labels.numpy())
            
    return torch.cat(image_embeds).numpy(), torch.cat(text_embeds).numpy(), np.array(label_list)

def align_embeddings(image_features, text_features):
    """
    Learn the orthogonal rotation matrix R to align Image features to Text features.
    Solves: min || X*R - Y ||_F  (Procrustes Problem)
    
    Args:
        image_features (X): (N, D)
        text_features (Y): (N, D)
        
    Returns:
        R: (D, D) Rotation matrix
        aligned_images: (N, D) X transformed by R
    """
    print("Computing Procrustes Alignment...")
    
    # scipy.linalg.orthogonal_procrustes solves for R s.t. || A*R - B || is minimized
    # It returns R and the scale (which we ignore usually for rotation)
    R, _ = orthogonal_procrustes(image_features, text_features)
    
    # Apply transformation
    aligned_images = image_features @ R
    
    return R, aligned_images