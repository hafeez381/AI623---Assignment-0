"""
2.5:Linear Probe Training

Compares CLS token representation vs mean pooling of patch tokens

"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from vit_utils import load_model, get_patch_embeddings


OUTPUT_DIR = "outputs/probes"

def extract_features(model, processor, dataset, pooling, device):
    """
    Loops over dataset and extracts fixed features.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    features_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for images, batch_labels in tqdm(loader, desc=f"Extracting {pooling} features"):
            
            # Iterate through the batch
            for img, label in zip(images, batch_labels):
                # Convert tensor to PIL Image
                img_pil = transforms.ToPILImage()(img)
                
                # Get embeddings
                embedding = get_patch_embeddings(model, processor, img_pil, pooling=pooling)
                features_list.append(embedding.cpu().numpy())
                labels_list.append(label.item())
    
    return np.array(features_list), np.array(labels_list)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    model, processor = load_model()
    model.to(device)

    transform = transforms.ToTensor()
    
    # Load Data (CIFAR-10)
    print("Loading CIFAR-10...")
    train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Small subset for speed
    train_subset = Subset(train_data, range(2000))
    test_subset = Subset(test_data, range(500))

    # Compare feature extractors
    img_check, _ = train_subset[0]
    img_check_pil = transforms.ToPILImage()(img_check)
    cls_embedding = get_patch_embeddings(model, processor, img_check_pil, pooling="cls")
    mean_embedding = get_patch_embeddings(model, processor, img_check_pil, pooling="mean")
    diff = torch.norm(cls_embedding - mean_embedding).item()
    print(f"Euclidean distance between CLS and MEAN embeddings: {diff:.4f}")
    print(f"Are features identical? {torch.allclose(cls_embedding, mean_embedding)}")

    results = {}
    
    # Run Experiment Loop
    for method in ['cls', 'mean']:
        print(f"\nMethod: {method.upper()} Pooling")
        
        # Extract Features
        X_train, y_train = extract_features(model, processor, train_subset, method, device)
        X_test, y_test = extract_features(model, processor, test_subset, method, device)
        
        # Train Classifier
        print("Training Linear Classifier...")
        clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        clf.fit(X_train, y_train)
        
        # Evaluate
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        print(f"Results ({method}): Train Acc={train_acc:.2%}, Test Acc={test_acc:.2%}")
        
        results[method] = {
            "train_acc": train_acc,
            "test_acc": test_acc
        }

    # Save Results
    save_path = os.path.join(OUTPUT_DIR, "probe_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison to {save_path}")

if __name__ == "__main__":
    main()