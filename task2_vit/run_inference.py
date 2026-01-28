"""
2.1: ViT Inference Script

Loads images, runs ViT inference, prints predictions, and saves results.
"""

import os
import json
import torch
from PIL import Image
from vit_utils import load_model

IMAGE_FOLDER = "data/images"

def main():
    print("Loading ViT model...")
    model, processor = load_model()

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_files)} images in '{IMAGE_FOLDER}'")

    results = {}

    for filename in image_files:
        print(f"\nProcessing: {filename}")
        image_path = os.path.join(IMAGE_FOLDER, filename)

        # Load and process image
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get Top-1 Prediction
        probs = torch.softmax(outputs.logits, dim=-1)

        top_prob, top_idx = probs.max(dim=-1)
        predicted_label = model.config.id2label[top_idx.item()]

        print(f"Prediction: {predicted_label}")
        print(f"Confidence: {top_prob.item():.4f}")
        
        results[filename] = {
            "top1_label": predicted_label,
            "top1_probability": top_prob.item()
        }

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/predictions.json', "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nInference complete. Results saved to 'outputs/predictions.json'")
    
if __name__ == "__main__":
    main()
    