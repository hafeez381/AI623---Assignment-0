"""
2.4: Robustness to Patch Masking

- Evaluates ViT robustness against "random" and "center" masking.
- Runs multiple trials for random masking to ensure statistical stability.
- Saves quantitative metrics to JSON.
- Saves qualitative example images

"""

import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from vit_utils import load_model, mask_patches

# Configuration
IMAGE_FOLDER = "data/images"
OUTPUT_DIR = "outputs/masking"
RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MODES = ["random", "center"]
NUM_RANDOM_TRIALS = 5  # Run random masking 5 times to average out variance

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model, processor = load_model()
    model.to(device)
    labels_map = model.config.id2label

    # load images
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', 'jpeg', '.png'))]
    print(f"Found {len(image_files)} images.")

    # get baseline predictions (ground truth)
    baselines = {}
    print("\nEstablishing Baselines")

    for filename in image_files:
        path = os.path.join(IMAGE_FOLDER, filename)
        image = Image.open(path).convert("RGB")
        # Resize to 224x224 to ensure patch alignment matches the model's grid
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred_idx = outputs.logits.argmax(-1).item()
            conf = torch.softmax(outputs.logits, dim=-1)[0, pred_idx].item()
        
        baselines[filename] = pred_idx
        print(f"{filename}: {labels_map[pred_idx]} ({conf:.2%})")

    # run masking experiment
    print("\nRunning Masking Experiments")

    # Structure to hold final accuracies: metrics[mode][ratio] = accuracy
    metrics = {mode: {} for mode in MODES}

    for mode in MODES:
        print(f"Processing Mode: {mode}")

        for ratio in tqdm(RATIOS, desc=f"  {mode} ratios"):
            correct_count = 0
            total_count = 0

            # Determine trials (1 for deterministic 'center', 5 for 'random')
            trials = NUM_RANDOM_TRIALS if mode == "random" else 1
            
            for _ in range(trials):
                for filename in image_files:
                    path = os.path.join(IMAGE_FOLDER, filename)
                    image = Image.open(path).convert("RGB")
                    
                    # Apply Mask
                    masked_img = mask_patches(image, mode=mode, ratio=ratio)
                    
                    # SAVE VISUALIZATION (Only for the first trial, first image, specific ratios)
                    # We save 0.25, 0.5, 0.75 for the report figures
                    if filename == image_files[0] and ratio in [0.25, 0.5, 0.75] and total_count == 0:
                        save_name = f"example_{mode}_{ratio}.png"
                        masked_img.save(os.path.join(OUTPUT_DIR, save_name))

                    # Inference
                    inputs = processor(images=masked_img, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    
                    pred_idx = logits.argmax(-1).item()
                    
                    # Check consistency
                    if pred_idx == baselines[filename]:
                        correct_count += 1
                    total_count += 1
            
            # Calculate accuracy for this setting
            accuracy = correct_count / total_count
            metrics[mode][str(ratio)] = accuracy
            print(f"Mode: {mode}, Ratio: {ratio} -> Consistency: {accuracy:.0%}")

    # Save Results
    results = {
        "config": {
            "ratios": RATIOS,
            "modes": MODES,
            "random_trials": NUM_RANDOM_TRIALS
        },
        "baselines": {
            k: labels_map[v] for k, v in baselines.items()
        },
        "accuracy_results": metrics
    }
    save_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Robustness Summary (Accuracy)")
    print(f"{'Ratio':<10} | {'Random':<10} | {'Center':<10}")

    for ratio in RATIOS:
        r_str = str(ratio)
        rand_acc = metrics["random"].get(r_str, 0)
        cent_acc = metrics["center"].get(r_str, 0)
        print(f"{ratio:<10} | {rand_acc:.2%}     | {cent_acc:.2%}")
    
    print(f"\nMetrics saved to: {save_path}")
    print(f"Example images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()