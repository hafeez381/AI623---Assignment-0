# Task 4: CLIP Model

This task explores OpenAI's CLIP model for zero-shot classification and analyzes the alignment between image and text embeddings.

## Files

| File | Description |
|------|-------------|
| `clip_utils.py` | Utility functions for CLIP: zero-shot classification, feature extraction, alignment |
| `analysis_clip.ipynb` | Comprehensive analysis notebook |

## Key Functions

### `clip_utils.py`
- `load_clip_model()` - Load CLIP ViT-B/32 model
- `load_stl10_data()` - Load STL-10 dataset with CLIP preprocessing
- `get_zeroshot_classifier()` - Build text embeddings from prompt templates
- `evaluate_zero_shot()` - Run zero-shot classification
- `extract_features()` - Extract image and text embeddings
- `align_embeddings()` - Procrustes alignment between modalities

## Experiments

All experiments are conducted in the analysis notebook:

### Zero-Shot Classification
- Evaluate CLIP on STL-10 without any training
- Compare single vs. ensemble prompt templates

### Prompt Engineering
Test different templates:
- `"a photo of a {}"`
- `"a picture of a {}"`
- `"an image of a {}"`
- Ensemble of multiple templates

### Feature Alignment
- Extract image and text embeddings
- Visualize in 2D using t-SNE
- Apply Procrustes alignment to study modality gap

## Outputs

- `outputs/`: Saved embeddings and alignment results

## Analysis

Open `analysis_clip.ipynb` to:
- Evaluate zero-shot accuracy on STL-10
- Compare prompt template effectiveness
- Visualize image-text embedding alignment
- Explore the modality gap phenomenon
- Apply orthogonal Procrustes alignment
