# LAION Aesthetic Predictor V2 - Installation & Scoring Guide

## Quick Start

### 1. Installation

```bash
# Install the wrapper library
pip install simple-aesthetics-predictor

# Dependencies that will be installed automatically:
# - torch (>=1.0)
# - torchvision (>=0.2.1)
# - transformers (>=4.30.2)
# - Pillow
```

### 2. Single Image Scoring

```python
import torch
from PIL import Image
from transformers import CLIPProcessor
from aesthetics_predictor import AestheticsPredictorV2

# Configuration
MODEL_ID = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
print("Loading model...")
predictor = AestheticsPredictorV2.from_pretrained(MODEL_ID)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
predictor = predictor.to(DEVICE)
predictor.eval()

def score_image(image_path):
    """
    Score a single image for aesthetic quality.
    
    Returns:
        float: Score from 1.0 (boring/meaningless) to 10.0 (exciting/interesting)
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = predictor(**inputs)
    
    score = outputs.logits.item()
    return score


# Example usage
if __name__ == "__main__":
    score = score_image("your_image.jpg")
    print(f"Aesthetic Score: {score:.2f}/10")
    
    # Interpretation
    if score >= 7:
        print("→ High quality / Exciting content")
    elif score >= 5:
        print("→ Moderate quality")
    else:
        print("→ Low quality / Boring or meaningless content")
```

### 3. Batch Processing Multiple Images

```python
import os
from pathlib import Path

def score_batch(image_folder, output_csv="scores.csv"):
    """
    Score all images in a folder and save results to CSV.
    """
    results = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    # Get all image files
    image_files = [
        f for f in Path(image_folder).iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    print(f"Found {len(image_files)} images to process...")
    
    for img_path in image_files:
        try:
            score = score_image(str(img_path))
            results.append({
                'filename': img_path.name,
                'score': round(score, 2),
                'category': categorize_score(score)
            })
            print(f"✓ {img_path.name}: {score:.2f}")
        except Exception as e:
            print(f"✗ {img_path.name}: Error - {e}")
    
    # Save to CSV
    import csv
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'score', 'category'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {output_csv}")
    return results

def categorize_score(score):
    """
    Categorize score for your use case: exciting vs boring.
    """
    if score >= 7.5:
        return "exciting"
    elif score >= 5.0:
        return "moderate"
    else:
        return "boring/meaningless"

# Usage
# score_batch("/path/to/images", "aesthetic_scores.csv")
```

### 4. Advanced: Custom Thresholds for Your Use Case

```python
def quick_score(image_path, threshold_high=6.5, threshold_low=4.0):
    """
    Quick binary/tri-state scoring for content filtering.
    
    Returns:
        dict: {
            'score': float (1-10),
            'is_interesting': bool,
            'is_boring': bool,
            'decision': str ('keep', 'review', 'discard')
        }
    """
    score = score_image(image_path)
    
    return {
        'score': round(score, 2),
        'is_interesting': score >= threshold_high,
        'is_boring': score <= threshold_low,
        'decision': (
            'keep' if score >= threshold_high 
            else 'discard' if score <= threshold_low 
            else 'review'
        )
    }

# Example
result = quick_score("photo.jpg", threshold_high=7.0, threshold_low=3.5)
print(f"Decision: {result['decision']} (score: {result['score']})")
```

## Score Interpretation

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| 8.0 - 10.0 | Exceptional, highly engaging | High priority content |
| 6.5 - 7.9 | Good quality, interesting | Keep |
| 5.0 - 6.4 | Average, moderate interest | Review if needed |
| 3.0 - 4.9 | Below average, boring | Likely discard |
| 1.0 - 2.9 | Poor quality, meaningless | Discard |

## Performance Notes

- **Inference speed**: ~50-100 images/second on GPU (CUDA), ~5-10 on CPU
- **Model size**: ~300MB (CLIP ViT-L/14 + linear layer)
- **Memory**: Requires ~2GB GPU RAM for batch processing

## Hardware Recommendations

| Setup | Speed | Good For |
|-------|-------|----------|
| Apple Silicon (MPS) | Medium | Mac users, moderate batches |
| NVIDIA GPU (CUDA) | Fast | Large-scale processing |
| CPU only | Slow | Single images, small batches |

## Troubleshooting

**Model download fails:**
```python
# Try manual download first
from huggingface_hub import snapshot_download
snapshot_download(repo_id="shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE")
```

**Out of memory (GPU):**
- Reduce batch size
- Use `DEVICE = "cpu"` for small batches
- Enable mixed precision: `predictor = predictor.half().to(DEVICE)`

**Import errors:**
```bash
pip install --upgrade transformers torch torchvision
```
