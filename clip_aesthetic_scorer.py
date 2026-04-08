#!/usr/bin/env python3
"""
CLIP-based Aesthetic Scorer - No External Downloads Required
Uses CLIP's zero-shot classification with aesthetic prompts.

REQUIREMENTS:
  pip install transformers torch torchvision pillow tqdm

For HEIC support:
  pip install pillow-heif

USAGE:
  python clip_aesthetic_scorer.py <directory> [-o output.csv]

METHOD:
  Uses CLIP to compare images against aesthetic text prompts.
  Higher similarity to "beautiful/exciting" descriptors = higher score.
  No additional model weights needed - CLIP is sufficient.
"""

import os
import sys
import csv
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
import numpy as np

# HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False


def check_install():
    """Check and install dependencies."""
    required = ['transformers', 'torch', 'torchvision', 'pillow', 'tqdm']
    missing = []

    for pkg in required:
        try:
            if pkg == 'pillow':
                from PIL import Image
            elif pkg == 'tqdm':
                from tqdm import tqdm
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if not missing and not HEIC_SUPPORT:
        print("Optional: pillow-heif (for HEIC support)")
        resp = input("Install HEIC support? (y/n): ").lower().strip()
        if resp in ('y', 'yes', ''):
            missing.append('pillow-heif')

    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        resp = input("Install now? (y/n): ").lower().strip()
        if resp not in ('y', 'yes', ''):
            sys.exit(1)

        print(f"\nInstalling...")
        for cmd in [
            [sys.executable, '-m', 'pip', 'install', '--user'] + missing,
            [sys.executable, '-m', 'pip', 'install', '--break-system-packages', '--user'] + missing,
        ]:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                print("✓ Installed! Please restart the script.")
                sys.exit(0)

        print("✗ Install failed. Try manually:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

    return True


# Check dependencies
check_install()

# Imports
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class CLIPAestheticScorer:
    """
    CLIP-based aesthetic scorer.
    Uses text-image similarity with carefully crafted aesthetic prompts.
    """

    # Aesthetic text prompts - positive qualities
    POSITIVE_PROMPTS = [
        "beautiful photograph",
        "high quality image",
        "professional photography",
        "visually stunning",
        "excellent composition",
        "captivating scene",
        "artistic masterpiece",
        "exciting moment",
        "perfect lighting",
        "aesthetic appeal",
        "masterful photography",
        "engaging content",
    ]

    # Negative qualities
    NEGATIVE_PROMPTS = [
        "low quality image",
        "poor photography",
        "blurry photo",
        "bad composition",
        "uninteresting scene",
        "boring picture",
        "amateur snapshot",
        "poor lighting",
        "dull image",
        "meaningless content",
        "worthless photo",
        "ugly picture",
    ]

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize with CLIP model.
        Using ViT-B/32 for balance of quality and speed.
        """
        self.device = DEVICE
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.positive_features = None
        self.negative_features = None

        self.load_model()
        self._precompute_text_features()

    def load_model(self):
        """Load CLIP model."""
        print(f"Loading CLIP model: {self.model_name}")
        print(f"Device: {self.device}")
        print("(No additional downloads needed)\n")

        try:
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print("✓ CLIP model loaded\n")
        except Exception as e:
            print(f"✗ Failed to load CLIP: {e}")
            print("\nPlease check your internet connection for initial model download.")
            sys.exit(1)

    def _precompute_text_features(self):
        """Precompute text embeddings for prompts."""
        print("Precomputing text features...")

        with torch.no_grad():
            # Positive features
            pos_inputs = self.processor(
                text=self.POSITIVE_PROMPTS,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            self.positive_features = self.model.get_text_features(**pos_inputs)
            self.positive_features = F.normalize(self.positive_features, dim=-1)

            # Negative features
            neg_inputs = self.processor(
                text=self.NEGATIVE_PROMPTS,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            self.negative_features = self.model.get_text_features(**neg_inputs)
            self.negative_features = F.normalize(self.negative_features, dim=-1)

        print("✓ Text features ready\n")

    def score_image(self, image_path: Path) -> Optional[float]:
        """Score single image."""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Process
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                # Get image features
                image_features = self.model.get_image_features(**inputs)
                image_features = F.normalize(image_features, dim=-1)

                # Compare with positive prompts
                pos_sim = (image_features @ self.positive_features.T).squeeze()
                pos_score = pos_sim.mean().item()
                pos_max = pos_sim.max().item()

                # Compare with negative prompts
                neg_sim = (image_features @ self.negative_features.T).squeeze()
                neg_score = neg_sim.mean().item()

                # Calculate aesthetic score
                # Weight: mean similarity + boost for any strong positive match
                raw_score = (pos_score * 0.7 + pos_max * 0.3) - neg_score * 0.5

                # Scale to 1-10 range
                # CLIP similarities typically in range [-1, 1], usually [0.2, 0.4] for decent matches
                # Transform: sigmoid to get [0,1], then scale
                normalized = torch.sigmoid(torch.tensor(raw_score * 5)).item()
                final_score = 1.0 + normalized * 9.0

                # Clamp
                final_score = max(1.0, min(10.0, final_score))

            return float(final_score)

        except Exception as e:
            print(f"  ✗ Error: {image_path.name}: {str(e)[:50]}")
            return None


def categorize(score: float) -> str:
    """Categorize score."""
    if score >= 7.0:
        return "exciting"
    elif score >= 5.0:
        return "moderate"
    else:
        return "boring/meaningless"


def score_directory(input_dir: str, output_csv: str):
    """Score all images in directory."""
    input_path = Path(input_dir)
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.heic', '.heif'}

    all_files = [f for f in input_path.iterdir() if f.suffix.lower() in exts]
    all_files.sort()

    # Filter HEIC if needed
    heic_files = [f for f in all_files if f.suffix.lower() in {'.heic', '.heif'}]
    if heic_files and not HEIC_SUPPORT:
        print(f"⚠ Skipping {len(heic_files)} HEIC files (install pillow-heif)")
        files = [f for f in all_files if f.suffix.lower() not in {'.heic', '.heif'}]
    else:
        files = all_files

    if not files:
        print(f"No images found in {input_path}")
        return []

    print(f"Processing {len(files)} images...")
    print(f"Output: {output_csv}\n")

    # Initialize scorer
    scorer = CLIPAestheticScorer()

    results = []
    failed = 0

    for img_path in tqdm(files, desc="Scoring"):
        score = scorer.score_image(img_path)
        if score is None:
            failed += 1
            continue

        results.append({
            'filename': img_path.name,
            'filepath': str(img_path),
            'aesthetic_score': round(score, 4),
            'category': categorize(score)
        })

    if failed:
        print(f"\n⚠ {failed} images failed")

    if not results:
        print("\nNo images scored.")
        return []

    # Save results
    output_path = Path(output_csv)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'filepath', 'aesthetic_score', 'category'])
        writer.writeheader()
        writer.writerows(results)

    # Statistics
    scores = [r['aesthetic_score'] for r in results]
    print(f"\n✓ Saved: {output_path}")
    print(f"Scored: {len(results)} images")
    print(f"\nScore Statistics:")
    print(f"  Mean: {sum(scores)/len(scores):.2f}")
    print(f"  Min: {min(scores):.2f}")
    print(f"  Max: {max(scores):.2f}")

    # Category breakdown
    cats = {}
    for r in results:
        cat = r['category']
        cats[cat] = cats.get(cat, 0) + 1

    print(f"\nContent Quality Breakdown:")
    for cat, count in sorted(cats.items(), key=lambda x: x[1], reverse=True):
        emoji = "🔥" if cat == "exciting" else "😐" if cat == "moderate" else "😴"
        pct = 100 * count / len(results)
        print(f"  {emoji} {cat}: {count} ({pct:.1f}%)")

    # Show top/bottom
    print(f"\nTop 3 Most Exciting:")
    for r in sorted(results, key=lambda x: x['aesthetic_score'], reverse=True)[:3]:
        print(f"  {r['filename']}: {r['aesthetic_score']:.2f}")

    print(f"\nBottom 3 Most Boring:")
    for r in sorted(results, key=lambda x: x['aesthetic_score'])[:3]:
        print(f"  {r['filename']}: {r['aesthetic_score']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='CLIP-based aesthetic scorer - uses text-image similarity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
METHOD:
  This scorer uses CLIP to compare images against aesthetic text descriptions.
  - Positive prompts: "beautiful photograph", "high quality image", etc.
  - Negative prompts: "low quality image", "boring picture", etc.
  - Score is based on similarity to positive vs negative descriptors.

  No additional model training or downloads required beyond CLIP itself.
        """
    )
    parser.add_argument('input_dir', nargs='?', help='Directory containing images')
    parser.add_argument('-o', '--output', help='Output CSV file path')
    parser.add_argument('--install', action='store_true', help='Install dependencies')
    args = parser.parse_args()

    if args.install:
        check_install()
        sys.exit(0)

    if not args.input_dir:
        print("Usage: python clip_aesthetic_scorer.py <directory> [-o output.csv]")
        print("\nExample:")
        print("  python clip_aesthetic_scorer.py exciting/")
        print("  python clip_aesthetic_scorer.py boring/ -o boring_scores.csv")
        sys.exit(1)

    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Directory not found: {input_path}")
        sys.exit(1)

    # Determine output path
    output_dir = Path("/Users/gavinxiang/Downloads/MLDataset")
    if args.output:
        output_csv = Path(args.output)
    else:
        output_csv = output_dir / f"{input_path.name}_clip_scores.csv"

    # Clear old file
    if output_csv.exists():
        output_csv.unlink()

    # Print header
    print("=" * 60)
    print("CLIP Aesthetic Scorer")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Device: {DEVICE}")
    print(f"HEIC Support: {'✓' if HEIC_SUPPORT else '✗'}")
    print("=" * 60)
    print()

    # Run scoring
    score_directory(input_path, output_csv)

    print("\n" + "=" * 60)
    print("✓ SCORING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
