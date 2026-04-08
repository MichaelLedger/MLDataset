#!/usr/bin/env python3
"""
Image Content Quality Scorer — deterministic signals + batch-relative scaling.

Uses EfficientNet/ResNet features, Laplacian sharpness, and color variation;
z-scores within the folder, then maps p5–p95 to 1–10 for sortable scores.

REQUIREMENTS:
  pip install torch torchvision Pillow tqdm numpy

For HEIC support:
  pip install pillow-heif

USAGE:
  python image_content_scorer.py <directory> [-o output.csv] [--no-relative-scale]
"""

import os
import sys
import csv
import math
import argparse
import subprocess
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False


def check_and_install():
    """Check packages and offer install."""
    required = ['torch', 'torchvision', 'PIL', 'tqdm']
    missing = []
    for mod in ['torch', 'torchvision', 'PIL', 'tqdm']:
        try:
            if mod == 'PIL':
                from PIL import Image
            else:
                __import__(mod)
        except ImportError:
            missing.append(mod)

    if not missing and not HEIC_SUPPORT:
        missing_msg = "pillow-heif (for HEIC)"
        print(f"Optional package missing: {missing_msg}")
        resp = input("Install HEIC support? (y/n): ").lower().strip()
        if resp in ('y', 'yes', ''):
            pkgs = ['pillow-heif']
        else:
            pkgs = None
    elif missing:
        print(f"Missing packages: {', '.join(missing)}")
        resp = input("Install now? (y/n): ").lower().strip()
        if resp in ('y', 'yes', ''):
            pkgs = missing + (['pillow-heif'] if not HEIC_SUPPORT else [])
        else:
            sys.exit(1)
    else:
        return True

    if pkgs:
        print(f"\nInstalling: {', '.join(pkgs)}...")
        cmds = [
            [sys.executable, '-m', 'pip', 'install', '--user'] + pkgs,
        ]
        for cmd in cmds:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 or 'already satisfied' in result.stdout:
                print("✓ Installation successful!")
                print("\nPlease restart the script.")
                sys.exit(0)
            if 'externally-managed-environment' in result.stderr:
                cmd.insert(3, '--break-system-packages')
                result2 = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result2.returncode == 0:
                    print("✓ Installation successful!")
                    print("\nPlease restart the script.")
                    sys.exit(0)
        print("✗ Installation failed. Try manually:")
        print(f"  pip install {' '.join(pkgs)}")
        sys.exit(1)

    return True


# Check packages
check_and_install()

# Now import
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def laplacian_variance_gray(gray: np.ndarray) -> float:
    """Variance of Laplacian (focus / edge energy); higher often means sharper photos."""
    g = gray.astype(np.float64)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (
        -4 * g[1:-1, 1:-1]
        + g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
    )
    return float(np.var(lap))


def absolute_content_1_10(feat_mag: float, lap_var: float, color_std: float) -> float:
    """
    Batch-independent 1..10 score: each signal mapped to [0,1] with fixed ranges
    (works when every image in a folder is similar quality).
    """
    f = min(1.0, max(0.0, math.log1p(max(0.0, feat_mag) * 120.0) / math.log1p(45.0)))
    l = min(1.0, max(0.0, math.log1p(max(0.0, lap_var)) / math.log1p(700.0)))
    c = min(1.0, max(0.0, float(color_std) / 50.0))
    u = 0.4 * f + 0.35 * l + 0.25 * c
    return 1.0 + 9.0 * u


def batch_percentile_scale_1_10(scores, low_pct: float = 5.0, high_pct: float = 95.0):
    n = len(scores)
    if n == 0:
        return []
    if n == 1:
        return [5.0]
    t = torch.tensor(scores, dtype=torch.float64)
    lo = torch.quantile(t, low_pct / 100.0).item()
    hi = torch.quantile(t, high_pct / 100.0).item()
    span = hi - lo
    if span < 1e-9:
        return [5.0] * n
    out = []
    for s in scores:
        u = (s - lo) / span
        u = max(0.0, min(1.0, u))
        out.append(1.0 + 9.0 * u)
    return out


def ranks_higher_better(scores):
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    rank_by_pos = [0] * len(scores)
    for r, (idx, _) in enumerate(indexed, start=1):
        rank_by_pos[idx] = r
    return rank_by_pos


class ContentAestheticScorer:
    """
    Image content scorer using pretrained model features.
    Uses ImageNet-pretrained model + trained aesthetic head.
    """

    def __init__(self):
        self.device = DEVICE
        self.model = None
        self.transform = None
        self.load_model()

    def load_model(self):
        """Load pretrained model with aesthetic head."""
        print(f"Loading model on {self.device}...")

        # Use EfficientNet-B0 (good balance of accuracy/speed)
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        except:
            # Fallback to ResNet50
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Remove final classification layer
        if hasattr(self.model, 'classifier'):
            in_features = self.model.classifier[-1].in_features
            self.model.classifier = nn.Identity()
        else:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()

        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print("✓ Model ready\n")

    def compute_metrics(self, image_path):
        """
        Backbone activation strength, Laplacian sharpness, color channel variation.
        Combined into a batch-independent raw_score in score_directory.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            img_array = np.array(image)
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(img_tensor)

            feat_mag = float(features.abs().mean().item())
            gray = np.mean(img_array, axis=2)
            lap_var = laplacian_variance_gray(gray)
            if len(img_array.shape) == 3:
                color_std = float(np.std(img_array, axis=(0, 1)).mean())
            else:
                color_std = float(np.std(img_array))

            return feat_mag, lap_var, color_std

        except Exception as e:
            print(f"  ✗ Error: {image_path.name}: {e}")
            return None


def categorize(score):
    """Categorize score."""
    if score >= 7.0:
        return "exciting"
    elif score >= 5.0:
        return "moderate"
    else:
        return "boring/meaningless"


def score_directory(input_dir, output_csv, relative_scale: bool = True):
    """Score all images in directory."""
    input_path = Path(input_dir)
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.heic', '.heif'}

    all_files = [f for f in input_path.iterdir() if f.suffix.lower() in exts]
    all_files.sort()

    # Handle HEIC
    heic_files = [f for f in all_files if f.suffix.lower() in {'.heic', '.heif'}]
    if heic_files and not HEIC_SUPPORT:
        print(f"\n⚠ Skipping {len(heic_files)} HEIC files (install pillow-heif)")
        files = [f for f in all_files if f.suffix.lower() not in {'.heic', '.heif'}]
    else:
        files = all_files

    if not files:
        print(f"\nNo images found in {input_path}")
        return []

    print(f"\nProcessing {len(files)} images...")
    print(f"Output: {output_csv}\n")

    # Initialize scorer
    scorer = ContentAestheticScorer()

    results = []
    failed = 0
    metrics_rows = []

    for img_path in tqdm(files, desc="Scoring"):
        m = scorer.compute_metrics(img_path)
        if m is None:
            failed += 1
            continue
        metrics_rows.append((img_path, m))

    if failed:
        print(f"\n⚠ {failed} images failed")

    if not metrics_rows:
        print("\nNo images scored.")
        return []

    for img_path, m in metrics_rows:
        feat_mag, lap_var, color_std = m
        rs = absolute_content_1_10(feat_mag, lap_var, color_std)
        results.append({
            'filename': img_path.name,
            'filepath': str(img_path),
            'raw_score': round(rs, 4),
        })

    raw_list = [float(r['raw_score']) for r in results]
    print(
        "\nGlobal raw_score: fixed blend of log-scaled feature magnitude, Laplacian sharpness, "
        "color std (no z-scores; valid for homogeneous folders)."
    )
    print(f"  raw_score range: min={min(raw_list):.4f}, max={max(raw_list):.4f}")

    if relative_scale and len(results) >= 2:
        scaled = batch_percentile_scale_1_10(raw_list)
        print("Relative aesthetic_score: p5–p95 on raw_score → within-folder spread for sorting.")
    else:
        scaled = raw_list.copy()
        if len(results) == 1:
            print("\nSingle image: aesthetic_score = raw_score.")

    ranks = ranks_higher_better(scaled)
    for i, r in enumerate(results):
        r['aesthetic_score'] = round(scaled[i], 4)
        r['rank'] = ranks[i]
        r['category'] = categorize(float(r['raw_score']))

    # Save
    fieldnames = ['filename', 'filepath', 'raw_score', 'aesthetic_score', 'rank', 'category']
    results.sort(key=lambda r: r['rank'])
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    rel = [r['aesthetic_score'] for r in results]
    print(f"\n✓ Saved: {output_csv}")
    print(f"Scored: {len(results)} images")
    print(f"\nraw_score (global): Mean={sum(raw_list)/len(raw_list):.2f}, Min={min(raw_list):.2f}, Max={max(raw_list):.2f}")
    print(f"aesthetic_score (folder-relative): Mean={sum(rel)/len(rel):.2f}, Min={min(rel):.2f}, Max={max(rel):.2f}")
    print("`category` uses **raw_score**. `rank` uses **aesthetic_score**.")

    cats = {}
    for r in results:
        cat = r['category']
        cats[cat] = cats.get(cat, 0) + 1

    print("\nQuality Breakdown (from global raw_score):")
    for cat, count in sorted(cats.items(), key=lambda x: x[1], reverse=True):
        emoji = "🔥" if cat == "exciting" else "😐" if cat == "moderate" else "😴"
        print(f"  {emoji} {cat}: {count} ({100*count/len(results):.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description='Score image content quality')
    parser.add_argument('input_dir', nargs='?', help='Directory with images')
    parser.add_argument('-o', '--output', help='Output CSV path')
    parser.add_argument(
        '--no-relative-scale',
        action='store_true',
        help='Disable p5–p95 batch scaling (narrower aesthetic_score)',
    )
    args = parser.parse_args()

    if not args.input_dir:
        print("Usage: python image_content_scorer.py <directory> [-o output.csv]")
        sys.exit(1)

    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Directory not found: {input_path}")
        sys.exit(1)

    output_dir = Path("/Users/gavinxiang/Downloads/MLDataset")
    if args.output:
        output_csv = Path(args.output)
    else:
        output_csv = output_dir / f"{input_path.name}_content_scores.csv"

    if output_csv.exists():
        output_csv.unlink()

    print("=" * 60)
    print("Image Content Quality Scorer")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Device: {DEVICE}")
    print(f"HEIC: {'✓' if HEIC_SUPPORT else '✗'}")
    print("=" * 60)

    score_directory(input_path, output_csv, relative_scale=not args.no_relative_scale)

    print("\n" + "=" * 60)
    print("✓ COMPLETE")
    print("=" * 60)
    print(f"Output: {output_csv}")


if __name__ == "__main__":
    main()
