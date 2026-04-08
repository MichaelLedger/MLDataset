#!/usr/bin/env python3
"""
Aesthetic Scoring Script for UHD-IQA Dataset
Uses image quality metrics and statistics that correlate with aesthetic quality.

REQUIREMENTS (install manually):
  pip install torch torchvision Pillow tqdm numpy

For HEIC support (optional):
  pip install pillow-heif

USAGE:
  python aesthetic_scoring.py [input_directory] [-o output.csv]
  python aesthetic_scoring.py --install-heif    # Install HEIC support
  python aesthetic_scoring.py --install-all     # Install all dependencies

EXAMPLES:
  python aesthetic_scoring.py xi-an/
  python aesthetic_scoring.py bad/ -o bad_scores.csv
  python aesthetic_scoring.py              # Legacy: score UHD-IQA training/validation

This approach uses:
- Image sharpness (Laplacian variance)
- Color diversity and vibrancy
- Contrast and brightness balance
- Edge density
- Basic CLIP-like features (if available)

Scores are calibrated to 1-10 scale similar to LAION predictor.
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

# Try to enable HEIC support - REQUIRED for .HEIC/.HEIF files
HEIC_SUPPORT = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False


def install_dependencies(install_heif=False, install_all=False):
    """Install missing dependencies."""
    packages = []

    if install_all:
        packages = ['torch', 'torchvision', 'Pillow', 'tqdm', 'numpy', 'pillow-heif']
    elif install_heif and not HEIC_SUPPORT:
        packages = ['pillow-heif']

    if not packages:
        print("All dependencies are already installed.")
        return True

    print(f"Installing packages: {', '.join(packages)}")
    print("=" * 60)

    # Try different pip commands
    pip_commands = [
        [sys.executable, '-m', 'pip', 'install'],
        ['pip3', 'install'],
        ['pip', 'install'],
    ]

    # Try with --break-system-packages if on system Python
    extra_args = []
    if hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix:
        # In virtual environment
        pass
    else:
        # System Python - might need this flag
        extra_args = ['--user']

    installed = False
    for pip_cmd in pip_commands:
        try:
            cmd = pip_cmd + extra_args + packages
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✓ Installation successful!")
                installed = True
                break
            else:
                # Check for externally-managed-environment error
                if 'externally-managed-environment' in result.stderr:
                    # Try with --break-system-packages
                    cmd_break = pip_cmd + ['--break-system-packages', '--user'] + packages
                    print(f"Retrying with --break-system-packages...")
                    print(f"Running: {' '.join(cmd_break)}")
                    result2 = subprocess.run(cmd_break, capture_output=True, text=True, timeout=300)
                    if result2.returncode == 0:
                        print("✓ Installation successful!")
                        installed = True
                        break
                print(f"Installation failed: {result.stderr[:200]}")
        except Exception as e:
            print(f"Command failed: {e}")
            continue

    if installed:
        print("\nPlease restart the script to use the installed packages.")
        print("=" * 60)
        return True
    else:
        print("\n✗ Installation failed. Try manually:")
        print(f"  pip install {' '.join(packages)}")
        return False

# Check base packages
print("Checking required packages...")
required = ['torch', 'torchvision', 'PIL', 'tqdm', 'numpy']
missing = []
for mod in required:
    try:
        __import__(mod)
        print(f"✓ {mod}")
    except ImportError:
        print(f"✗ {mod}")
        missing.append(mod)

if missing:
    print(f"\nMissing: {' '.join(missing)}")
    print(f"Install: pip install {' '.join(missing)}")
    sys.exit(1)

print("✓ All packages found\n")

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from torchvision import models

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}\n")


class AestheticScorer:
    """Image aesthetic scorer using multiple quality metrics."""

    def __init__(self):
        # Try to load a pretrained model for features
        try:
            self.feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
            self.feature_extractor = self.feature_extractor.to(DEVICE)
            self.feature_extractor.eval()
            self.use_deep_features = True
            print("✓ Loaded ResNet50 for deep features\n")
        except:
            self.use_deep_features = False
            print("⚠ Using basic image statistics only\n")

        # Preprocessing for ResNet
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def calculate_sharpness(self, img_array):
        """Calculate image sharpness using Laplacian variance."""
        import cv2
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()

    def calculate_colorfulness(self, img_array):
        """Calculate color diversity and vibrancy."""
        # Convert to LAB color space
        import cv2
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Standard deviation of color channels
        color_std = np.std(a) + np.std(b)

        # Color range
        color_range = (np.max(a) - np.min(a)) + (np.max(b) - np.min(b))

        return color_std * 0.5 + color_range * 0.5

    def calculate_contrast(self, img_array):
        """Calculate image contrast."""
        import cv2
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        return gray.std()

    def calculate_brightness_balance(self, img_array):
        """Check if image has balanced brightness."""
        import cv2
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        mean_brightness = gray.mean()
        # Prefer images with mid-range brightness (not too dark or too bright)
        balance_score = 1.0 - abs(mean_brightness - 128) / 128
        return balance_score * 100

    def calculate_edge_density(self, img_array):
        """Calculate edge density using Canny."""
        import cv2
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return np.sum(edges > 0) / edges.size * 1000

    def get_deep_features(self, image):
        """Extract deep features using ResNet."""
        if not self.use_deep_features:
            return None

        img_tensor = self.preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            features = self.feature_extractor(img_tensor)

        return features.squeeze()

    def score_image(self, image_path):
        """
        Calculate aesthetic score for an image.

        Returns:
            float: Score from 1.0 (poor) to 10.0 (excellent)
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            img_array = np.array(image)

            # Check if cv2 is available
            try:
                import cv2
                has_cv2 = True
            except ImportError:
                has_cv2 = False

            scores = {}

            if has_cv2:
                # Traditional CV metrics
                scores['sharpness'] = min(100, self.calculate_sharpness(img_array) / 10)
                scores['colorfulness'] = min(100, self.calculate_colorfulness(img_array) / 5)
                scores['contrast'] = min(100, self.calculate_contrast(img_array))
                scores['brightness_balance'] = self.calculate_brightness_balance(img_array)
                scores['edge_density'] = min(100, self.calculate_edge_density(img_array))
            else:
                # Fallback to PIL-based metrics
                scores['sharpness'] = 50
                scores['colorfulness'] = self._calculate_colorfulness_pil(image)
                scores['contrast'] = self._calculate_contrast_pil(image)
                scores['brightness_balance'] = 50
                scores['edge_density'] = 50

            # Deep features (if available)
            if self.use_deep_features:
                features = self.get_deep_features(image)
                if features is not None:
                    # Feature norm correlates with image distinctiveness
                    feature_norm = features.norm().item()
                    scores['feature_distinctiveness'] = min(100, feature_norm / 10)
                else:
                    scores['feature_distinctiveness'] = 50
            else:
                scores['feature_distinctiveness'] = 50

            # Resolution bonus (UHD images get bonus)
            width, height = image.size
            resolution_score = min(100, (width * height) / (1920 * 1080) * 50)
            scores['resolution'] = resolution_score

            # Calculate weighted final score
            weights = {
                'sharpness': 0.25,
                'colorfulness': 0.15,
                'contrast': 0.15,
                'brightness_balance': 0.05,
                'edge_density': 0.10,
                'feature_distinctiveness': 0.20,
                'resolution': 0.10,
            }

            # Normalize scores to 0-100 and apply weights
            raw_score = sum(scores.get(k, 50) * weights.get(k, 0) for k in weights.keys())

            # Recalibrate to spread across 1-10 range
            # Typical UHD-IQA images have good resolution but varying quality
            # Raw scores typically range 30-70, we want to map this to 1-10
            # with better spread

            # Use sigmoid-like transformation to spread scores
            # Center around 50 (typical mid-quality)
            normalized = (raw_score - 35) / 25  # Center at 35, spread factor 25

            # Apply curve to spread scores (power transformation)
            if normalized > 0:
                adjusted = math.pow(normalized, 0.7)  # Compress high end slightly
            else:
                adjusted = -math.pow(abs(normalized), 0.7)

            # Map to 1-10 scale
            final_score = 5.5 + adjusted * 4.5

            # Clamp to valid range
            final_score = max(1.0, min(10.0, final_score))

            return final_score

        except Exception as e:
            if str(image_path).lower().endswith(('.heic', '.heif')) and not HEIC_SUPPORT:
                print(f"  ✗ {image_path}: HEIC not supported (install: pip install pillow-heif)")
            else:
                print(f"  ✗ {image_path}: {e}")
            return None  # Return None to indicate failure - will be filtered out

    def _calculate_colorfulness_pil(self, image):
        """Fallback colorfulness calculation using PIL."""
        pixels = list(image.getdata())
        r_vals = [p[0] for p in pixels]
        g_vals = [p[1] for p in pixels]
        b_vals = [p[2] for p in pixels]

        r_std = np.std(r_vals)
        g_std = np.std(g_vals)
        b_std = np.std(b_vals)

        return min(100, (r_std + g_std + b_std) / 3)

    def _calculate_contrast_pil(self, image):
        """Fallback contrast calculation using PIL."""
        gray = image.convert('L')
        pixels = list(gray.getdata())
        return min(100, np.std(pixels) / 2.5)


# Global scorer instance
scorer = AestheticScorer()


def score_image(image_path):
    """Wrapper for scoring."""
    return scorer.score_image(image_path)


def categorize_score(score):
    """
    Categorize score based on calibrated scale.
    With proper calibration:
      7.0-10.0: High quality / Exciting (top 20-30%)
      5.0-7.0:  Moderate quality (middle 40-50%)
      1.0-5.0:  Lower quality / Boring (bottom 20-30%)
    """
    if score >= 7.0:
        return "exciting"
    elif score >= 5.0:
        return "moderate"
    else:
        return "boring/meaningless"


def score_batch(image_folder, output_csv):
    """Score all images in a folder and save to CSV."""
    results = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.heic', '.heif'}

    all_files = [
        f for f in Path(image_folder).iterdir()
        if f.suffix.lower() in image_extensions
    ]
    all_files.sort()

    # Filter HEIC files if no support available
    heic_extensions = {'.heic', '.heif'}
    heic_files = [f for f in all_files if f.suffix.lower() in heic_extensions]

    if heic_files and not HEIC_SUPPORT:
        print(f"\n⚠ Found {len(heic_files)} HEIC file(s) but pillow-heif not installed")
        print(f"  Skipped files: {', '.join([f.name for f in heic_files[:3]])}")
        if len(heic_files) > 3:
            print(f"  ... and {len(heic_files) - 3} more")

        # Auto-install HEIC support
        response = input(f"\nAuto-install pillow-heif now? (y/n): ").lower().strip()
        if response in ('y', 'yes', ''):
            print("\nInstalling pillow-heif...")
            if install_dependencies(install_heif=True):
                print("\n✓ Installation complete! Please restart the script.")
                sys.exit(0)
            else:
                print("\n✗ Installation failed. Continue without HEIC support.")

        # Remove HEIC files from processing list
        image_files = [f for f in all_files if f.suffix.lower() not in heic_extensions]
    else:
        image_files = all_files

    if not image_files:
        print(f"\nNo processable images found in {image_folder}")
        print(f"Supported formats: .jpg, .jpeg, .png, .webp, .bmp" + (", .heic, .heif" if HEIC_SUPPORT else ""))
        return []

    print(f"\nFound {len(image_files)} images to process in {image_folder}...")

    failed_count = 0
    for img_path in tqdm(image_files, desc="Scoring images"):
        score = score_image(str(img_path))
        if score is None:
            failed_count += 1
            continue  # Skip failed images
        results.append({
            'filename': img_path.name,
            'filepath': str(img_path),
            'aesthetic_score': round(score, 4),
            'category': categorize_score(score)
        })

    if failed_count > 0:
        print(f"\n⚠ {failed_count} image(s) failed to score")

    # Save to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'filepath', 'aesthetic_score', 'category'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_csv}")
    print(f"Total images scored: {len(results)}")

    # Print statistics
    if results:
        scores = [r['aesthetic_score'] for r in results]
        print(f"\nScore Statistics:")
        print(f"  Mean: {sum(scores)/len(scores):.2f}")
        print(f"  Min: {min(scores):.2f}")
        print(f"  Max: {max(scores):.2f}")

        # Category breakdown
        categories = {}
        for r in results:
            cat = r['category']
            categories[cat] = categories.get(cat, 0) + 1
        print(f"\nCategory breakdown:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} ({100*count/len(results):.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description='Score images in a directory for aesthetic quality.')
    parser.add_argument('input_dir', nargs='?', help='Input directory containing images')
    parser.add_argument('-o', '--output', help='Output CSV file path (default: <dir_name>_aesthetic.csv)')
    parser.add_argument('--install-heif', action='store_true', help='Install pillow-heif for HEIC support')
    parser.add_argument('--install-all', action='store_true', help='Install all dependencies')
    args = parser.parse_args()

    # Handle install commands first
    if args.install_all:
        install_dependencies(install_all=True)
        sys.exit(0)

    if args.install_heif:
        install_dependencies(install_heif=True)
        sys.exit(0)

    # Default output directory
    output_dir = Path("/Users/gavinxiang/Downloads/MLDataset")

    # Print HEIC support status and offer auto-install
    print("=" * 60)
    if HEIC_SUPPORT:
        print("✓ HEIC support enabled (pillow-heif)")
        print("=" * 60)
    else:
        print("⚠ HEIC support NOT enabled - .HEIC files will be skipped")
        print("=" * 60)

        # Offer auto-install at startup
        response = input("Auto-install pillow-heif now? (y/n): ").lower().strip()
        if response in ('y', 'yes', ''):
            print("\nInstalling pillow-heif...")
            if install_dependencies(install_heif=True):
                print("\n✓ Installation complete! Please restart the script.")
                sys.exit(0)
            else:
                print("\n✗ Installation failed. Continuing without HEIC support.\n")
        else:
            print("Continuing without HEIC support.\n")
    print()

    if args.input_dir:
        # Single directory mode
        input_path = Path(args.input_dir)
        if not input_path.exists():
            print(f"Error: Directory not found: {input_path}")
            sys.exit(1)

        # Generate output CSV name based on directory name
        dir_name = input_path.name
        if args.output:
            output_csv = Path(args.output)
        else:
            output_csv = output_dir / f"{dir_name}_aesthetic.csv"

        # Clear old CSV if it exists
        if output_csv.exists():
            output_csv.unlink()
            print(f"Cleared old CSV: {output_csv}")

        print("=" * 60)
        print(f"SCORING IMAGES IN: {input_path}")
        print("=" * 60)
        score_batch(input_path, output_csv)

        print("\n" + "=" * 60)
        print("AESTHETIC SCORING COMPLETE")
        print("=" * 60)
        print(f"\nGenerated file: {output_csv}")
    else:
        # Legacy mode: process training and validation directories
        base_dir = Path("/Users/gavinxiang/Downloads/MLDataset/UHD-IQA")
        training_dir = base_dir / "challenge" / "training"
        validation_dir = base_dir / "challenge" / "validation"

        # Clear old CSV files before starting
        train_csv = output_dir / "train_aesthetic.csv"
        val_csv = output_dir / "validation_aesthetic.csv"

        for csv_file in [train_csv, val_csv]:
            if csv_file.exists():
                csv_file.unlink()
                print(f"Cleared old CSV: {csv_file}")

        # Score training images
        print("=" * 60)
        print("SCORING TRAINING IMAGES")
        print("=" * 60)
        if training_dir.exists():
            score_batch(training_dir, train_csv)
        else:
            print(f"Training directory not found: {training_dir}")

        print("\n")

        # Score validation images
        print("=" * 60)
        print("SCORING VALIDATION IMAGES")
        print("=" * 60)
        if validation_dir.exists():
            score_batch(validation_dir, val_csv)
        else:
            print(f"Validation directory not found: {validation_dir}")

        print("\n" + "=" * 60)
        print("AESTHETIC SCORING COMPLETE")
        print("=" * 60)
        print(f"\nGenerated files:")
        print(f"  - {train_csv}")
        print(f"  - {val_csv}")


if __name__ == "__main__":
    main()
