#!/usr/bin/env python3
"""
Aesthetic Predictor V2 - Image Content Quality Scorer
Uses LAION's Aesthetic Predictor V2 model trained on human preference data.

This model understands image content and predicts aesthetic quality (exciting vs boring)
based on actual human ratings from the AVA dataset.

REQUIREMENTS:
  pip install simple-aesthetics-predictor transformers torch torchvision pillow

For HEIC support (optional):
  pip install pillow-heif

USAGE:
  python aesthetic_predictor_v2.py [input_directory] [-o output.csv]
  python aesthetic_predictor_v2.py xi-an/
  python aesthetic_predictor_v2.py bad/ -o bad_content_scores.csv

MODEL INFO:
  - Model: shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE
  - Based on: CLIP ViT-L/14 with aesthetic regression head
  - Training: LAION + SAC + Logos + AVA1 datasets
  - Score Range: 1.0 (boring/meaningless) to 10.0 (exciting/high quality)
"""

import os
import sys
import csv
import math
import argparse
import subprocess
from pathlib import Path

# Check for basic packages first
MISSING_PACKAGES = []

try:
    from PIL import Image
except ImportError:
    MISSING_PACKAGES.append('pillow')

try:
    from tqdm import tqdm
except ImportError:
    MISSING_PACKAGES.append('tqdm')

# Try to enable HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

# Check for ML packages
ML_PACKAGES = ['torch', 'transformers', 'aesthetics_predictor']
ML_MISSING = []
AESTHETICS_MODEL = None

for pkg in ML_PACKAGES:
    try:
        if pkg == 'torch':
            import torch
        elif pkg == 'transformers':
            from transformers import CLIPProcessor, CLIPModel
        elif pkg == 'aesthetics_predictor':
            # Try different import paths
            try:
                from aesthetics_predictor import AestheticsPredictorV2
                AESTHETICS_MODEL = 'v2'
            except ImportError:
                try:
                    from aesthetics_predictor.modeling_v2 import AestheticsPredictorV2Linear as AestheticsPredictorV2
                    AESTHETICS_MODEL = 'v2_linear'
                except ImportError:
                    raise ImportError("Cannot import AestheticsPredictorV2")
    except ImportError as e:
        ML_MISSING.append(pkg)


def install_dependencies(force_ml=False):
    """Install required packages."""
    packages = MISSING_PACKAGES.copy()
    if force_ml or ML_MISSING:
        packages.extend(['simple-aesthetics-predictor', 'transformers', 'torch', 'torchvision'])
    if not HEIC_SUPPORT:
        packages.append('pillow-heif')

    if not packages:
        return True

    print(f"Installing packages: {', '.join(packages)}")
    print("=" * 60)

    pip_commands = [
        [sys.executable, '-m', 'pip', 'install'],
        ['pip3', 'install'],
        ['pip', 'install'],
    ]

    for pip_cmd in pip_commands:
        try:
            cmd = pip_cmd + ['--user'] + packages
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✓ Installation successful!")
                return True
            else:
                if 'externally-managed-environment' in result.stderr:
                    cmd_break = pip_cmd + ['--break-system-packages', '--user'] + packages
                    print(f"Retrying with --break-system-packages...")
                    result2 = subprocess.run(cmd_break, capture_output=True, text=True, timeout=300)
                    if result2.returncode == 0:
                        print("✓ Installation successful!")
                        return True
                print(f"Error: {result.stderr[:200]}")
        except Exception as e:
            continue

    print("\n✗ Installation failed. Try manually:")
    print(f"  pip install {' '.join(packages)}")
    return False


def verify_installation():
    """Verify that packages can be imported after installation."""
    # Add user site-packages to path if needed
    import site
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.append(user_site)

    # Try importing again
    success = True
    global AESTHETICS_MODEL
    for pkg in ['torch', 'transformers', 'aesthetics_predictor']:
        try:
            if pkg == 'torch':
                import torch
            elif pkg == 'transformers':
                from transformers import CLIPProcessor, CLIPModel
            elif pkg == 'aesthetics_predictor':
                # Try different import paths
                try:
                    from aesthetics_predictor import AestheticsPredictorV2
                    AESTHETICS_MODEL = 'v2'
                except ImportError:
                    from aesthetics_predictor.modeling_v2 import AestheticsPredictorV2Linear as AestheticsPredictorV2
                    AESTHETICS_MODEL = 'v2_linear'
        except ImportError as e:
            print(f"  ✗ Failed to import {pkg}: {e}")
            success = False
    return success


# Handle missing basic packages
if MISSING_PACKAGES:
    print(f"Missing basic packages: {', '.join(MISSING_PACKAGES)}")
    if install_dependencies():
        print("\n✓ Basic packages installed.")
        # Verify
        if not verify_installation():
            print("\n⚠ Packages installed but not found. Trying to add to Python path...")
            if not verify_installation():
                print("\nPlease restart your terminal and try again.")
        sys.exit(0)
    else:
        sys.exit(1)

# Handle missing ML packages
if ML_MISSING:
    print(f"Missing ML packages: {', '.join(ML_MISSING)}")
    print("\nThese are large packages that need to be installed:")
    print("  - torch (~2GB): Deep learning framework")
    print("  - transformers (~500MB): HuggingFace models")
    print("  - simple-aesthetics-predictor: LAION aesthetic model")
    print("\nInstall now? (This may take 5-10 minutes)")
    response = input("Install ML dependencies? (y/n): ").lower().strip()
    if response in ('y', 'yes', ''):
        if install_dependencies(force_ml=True):
            print("\n" + "=" * 60)
            print("✓ ML packages installed!")
            print("=" * 60)
            # Verify
            if verify_installation():
                print("\n✓ Packages verified! Please restart the script:")
            else:
                print("\n⚠ Packages installed. Please restart your terminal and run again:")
            print(f"  python3 {sys.argv[0]} {' '.join(sys.argv[1:])}")
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        print("Cannot run without ML packages. Exiting.")
        sys.exit(1)

# Now import the ML packages with fallback
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from aesthetics_predictor import AestheticsPredictorV2
    AESTHETICS_MODEL = 'v2'
except ImportError:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from aesthetics_predictor.modeling_v2 import AestheticsPredictorV2Linear as AestheticsPredictorV2
    AESTHETICS_MODEL = 'v2_linear'

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"

print(f"Using device: {DEVICE}")
print(f"Loading model: {MODEL_ID}...")

try:
    predictor = AestheticsPredictorV2.from_pretrained(MODEL_ID)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    predictor = predictor.to(DEVICE)
    predictor.eval()
    print("✓ Model loaded successfully\n")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    print("Try: pip install --upgrade transformers torch torchvision")
    sys.exit(1)


def score_image(image_path):
    """
    Score a single image for aesthetic quality.

    Returns:
        float: Score from 1.0 (boring/meaningless) to 10.0 (exciting/interesting)
        or None if scoring fails
    """
    try:
        image = Image.open(image_path).convert("RGB")

        # Preprocess
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = predictor(**inputs)

        score = outputs.logits.item()
        return score

    except Exception as e:
        if str(image_path).lower().endswith(('.heic', '.heif')) and not HEIC_SUPPORT:
            print(f"  ✗ {image_path}: HEIC not supported (install: pip install pillow-heif)")
        else:
            print(f"  ✗ {image_path}: {e}")
        return None


def categorize_score(score):
    """
    Categorize score for exciting vs boring detection.
    """
    if score >= 7.5:
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
        print(f"  Skipped: {', '.join([f.name for f in heic_files[:3]])}")
        if len(heic_files) > 3:
            print(f"  ... and {len(heic_files) - 3} more")

        response = input("\nAuto-install pillow-heif now? (y/n): ").lower().strip()
        if response in ('y', 'yes', ''):
            if install_dependencies():
                print("\n✓ Installation complete! Please restart the script.")
                sys.exit(0)
            else:
                print("\n✗ Installation failed. Continuing without HEIC support.")

        # Remove HEIC files from processing list
        image_files = [f for f in all_files if f.suffix.lower() not in heic_extensions]
    else:
        image_files = all_files

    if not image_files:
        print(f"\nNo processable images found in {image_folder}")
        supported = ".jpg, .jpeg, .png, .webp, .bmp" + (", .heic, .heif" if HEIC_SUPPORT else "")
        print(f"Supported formats: {supported}")
        return []

    print(f"\nFound {len(image_files)} images to process in {image_folder}...")
    print(f"Output: {output_csv}\n")

    failed_count = 0
    for img_path in tqdm(image_files, desc="Scoring images"):
        score = score_image(str(img_path))
        if score is None:
            failed_count += 1
            continue

        results.append({
            'filename': img_path.name,
            'filepath': str(img_path),
            'aesthetic_score': round(score, 4),
            'category': categorize_score(score)
        })

    if failed_count > 0:
        print(f"\n⚠ {failed_count} image(s) failed to score")

    if not results:
        print("\nNo images were successfully scored.")
        return []

    # Save to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'filepath', 'aesthetic_score', 'category'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Results saved to {output_csv}")
    print(f"Total images scored: {len(results)}")

    # Print statistics
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

    print(f"\nContent Quality Breakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / len(results)
        emoji = "🔥" if cat == "exciting" else "😐" if cat == "moderate" else "😴"
        print(f"  {emoji} {cat}: {count} ({pct:.1f}%)")

    # Top and bottom images
    print(f"\nTop 3 Most Exciting:")
    top3 = sorted(results, key=lambda x: x['aesthetic_score'], reverse=True)[:3]
    for r in top3:
        print(f"  {r['filename']}: {r['aesthetic_score']:.2f}")

    print(f"\nBottom 3 Most Boring:")
    bottom3 = sorted(results, key=lambda x: x['aesthetic_score'])[:3]
    for r in bottom3:
        print(f"  {r['filename']}: {r['aesthetic_score']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Score images for content quality (exciting vs boring) using LAION Aesthetic Predictor V2'
    )
    parser.add_argument('input_dir', nargs='?', help='Input directory containing images')
    parser.add_argument('-o', '--output', help='Output CSV file path (default: <dir_name>_aesthetic.csv)')
    parser.add_argument('--install', action='store_true', help='Install dependencies and exit')
    args = parser.parse_args()

    if args.install:
        install_dependencies()
        sys.exit(0)

    # Default output directory
    output_dir = Path("/Users/gavinxiang/Downloads/MLDataset")

    # Print status
    print("=" * 60)
    print("LAION Aesthetic Predictor V2")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"HEIC Support: {'✓' if HEIC_SUPPORT else '✗'}")
    if not HEIC_SUPPORT:
        print("  (Install pillow-heif for HEIC support)")
    print("=" * 60)
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
        print("Error: Please specify an input directory")
        print("\nUsage:")
        print("  python aesthetic_predictor_v2.py <directory> [-o output.csv]")
        print("\nExamples:")
        print("  python aesthetic_predictor_v2.py xi-an/")
        print("  python aesthetic_predictor_v2.py bad/ -o bad_scores.csv")
        sys.exit(1)


if __name__ == "__main__":
    main()
