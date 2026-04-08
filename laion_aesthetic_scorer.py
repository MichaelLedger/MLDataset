#!/usr/bin/env python3
"""
LAION Aesthetic Predictor V2 - Proper Implementation
Downloads and loads the actual pretrained model from HuggingFace.

REQUIREMENTS:
  pip install transformers torch torchvision pillow tqdm

For HEIC support:
  pip install pillow-heif

USAGE:
  python laion_aesthetic_scorer.py <directory> [-o output.csv]

MODEL:
  - Repository: shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE
  - Base: CLIP ViT-L/14
  - Training: LAION + AVA aesthetic ratings
  - Score: 1-10 (human preference scale)
"""

import os
import sys
import csv
import time
import argparse
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

# HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False


def install_deps():
    """Install dependencies."""
    pkgs = ['transformers', 'torch', 'torchvision', 'pillow', 'tqdm', 'huggingface_hub']
    if not HEIC_SUPPORT:
        pkgs.append('pillow-heif')

    print(f"Installing: {', '.join(pkgs)}...")
    cmds = [
        [sys.executable, '-m', 'pip', 'install', '--user'] + pkgs,
        [sys.executable, '-m', 'pip', 'install', '--break-system-packages', '--user'] + pkgs,
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 or 'already satisfied' in result.stdout:
            print("✓ Installed! Please restart.")
            sys.exit(0)
    print("✗ Failed. Try: pip install " + ' '.join(pkgs))
    sys.exit(1)


# Check deps
try:
    import torch
    from transformers import AutoModel, CLIPModel, CLIPProcessor
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies.\n")
    resp = input("Install? (y/n): ").lower().strip()
    if resp in ('y', 'yes', ''):
        install_deps()
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _is_transient_hub_error(err: Exception) -> bool:
    msg = str(err).lower()
    if isinstance(err, (ConnectionError, TimeoutError, OSError)):
        return True
    return any(
        s in msg
        for s in ("connection reset", "connection broken", "broken pipe", "timed out", "timeout", "errno 54")
    )


def hf_retry(fn: Callable[[], Any], desc: str, retries: int = 5, base_delay: float = 4.0) -> Any:
    """Retry Hugging Face downloads on flaky connections (resume is automatic on cache)."""
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt >= retries - 1 or not _is_transient_hub_error(e):
                raise
            wait = base_delay * (2**attempt)
            print(f"  {desc} failed ({e}); retry {attempt + 2}/{retries} in {wait:.0f}s...")
            time.sleep(wait)
    assert last is not None
    raise last


class AestheticScorer:
    """Main scorer class."""

    def __init__(self, model_id: str = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"):
        self.device = DEVICE
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.load_model()

    def load_model(self):
        """Load model from HuggingFace (official AestheticsPredictorV2Linear from repo)."""
        print(f"Loading model: {self.model_id}")
        print(f"Device: {self.device}")
        print("Downloading... (first time only, ~2GB; interrupted downloads resume from cache)")
        print()

        try:
            self.processor = hf_retry(
                lambda: CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"),
                "CLIP processor",
            )
            print("✓ CLIP processor loaded")

            # Remote modeling code matches HF weights (CLIPVisionModelWithProjection + head)
            self.model = hf_retry(
                lambda: AutoModel.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                ).to(self.device),
                "Aesthetic model",
            )
            self.model.eval()
            self.use_clip_only = False
            print("✓ Pretrained aesthetic model loaded")
            print(f"\n✓ Model ready\n")

        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print(f"\nTrying alternative model...")
            self._load_alternative()

    def _load_alternative(self):
        """Load alternative model if primary fails."""
        try:
            self.processor = hf_retry(
                lambda: CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"),
                "CLIP processor (fallback)",
            )
            self.model = hf_retry(
                lambda: CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device),
                "CLIP base (fallback)",
            )
            self.model.eval()
            self.use_clip_only = True
            print("✓ Using CLIP-only mode (text-image similarity)")

        except Exception as e:
            print(f"✗ Alternative also failed: {e}")
            sys.exit(1)

    def score_image(self, image_path: Path) -> Optional[float]:
        """Score single image."""
        try:
            image = Image.open(image_path).convert("RGB")

            # Preprocess (do not pass padding= for images — it is tokenizer-only and triggers warnings)
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            with torch.no_grad():
                if getattr(self, "use_clip_only", False):
                    text_inputs = self.processor(
                        text=["high quality photo", "low quality photo"],
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device)

                    image_features = self.model.get_image_features(pixel_values=pixel_values)
                    text_features = self.model.get_text_features(**text_inputs)

                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    similarity = (image_features @ text_features.T).squeeze()
                    high_quality_sim = similarity[0].item()
                    low_quality_sim = similarity[1].item()

                    score = 1.0 + (high_quality_sim - low_quality_sim + 1) / 2 * 9.0

                else:
                    outputs = self.model(pixel_values=pixel_values)
                    score = outputs.logits.squeeze().item()
                    score = max(1.0, min(10.0, float(score)))

            return float(score)

        except Exception as e:
            print(f"  ✗ Error scoring {image_path.name}: {e}")
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
    """Score directory of images."""
    input_path = Path(input_dir)
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.heic', '.heif'}

    all_files = [f for f in input_path.iterdir() if f.suffix.lower() in exts]
    all_files.sort()

    # Handle HEIC
    heic_files = [f for f in all_files if f.suffix.lower() in {'.heic', '.heif'}]
    if heic_files and not HEIC_SUPPORT:
        print(f"⚠ Skipping {len(heic_files)} HEIC files (install pillow-heif)")
        files = [f for f in all_files if f.suffix.lower() not in {'.heic', '.heif'}]
    else:
        files = all_files

    if not files:
        print(f"No images found in {input_path}")
        return []

    print(f"Processing {len(files)} images...\n")

    # Initialize scorer
    scorer = AestheticScorer()

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
        print(f"\n⚠ {failed} failed")

    if not results:
        print("\nNo images scored.")
        return []

    # Save
    output_path = Path(output_csv)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'filepath', 'aesthetic_score', 'category'])
        writer.writeheader()
        writer.writerows(results)

    # Stats
    scores = [r['aesthetic_score'] for r in results]
    print(f"\n✓ Saved: {output_path}")
    print(f"Scored: {len(results)}")
    print(f"\nScores: Mean={sum(scores)/len(scores):.2f}, Min={min(scores):.2f}, Max={max(scores):.2f}")

    # Categories
    cats = {}
    for r in results:
        cat = r['category']
        cats[cat] = cats.get(cat, 0) + 1

    print("\nQuality:")
    for cat, count in sorted(cats.items(), key=lambda x: x[1], reverse=True):
        emoji = "🔥" if cat == "exciting" else "😐" if cat == "moderate" else "😴"
        print(f"  {emoji} {cat}: {count} ({100*count/len(results):.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description='LAION Aesthetic Scorer V2')
    parser.add_argument('input_dir', nargs='?', help='Directory with images')
    parser.add_argument('-o', '--output', help='Output CSV')
    parser.add_argument('--install', action='store_true', help='Install deps')
    args = parser.parse_args()

    if args.install:
        install_deps()
        return

    if not args.input_dir:
        print("Usage: python laion_aesthetic_scorer.py <directory>")
        sys.exit(1)

    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    output_dir = Path("/Users/gavinxiang/Downloads/MLDataset")
    if args.output:
        output_csv = Path(args.output)
    else:
        output_csv = output_dir / f"{input_path.name}_laion_scores.csv"

    if output_csv.exists():
        output_csv.unlink()

    print("=" * 60)
    print("LAION Aesthetic Predictor V2")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Device: {DEVICE}")
    print(f"HEIC: {'✓' if HEIC_SUPPORT else '✗'}")
    print("=" * 60)
    print()

    score_directory(input_path, output_csv)

    print("\n" + "=" * 60)
    print("✓ DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
