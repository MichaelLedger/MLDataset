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
import math
import time
import argparse
import subprocess
from pathlib import Path
from typing import Any, Callable, List, Optional

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

# Batch-independent calibration: typical shunk031 head outputs ~5.2–5.9 for phone photos.
# Tune with --laion-center / --laion-scale if your corpus sits elsewhere.
DEFAULT_LAION_LOGIT_CENTER = 5.5
DEFAULT_LAION_LOGIT_SCALE = 0.11


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def laion_logit_to_raw_score_1_10(logit: float, center: float, scale: float) -> float:
    """Map model logit to 1..10 without using batch stats (works for homogeneous folders)."""
    if scale <= 0:
        scale = 1e-6
    z = (logit - center) / scale
    return 1.0 + 9.0 * _sigmoid(z)


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


def batch_percentile_scale_1_10(
    scores: List[float], low_pct: float = 5.0, high_pct: float = 95.0
) -> List[float]:
    """
    Stretch scores to 1..10 using the batch's low/high percentiles as anchors.
    Improves sorting when raw model outputs sit in a narrow band (typical for similar photos).
    """
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
    out: List[float] = []
    for s in scores:
        u = (s - lo) / span
        u = max(0.0, min(1.0, u))
        out.append(1.0 + 9.0 * u)
    return out


def ranks_higher_better(scores: List[float]) -> List[int]:
    """1 = best (highest score). Ties get sequential ranks after sorting stable."""
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    rank_by_pos = [0] * len(scores)
    for r, (idx, _) in enumerate(indexed, start=1):
        rank_by_pos[idx] = r
    return rank_by_pos


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


def score_directory(
    input_dir: str,
    output_csv: str,
    relative_scale: bool = True,
    laion_center: float = DEFAULT_LAION_LOGIT_CENTER,
    laion_scale: float = DEFAULT_LAION_LOGIT_SCALE,
) -> List[dict]:
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
            'model_logit': round(score, 6),
        })

    if failed:
        print(f"\n⚠ {failed} failed")

    if not results:
        print("\nNo images scored.")
        return []

    logit_list = [r['model_logit'] for r in results]
    if getattr(scorer, "use_clip_only", False):
        for r in results:
            r['raw_score'] = round(max(1.0, min(10.0, float(r['model_logit']))), 4)
        print("\nGlobal raw_score: CLIP fallback (already ~1–10; no LAION logit calibration).")
    else:
        for r in results:
            r['raw_score'] = round(
                laion_logit_to_raw_score_1_10(r['model_logit'], laion_center, laion_scale), 4
            )
        print(
            f"\nGlobal raw_score: sigmoid map (logit − {laion_center}) / {laion_scale} → 1–10 "
            f"(batch-independent; tune --laion-center / --laion-scale)"
        )
    print(f"  model_logit range: min={min(logit_list):.4f}, max={max(logit_list):.4f}")

    # Within-folder ranking uses model logits (preserves model order; not z-scored)
    if relative_scale and len(results) >= 2:
        scaled = batch_percentile_scale_1_10(logit_list)
        print(
            f"Relative aesthetic_score: p5–p95 on model_logit → 1–10 "
            f"(for sorting in this folder only)"
        )
    else:
        scaled = [float(r['raw_score']) for r in results]
        if len(results) == 1:
            print("\nSingle image: aesthetic_score = raw_score.")

    ranks = ranks_higher_better(scaled)
    for i, r in enumerate(results):
        r['aesthetic_score'] = round(scaled[i], 4)
        r['rank'] = ranks[i]
        r['category'] = categorize(float(r['raw_score']))

    # Save
    output_path = Path(output_csv)
    fieldnames = [
        'filename',
        'filepath',
        'model_logit',
        'raw_score',
        'aesthetic_score',
        'rank',
        'category',
    ]
    results.sort(key=lambda r: r['rank'])
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Stats
    rel = [r['aesthetic_score'] for r in results]
    raw_vals = [float(r['raw_score']) for r in results]
    print(f"\n✓ Saved: {output_path}")
    print(f"Scored: {len(results)}")
    print(f"\nraw_score (global): Mean={sum(raw_vals)/len(raw_vals):.2f}, Min={min(raw_vals):.2f}, Max={max(raw_vals):.2f}")
    print(f"aesthetic_score (folder-relative): Mean={sum(rel)/len(rel):.2f}, Min={min(rel):.2f}, Max={max(rel):.2f}")
    print("`category` uses **raw_score** (global). `rank` uses **aesthetic_score** (relative).")

    # Categories
    cats = {}
    for r in results:
        cat = r['category']
        cats[cat] = cats.get(cat, 0) + 1

    print("\nQuality (from global raw_score):")
    for cat, count in sorted(cats.items(), key=lambda x: x[1], reverse=True):
        emoji = "🔥" if cat == "exciting" else "😐" if cat == "moderate" else "😴"
        print(f"  {emoji} {cat}: {count} ({100*count/len(results):.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description='LAION Aesthetic Scorer V2')
    parser.add_argument('input_dir', nargs='?', help='Directory with images')
    parser.add_argument('-o', '--output', help='Output CSV')
    parser.add_argument('--install', action='store_true', help='Install deps')
    parser.add_argument(
        '--no-relative-scale',
        action='store_true',
        help='Set aesthetic_score = raw_score (no within-folder percentile stretch)',
    )
    parser.add_argument(
        '--laion-center',
        type=float,
        default=DEFAULT_LAION_LOGIT_CENTER,
        metavar='X',
        help=f'Logit value mapped to raw_score 5.5 (default {DEFAULT_LAION_LOGIT_CENTER})',
    )
    parser.add_argument(
        '--laion-scale',
        type=float,
        default=DEFAULT_LAION_LOGIT_SCALE,
        metavar='S',
        help=f'Smaller = steeper raw_score vs logit (default {DEFAULT_LAION_LOGIT_SCALE})',
    )
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

    score_directory(
        input_path,
        output_csv,
        relative_scale=not args.no_relative_scale,
        laion_center=args.laion_center,
        laion_scale=args.laion_scale,
    )

    print("\n" + "=" * 60)
    print("✓ DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
