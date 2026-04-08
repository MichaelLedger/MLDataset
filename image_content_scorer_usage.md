# `image_content_scorer.py` — usage

Scores images with a **lightweight content / “interestingness” heuristic**: ImageNet-pretrained backbone features, a small trainable-style MLP head (randomly initialized in this script), plus a **color-complexity** blend. Outputs are on a **1–10** scale for comparison with the other scorer, but this is **not** the same as the LAION human-rating model.

## Requirements

```bash
pip install torch torchvision pillow tqdm numpy
```

Optional — **HEIC / HEIF**:

```bash
pip install pillow-heif
```

On first run, if packages are missing, the script may prompt to install them.

## Basic usage

```bash
python image_content_scorer.py <directory>
```

Example:

```bash
python image_content_scorer.py ./my_images
```

## Arguments

| Argument | Description |
|----------|-------------|
| `input_dir` | Directory containing images (required). Only **direct** children are scored (not subfolders). |
| `-o`, `--output` | Path to the output CSV. If omitted, see **Default output path** below. |

There is no `--install` flag; dependency handling is interactive when imports fail.

## Default output path

If `-o` / `--output` is **not** given, the CSV is written to:

`{MLDataset}/{dirname}_content_scores.csv`

`{MLDataset}` is the path hardcoded in the script (currently `/Users/gavinxiang/Downloads/MLDataset`). `{dirname}` is the last segment of the input path.

If that output file already exists, it is **deleted** before writing.

## Supported image formats

`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.heic`, `.heif`

HEIC/HEIF requires `pillow-heif`; otherwise those files are skipped.

## Output CSV

Columns:

| Column | Meaning |
|--------|---------|
| `filename` | Base name of the image |
| `filepath` | Full path to the image |
| `aesthetic_score` | Numeric score (1–10 after blending) |
| `category` | `exciting`, `moderate`, or `boring/meaningless` |

## Score categories

| Category | Condition |
|----------|-----------|
| `exciting` | score ≥ 7.0 |
| `moderate` | 5.0 ≤ score < 7.0 |
| `boring/meaningless` | score < 5.0 |

## How scoring works (implementation summary)

1. **Backbone:** Prefer **EfficientNet-B0** with ImageNet weights (`torchvision`); on failure, **ResNet50** with ImageNet weights.
2. **Head:** A small MLP maps pooled features to one logit; weights are **not** loaded from an external aesthetic checkpoint (`_load_aesthetic_weights` is a no-op).
3. **Post-processing:** Sigmoid maps the logit toward 1–10, then the result is blended with a simple **color diversity / “complexity”** score (70% model-like signal, 30% complexity).

Use this script when you want something **fast and local** without Hugging Face LAION downloads. For **human-calibrated aesthetic** scores, prefer `laion_aesthetic_scorer.py`.

## Device

Uses **CUDA** if available, else **Apple MPS** (if available), else **CPU**.

## Comparison with `laion_aesthetic_scorer.py`

| | `image_content_scorer.py` | `laion_aesthetic_scorer.py` |
|--|---------------------------|------------------------------|
| Large HF download | No | Yes (first run) |
| Human aesthetic training | No | Yes (LAION / AVA–style model when primary load succeeds) |
| `transformers` | Not required | Required |
