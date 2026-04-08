# `image_content_scorer.py` — usage

Scores images with a **lightweight, deterministic** heuristic: ImageNet backbone **feature magnitude**, **Laplacian sharpness**, and **color variation** are mapped to **raw_score** with **fixed** ranges (no z-scores), then optionally **percentile-mapped** within the folder for `aesthetic_score`. This is **not** the LAION human-rating model.

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
| `--no-relative-scale` | Skip p5–p95 batch mapping; `aesthetic_score` becomes a clamped shift of the composite z-score (narrower spread). |

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
| `raw_score` | **Global** 1–10: fixed blend of log-scaled backbone magnitude, Laplacian sharpness, and color std (no z-scores, so it still works if every image in the folder looks the same). |
| `aesthetic_score` | **Default:** p5–p95 of `raw_score` within the folder for extra spread when sorting. |
| `rank` | `1` = best in this folder (`aesthetic_score`). |
| `category` | From **global** `raw_score`. |

## Score categories

| Category | Condition |
|----------|-----------|
| `exciting` | score ≥ 7.0 |
| `moderate` | 5.0 ≤ score < 7.0 |
| `boring/meaningless` | score < 5.0 |

## How scoring works (implementation summary)

1. **Backbone:** **EfficientNet-B0** (ImageNet weights) or **ResNet50** fallback; classifier replaced with identity; **global** embedding vectors are used.
2. **Signals (per image):** mean absolute activation, Laplacian variance (grayscale), mean RGB channel std — each mapped to [0, 1] with **fixed** log/linear ranges (not z-scored).
3. **raw_score:** `1 + 9 ×` weighted blend (0.4 / 0.35 / 0.25). **aesthetic_score:** optional second pass, p5–p95 on `raw_score` in this folder.

Use this script when you want something **fast and local** without Hugging Face LAION downloads. For **human-calibrated aesthetic** scores, prefer `laion_aesthetic_scorer.py`.

## Device

Uses **CUDA** if available, else **Apple MPS** (if available), else **CPU**.

## Comparison with `laion_aesthetic_scorer.py`

| | `image_content_scorer.py` | `laion_aesthetic_scorer.py` |
|--|---------------------------|------------------------------|
| Large HF download | No | Yes (first run) |
| Human aesthetic training | No | Yes (LAION / AVA–style model when primary load succeeds) |
| `transformers` | Not required | Required |
