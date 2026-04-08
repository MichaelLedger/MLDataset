# `laion_aesthetic_scorer.py` — usage

Scores images with the **LAION Aesthetic Predictor V2** model from Hugging Face (human-preference style scores on a 1–10 scale).

## Requirements

Install into your environment (venv recommended):

```bash
pip install transformers torch torchvision pillow tqdm huggingface_hub
```

Optional — **HEIC / HEIF** images:

```bash
pip install pillow-heif
```

The script can prompt to install missing packages interactively, or you can run:

```bash
python laion_aesthetic_scorer.py --install
```

## Basic usage

```bash
python laion_aesthetic_scorer.py <directory>
```

Example:

```bash
python laion_aesthetic_scorer.py ./my_images
```

## Arguments

| Argument | Description |
|----------|-------------|
| `input_dir` | Directory containing images (required). Only **direct** children of this folder are scored (not subfolders). |
| `-o`, `--output` | Path to the output CSV. If omitted, see **Default output path** below. |
| `--install` | Install listed dependencies via `pip` and exit. |

## Default output path

If `-o` / `--output` is **not** given, the CSV is written to:

`{MLDataset}/{dirname}_laion_scores.csv`

`{MLDataset}` is the path hardcoded in the script (currently `/Users/gavinxiang/Downloads/MLDataset`). `{dirname}` is the **last segment** of the input path (e.g. `exciting` → `exciting_laion_scores.csv`).

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
| `aesthetic_score` | Numeric score (1–10 scale after clamping in normal mode) |
| `category` | `exciting`, `moderate`, or `boring/meaningless` (see below) |

## Score categories

| Category | Condition |
|----------|-----------|
| `exciting` | score ≥ 7.0 |
| `moderate` | 5.0 ≤ score < 7.0 |
| `boring/meaningless` | score < 5.0 |

## Model behavior

- **Primary:** `shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE` loaded with `transformers` + `trust_remote_code=True` (CLIP ViT-L/14–based predictor).
- **First run:** Downloads model weights (on the order of ~2 GB total with dependencies). Interrupted downloads typically **resume** from the Hugging Face cache.
- **Retries:** Transient connection errors (e.g. connection reset) are retried with backoff.
- **Fallback:** If the primary model cannot be loaded, the script falls back to **CLIP-only** scoring (text–image similarity between “high quality photo” and “low quality photo”). That mode is a rough proxy, not the full LAION predictor.

## Device

Uses **CUDA** if available, else **Apple MPS** (if available), else **CPU**.

## See also

- Repository: [shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE](https://huggingface.co/shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE) on Hugging Face.
