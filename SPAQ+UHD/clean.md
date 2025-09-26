# CSV Cleaning Documentation

## Overview

This document describes the process and tools used to clean CSV files by removing rows where corresponding image files are missing from the filesystem.

## Problem Statement

During training, the system encountered `FileNotFoundError` when trying to load images referenced in CSV files but missing from the image directory. This was causing training failures with errors like:

```
FileNotFoundError: [Errno 2] No such file or directory: '/Users/macminiai/VSCODE_Projects/LAR-IQA/dataset/SPAQ+UHD/train_and_validation/3539.jpg'
```

## Solution

Created `clean_csv.py` script to automatically validate image existence and remove invalid entries from CSV files.

## Script Features

- **Automatic validation**: Checks if each image file referenced in the CSV actually exists
- **Backup creation**: Creates `.backup` files before making changes
- **Detailed reporting**: Shows progress and summary statistics
- **Pandas integration**: Uses pandas for efficient CSV handling
- **Error handling**: Graceful handling of missing files and directories

## Usage

### Prerequisites

Make sure you have the required dependencies installed:

```bash
# Activate virtual environment
source venv/bin/activate

# Install pandas if not already installed
pip install pandas
```

### Running the Script

```bash
# From the project root directory
python clean_csv.py
```

### What the Script Does

1. **Reads CSV files** from `dataset/SPAQ+UHD/`
   - `train.csv`
   - `validation.csv`

2. **Validates images** in `dataset/SPAQ+UHD/train_and_validation/`
   - Checks if each image file exists
   - Tracks missing images

3. **Creates backups** with `.backup` extension
   - `train.csv.backup`
   - `validation.csv.backup`

4. **Saves cleaned CSV files** with only valid entries

## Results from Last Run

### SPAQ+UHD Dataset Cleaning

**Date**: September 26, 2025

| File | Original Rows | Cleaned Rows | Removed |
|------|---------------|--------------|---------|
| `train.csv` | 17,193 | 16,296 | 897 |
| `validation.csv` | 17,193 | 16,296 | 897 |

### Sample Missing Images

The following images were among those removed (first 10 shown):
- `4.jpg`
- `23.jpg`
- `27.jpg`
- `32.jpg`
- `47.jpg`
- `48.jpg`
- `51.jpg`
- `59.jpg`
- `60.jpg`
- `61.jpg`

**Total missing images**: 897

## File Structure

```
dataset/SPAQ+UHD/
├── train.csv              # Cleaned training data
├── train.csv.backup       # Original training data backup
├── validation.csv         # Cleaned validation data
├── validation.csv.backup  # Original validation data backup
└── train_and_validation/  # Directory containing image files
    ├── 00001.jpg
    ├── 00002.jpg
    └── ... (16,296 valid images)
```

## CSV Format

The CSV files have the following structure:

```csv
image_name,quality_mos
00001.jpg,0.4886
00002.jpg,0.625
00003.jpg,0.7725
...
```

- **image_name**: Filename of the image (e.g., `00001.jpg`)
- **quality_mos**: Quality score (Mean Opinion Score)

## Benefits

1. **Eliminates FileNotFoundError**: No more training crashes due to missing images
2. **Data integrity**: Ensures CSV data matches available images
3. **Backup safety**: Original data preserved in backup files
4. **Performance**: Faster training without file existence checks during runtime
5. **Reproducibility**: Consistent dataset state across training runs

## Recovery

If you need to restore the original CSV files:

```bash
# Restore from backup
cp dataset/SPAQ+UHD/train.csv.backup dataset/SPAQ+UHD/train.csv
cp dataset/SPAQ+UHD/validation.csv.backup dataset/SPAQ+UHD/validation.csv
```

## Script Customization

The script can be easily modified for other datasets by changing the paths in the `main()` function:

```python
def main():
    # Define paths for your dataset
    base_dir = "/path/to/your/dataset"
    train_csv = os.path.join(base_dir, "train.csv")
    validation_csv = os.path.join(base_dir, "validation.csv")
    image_dir = os.path.join(base_dir, "images")
    # ... rest of the function
```

## Dependencies

- **Python 3.6+**
- **pandas**: For CSV file manipulation
- **os**: For file system operations
- **pathlib**: For path handling

## Error Handling

The script includes robust error handling for:
- Missing CSV files
- Missing image directories
- Empty datasets
- Permission errors
- Invalid file formats

## Performance Notes

- Processing time scales linearly with dataset size
- Memory usage is minimal due to row-by-row processing
- Disk I/O is the primary bottleneck for large datasets

---

*Last updated: September 26, 2025*
*Script version: 1.0*
