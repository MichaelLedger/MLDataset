# SPAQ + UHD Dataset Combination

This document describes the process of combining the SPAQ and UHD image quality assessment datasets into a unified training dataset.

## Dataset Sources

1. **SPAQ Dataset**
   - Original file: `spaq-scores.csv`
   - Total samples: 11,126
   - Score range: [0-100]
   - Features: Image name, MOS (Mean Opinion Score), Brightness, Colorfulness, Contrast, Noisiness, Sharpness

2. **UHD Dataset**
   - Original file: `uhd-scores.csv`
   - Total samples: 6,074
   - Score range: [0-1]
   - Features: Multiple columns including image_name, quality_mos, set, subset, etc.

## Combination Process

1. **Data Processing Steps**
   - SPAQ scores were normalized from [0-100] to [0-1] range by dividing by 100
   - UHD scores were kept as is (already in [0-1] range)
   - Only 'image_name' and 'quality_mos' columns were retained
   - Datasets were concatenated using pandas

2. **Implementation**
   ```python
   import pandas as pd

   # Read the CSV files
   spaq_df = pd.read_csv('spaq-scores.csv')
   uhd_df = pd.read_csv('uhd-scores.csv')

   # Process SPAQ scores
   spaq_processed = pd.DataFrame({
       'image_name': spaq_df['Image name'],
       'quality_mos': spaq_df['MOS'] / 100  # Normalize to [0-1]
   })

   # Process UHD scores - already in [0-1] range
   uhd_processed = pd.DataFrame({
       'image_name': uhd_df['image_name'],
       'quality_mos': uhd_df['quality_mos']
   })

   # Combine both datasets
   combined_df = pd.concat([spaq_processed, uhd_processed], ignore_index=True)

   # Save to train.csv
   combined_df.to_csv('train.csv', index=False)
   ```

## Combined Dataset Statistics

- **Output file**: `train.csv`
- **Total samples**: 17,198
- **Columns**: 
  - `image_name`: Original image filename from respective datasets
  - `quality_mos`: Normalized quality score in range [0-1]
- **Score range**:
  - Minimum: 0.0200
  - Maximum: 0.9600

## Environment Setup

The combination process requires Python with pandas library. A virtual environment was created to manage dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas
```

## File Structure
```
.
├── spaq-scores.csv    # Original SPAQ dataset
├── uhd-scores.csv     # Original UHD dataset
├── train.csv          # Combined output dataset
├── combine_scores.py  # Combination script
└── README.md         # This documentation
```
