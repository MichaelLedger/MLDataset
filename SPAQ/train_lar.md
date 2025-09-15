# SPAQ Dataset Split Process

This document describes the process of splitting the SPAQ dataset into training and validation sets following the UHD-IQA format.

## Dataset Overview
- Original dataset: SPAQ (Smartphone Photography Attribute and Quality)
- Total images: 11,125
- Source file: `/SPAQ/Annotations/MOS and Image attribute scores.csv`
- Target format: Following UHD-IQA dataset format (image_name,quality_mos)

## Split Details
- Training set: First 5,563 images (50%)
- Validation set: Remaining 5,562 images (50%)

## Data Processing Steps

1. **Data Format Conversion**
   - Original MOS scores: 0-100 scale
   - Converted to: 0-1 scale (divided by 100)
   - Column format changed to match UHD-IQA: `image_name,quality_mos`

2. **Training Set Creation**
   - File: `train_spaq.csv`
   - Content: First 5,563 images (00001.jpg to 05563.jpg)
   - Total lines: 5,564 (1 header + 5,563 data rows)
   - Command used:
     ```bash
     echo "image_name,quality_mos" > train_spaq.csv
     awk -F, 'NR>1 && NR<=5564 {printf "%s,%.4f\n", $1, $2/100}' "MOS and Image attribute scores.csv" >> train_spaq.csv
     ```

3. **Validation Set Creation**
   - File: `validation_spaq.csv`
   - Content: Remaining 5,562 images (05564.jpg to 11125.jpg)
   - Total lines: 5,563 (1 header + 5,562 data rows)
   - Command used:
     ```bash
     echo "image_name,quality_mos" > validation_spaq.csv
     awk -F, 'NR>5564 {printf "%s,%.4f\n", $1, $2/100}' "MOS and Image attribute scores.csv" >> validation_spaq.csv
     ```

## File Format
Both files follow the same format:
```csv
image_name,quality_mos
00001.jpg,0.4886
00002.jpg,0.6250
...
```

## Output Files Location
- Training set: `/SPAQ/Annotations/train_spaq.csv`
- Validation set: `/SPAQ/Annotations/validation_spaq.csv`

## Verification
- Both files maintain the UHD-IQA format
- MOS scores are properly normalized to 0-1 range
- Total images split matches original dataset size (5,563 + 5,562 = 11,125)
