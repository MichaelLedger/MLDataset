#!/usr/bin/env python3
"""
Script to clean CSV files by removing rows where the corresponding image files don't exist.
"""

import pandas as pd
import os
import sys
from pathlib import Path


def clean_csv(csv_path, image_dir):
    """
    Clean CSV file by removing rows where the corresponding image file doesn't exist.
    
    Args:
        csv_path: Path to the CSV file
        image_dir: Directory containing the images
    """
    
    # Read the CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Original CSV has {len(df)} rows")
    
    # Get the image column name (assuming it's the first column or named 'image_name')
    image_column = df.columns[0]
    print(f"Image column: {image_column}")
    
    # Create a list to store rows where images exist
    valid_rows = []
    missing_images = []
    
    for idx, row in df.iterrows():
        image_name = row[image_column]
        image_path = os.path.join(image_dir, image_name)
        
        if os.path.exists(image_path):
            valid_rows.append(row)
        else:
            missing_images.append(image_name)
            if len(missing_images) <= 10:  # Show first 10 missing images
                print(f"Missing image: {image_name}")
    
    if len(missing_images) > 10:
        print(f"... and {len(missing_images) - 10} more missing images")
    
    # Create new DataFrame with only valid rows
    if valid_rows:
        cleaned_df = pd.DataFrame(valid_rows)
        cleaned_df.reset_index(drop=True, inplace=True)
    else:
        print("Warning: No valid images found!")
        cleaned_df = pd.DataFrame(columns=df.columns)
    
    print(f"Cleaned CSV has {len(cleaned_df)} rows")
    print(f"Removed {len(df) - len(cleaned_df)} rows with missing images")
    
    # Create backup of original file
    backup_path = csv_path + '.backup'
    if not os.path.exists(backup_path):
        print(f"Creating backup: {backup_path}")
        df.to_csv(backup_path, index=False)
    
    # Save cleaned CSV
    print(f"Saving cleaned CSV: {csv_path}")
    cleaned_df.to_csv(csv_path, index=False)
    
    return len(missing_images)


def main():
    """Main function to clean SPAQ+UHD dataset CSV files."""
    
    # Define paths
    base_dir = "/Users/macminiai/VSCODE_Projects/LAR-IQA/dataset/SPAQ+UHD"
    train_csv = os.path.join(base_dir, "train.csv")
    validation_csv = os.path.join(base_dir, "validation.csv")
    image_dir = os.path.join(base_dir, "train_and_validation")
    
    # Check if files exist
    if not os.path.exists(train_csv):
        print(f"Error: Train CSV not found: {train_csv}")
        return 1
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        return 1
    
    print("=" * 60)
    print("Cleaning SPAQ+UHD CSV files")
    print("=" * 60)
    
    # Clean train.csv
    print("\nCleaning train.csv...")
    missing_train = clean_csv(train_csv, image_dir)
    
    # Clean validation.csv if it exists
    if os.path.exists(validation_csv):
        print("\nCleaning validation.csv...")
        missing_val = clean_csv(validation_csv, image_dir)
    else:
        print(f"Validation CSV not found: {validation_csv}")
        missing_val = 0
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"Train CSV: {missing_train} missing images removed")
    if os.path.exists(validation_csv):
        print(f"Validation CSV: {missing_val} missing images removed")
    print("Backup files created with .backup extension")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
