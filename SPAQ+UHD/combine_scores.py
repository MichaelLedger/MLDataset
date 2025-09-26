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

print(f"Total samples: {len(combined_df)}")
print("\nFirst few rows:")
print(combined_df.head())
print("\nQuality MOS range:")
print(f"Min: {combined_df['quality_mos'].min():.4f}")
print(f"Max: {combined_df['quality_mos'].max():.4f}")
