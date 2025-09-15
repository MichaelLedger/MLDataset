import pandas as pd

# Read the metadata file
df = pd.read_csv('/Users/gavinxiang/Downloads/MLDataset/UHD-IQA/uhd-iqa-metadata/uhd-iqa-metadata.csv')

# Split into train and validation sets
train_df = df[df['set'] == 'training'][['image_name', 'quality_mos']]
val_df = df[df['set'] == 'validation'][['image_name', 'quality_mos']]

# Save the files
train_df.to_csv('/Users/gavinxiang/Downloads/MLDataset/UHD-IQA/uhd-iqa-metadata/train.csv', index=False)
val_df.to_csv('/Users/gavinxiang/Downloads/MLDataset/UHD-IQA/uhd-iqa-metadata/validation.csv', index=False)

print(f"Created train.csv with {len(train_df)} samples")
print(f"Created validation.csv with {len(val_df)} samples")
