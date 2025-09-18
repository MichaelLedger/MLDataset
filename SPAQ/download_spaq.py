import json
import os
import subprocess
from pathlib import Path

def load_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        return json.load(f)

def download_spaq_dataset(metadata_path, output_dir):
    # Load and validate metadata
    metadata = load_metadata(metadata_path)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract dataset info from metadata
    distribution = metadata.get('distribution', [])
    if not distribution:
        raise ValueError("No distribution information found in metadata")
    
    # Get main archive info
    archive_info = next((d for d in distribution if d.get('@id') == 'archive.zip'), None)
    if not archive_info:
        raise ValueError("Archive information not found in metadata")
    
    # Extract Kaggle dataset name from contentUrl
    content_url = archive_info.get('contentUrl', '')
    if 'kaggle.com' not in content_url:
        raise ValueError("Not a Kaggle dataset URL")
    
    # Extract Kaggle dataset path (username/dataset-name)
    kaggle_path = "anamikakumari22/spaq-dataset"
    
    print(f"Downloading SPAQ dataset to {output_dir}")
    print(f"Dataset size: {archive_info.get('contentSize', 'unknown')}")
    
    # Download using Kaggle CLI
    try:
        subprocess.run([
            'kaggle', 'datasets', 'download',
            kaggle_path,
            '-p', output_dir,
            '--unzip'
        ], check=True)
        print("Download and extraction completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        return False
    
    return True

if __name__ == "__main__":
    metadata_path = "dataset/SPAQ/spaq-dataset-metadata.json"
    output_dir = "dataset/SPAQ"
    
    success = download_spaq_dataset(metadata_path, output_dir)
    if success:
        print("\nDataset files:")
        for path in Path(output_dir).rglob('*'):
            if path.is_file():
                print(f"- {path.relative_to(output_dir)}")
