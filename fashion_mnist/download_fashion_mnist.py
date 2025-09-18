from datasets import load_dataset
import os
from datasets import config

# Print cache directory
print("Default cache directory:", config.HF_DATASETS_CACHE)

# Load the Fashion-MNIST dataset
dataset = load_dataset("zalando-datasets/fashion_mnist")

# The dataset will be downloaded and cached automatically
print("\nDataset downloaded successfully!")
print("Available splits:", dataset.keys())
print("Number of training examples:", len(dataset["train"]))
print("Number of test examples:", len(dataset["test"]))

# Print an example to see the data structure
print("\nExample data structure:")
print(dataset["train"][0])