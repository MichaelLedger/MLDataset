#!/bin/bash
# Script to run aesthetic scoring for UHD-IQA dataset

echo "=========================================="
echo "UHD-IQA Aesthetic Scoring"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Check if virtual environment exists and activate it
if [ -d "$SCRIPT_DIR/venv" ]; then
    echo "Activating virtual environment..."
    source "$SCRIPT_DIR/venv/bin/activate"
fi

echo "Python executable: $(which python3)"
echo ""

# Run the script - it will check packages itself
echo "Starting aesthetic scoring..."
echo "This will:"
echo "  1. Load ResNet50 model (auto-downloads on first run)"
echo "  2. Score all images in UHD-IQA/training (4,485 images)"
echo "  3. Score all images in UHD-IQA/validation (999 images)"
echo "  4. Generate train_aesthetic.csv and validation_aesthetic.csv"
echo ""

python3 aesthetic_scoring.py

echo ""
echo "Done!"
