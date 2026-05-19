#!/bin/bash
set -euo pipefail

# Run once to set up the environment.
# No GPU or special hardware required — pure OpenCV on CPU.
# Usage: bash setup.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "Using $($PYTHON_BIN --version)"

echo "Creating virtual environment (.humming)..."
"$PYTHON_BIN" -m venv .humming
source .humming/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install jupyter opencv-python scikit-image pandas matplotlib numpy

echo ""
echo "Setup complete. To start the notebook:"
echo "  source .humming/bin/activate"
echo "  jupyter notebook hummingbird_analysis_local.ipynb"
