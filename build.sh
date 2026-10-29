#!/usr/bin/env bash
# exit on error
set -o errexit

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing CPU-only PyTorch (reduces memory from 3GB to ~180MB)..."
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing remaining dependencies..."
pip install -r requirements.txt

echo "==> Build finished successfully!"
