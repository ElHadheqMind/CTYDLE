#!/bin/bash

# Exit on error
set -e

echo "-------------------------------------------------------"
echo "Setting up EXPORT/MODEL Environment (Python 3.12)"
echo "-------------------------------------------------------"

# Check for Python 3.12 (or fallback to 3.11)
if ! command -v python3.12 &> /dev/null; then
    if ! command -v python3.11 &> /dev/null; then
        echo "Python 3.11/3.12 not found. Installing Python 3.12..."
        sudo apt-get update
        sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
    fi
fi

# Select available python
SYS_PYTHON="python3.12"
if ! command -v python3.12 &> /dev/null; then
    SYS_PYTHON="python3.11"
fi

# Create venv
VENV_PATH="./venv_export"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment in $VENV_PATH using $SYS_PYTHON..."
    $SYS_PYTHON -m venv "$VENV_PATH"
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install Ultralytics and export tools
echo "Installing Ultralytics and dependencies..."
# We use CPU-only torch to save space and time for exports
pip install "ultralytics[export]" torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Essential export tools for TPU
echo "Installing ONNX and Simplification tools..."
pip install onnx onnxslim onnxruntime 

# onnx2tf and dependencies
echo "Installing onnx2tf for TPU model conversion..."
pip install onnx2tf tf_keras sng4onnx onnx-graphsurgeon

# Ensure Edge TPU compiler is installed (System-wide)
echo "Ensuring Edge TPU compiler is installed..."
# Add Coral package repository
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/google-coral.gpg
echo "deb [signed-by=/usr/share/keyrings/google-coral.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update

# Install compiler
sudo apt-get install -y edgetpu-compiler

echo ""
echo "Export environment setup complete!"
echo "Python version: $($SYS_PYTHON --version)"
echo "To activate: source venv_export/bin/activate"
echo "-------------------------------------------------------"
