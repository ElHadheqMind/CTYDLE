#!/bin/bash

# Exit on error
set -e

echo "-------------------------------------------------------"
echo "Setting up INFERENCE Environment (Python 3.9)"
echo "-------------------------------------------------------"

# Check for Python 3.9
if ! command -v python3.9 &> /dev/null; then
    echo "Python 3.9 not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.9 python3.9-venv python3.9-dev
fi

# Create venv
VENV_PATH="./venv_inference"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment in $VENV_PATH..."
    python3.9 -m venv "$VENV_PATH"
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install compatible numpy
pip install "numpy<2"

# Detect architecture and install specific wheels
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

if [ "$ARCH" == "x86_64" ]; then
    TFLITE_URL="https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_x86_64.whl"
    PYCORAL_URL="https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp39-cp39-linux_x86_64.whl"
elif [ "$ARCH" == "aarch64" ]; then
    TFLITE_URL="https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl"
    PYCORAL_URL="https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp39-cp39-linux_aarch64.whl"
else
    echo "❌ Error: Unsupported architecture $ARCH"
    exit 1
fi

echo "Installing tflite-runtime from $TFLITE_URL..."
pip install "$TFLITE_URL"

echo "Installing PyCoral from $PYCORAL_URL..."
pip install "$PYCORAL_URL"

# Install LiteRT (the new TFLite branding as requested)
echo "Installing ai-edge-litert..."
pip install ai-edge-litert

# YOLO and other inference dependencies
echo "Installing YOLOv12 and other dependencies..."
pip install ultralytics torch torchvision opencv-python Pillow pyyaml

# System dependencies for Edge TPU (libedgetpu)
echo "Ensuring Edge TPU runtime and compiler are installed..."
# Add Coral package repository
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/google-coral.gpg
echo "deb [signed-by=/usr/share/keyrings/google-coral.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update

# Install runtime and compiler
sudo apt-get install -y libedgetpu1-std edgetpu-compiler

# Setup USB rules if not present
if [ ! -f "/etc/udev/rules.d/99-edgetpu-accelerator.rules" ]; then
    echo "Setting up USB rules for Edge TPU..."
    echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="1a6e", GROUP="plugdev", MODE="0660"' | sudo tee /etc/udev/rules.d/99-edgetpu-accelerator.rules
    echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", ATTR{idProduct}=="9302", GROUP="plugdev", MODE="0660"' | sudo tee -a /etc/udev/rules.d/99-edgetpu-accelerator.rules
    sudo udevadm control --reload-rules && sudo udevadm trigger
    
    # Ensure plugdev group exists and add user
    sudo groupadd -f plugdev
    sudo usermod -aG plugdev $USER
    echo "USB rules installed. You might need to re-plug your TPU or log out/in."
fi

echo ""
echo "Inference environment setup complete!"
echo "Verifying environment..."
python check_tpu.py || echo "Warning: check_tpu.py failed, but environment setup finished."

echo "To activate: source venv_inference/bin/activate"
echo "-------------------------------------------------------"
