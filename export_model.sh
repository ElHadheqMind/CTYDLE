#!/bin/bash

# Configuration
VENV_PATH="./venv_export"
PYTHON_EXE="$VENV_PATH/bin/python"

# Help message
show_help() {
    echo "Usage: ./export_model.sh [model_path] [imgsz] [data_yaml] [extra_args...]"
    echo "Example: ./export_model.sh yolo12n.pt 640 coco.yaml"
    echo ""
    echo "Arguments:"
    echo "  model_path    Path to the .pt model (Default: yolo12n.pt)"
    echo "  imgsz         Input image size (Default: 640)"
    echo "  data_yaml     Dataset config YAML for calibration (Default: coco8.yaml)"
    echo "  extra_args    Additional arguments for the YOLO export command"
    echo ""
    echo "Options:"
    echo "  --setup       Force environment setup"
    echo "  --help, -h    Show this help message"
}

# --- SETUP PHASE ---
setup_environment() {
    echo "-------------------------------------------------------"
    echo "🚀 Setting up EXPORT Environment (Python 3.12)"
    echo "-------------------------------------------------------"

    # Check for Python 3.12 (Primary target for modern export)
    # If 3.12 is not available, we can fallback to 3.11
    if ! command -v python3.12 &> /dev/null; then
        if ! command -v python3.11 &> /dev/null; then
            echo "Python 3.11/3.12 not found. Installing Python 3.12..."
            sudo apt-get update
            sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
        fi
    fi

    # Determine which python to use for venv creation
    SYS_PYTHON="python3.12"
    if ! command -v python3.12 &> /dev/null; then
        SYS_PYTHON="python3.11"
    fi

    # Create venv if missing
    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating virtual environment in $VENV_PATH using $SYS_PYTHON..."
        $SYS_PYTHON -m venv "$VENV_PATH"
    fi

    # Upgrade pip
    $PYTHON_EXE -m pip install --upgrade pip

    echo "Installing Ultralytics and dependencies..."
    # Exporting typically doesn't need Coral wheels (just edgetpu-compiler and ultralytics)
    # We use CPU torch for smaller footprint
    $PYTHON_EXE -m pip install "ultralytics[export]" torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
    
    echo "Installing ONNX and Simplification tools..."
    $PYTHON_EXE -m pip install onnx onnxslim onnxruntime
    
    echo "Installing onnx2tf for TPU model conversion..."
    $PYTHON_EXE -m pip install onnx2tf tf_keras sng4onnx onnx-graphsurgeon

    # Ensure Edge TPU compiler is installed (System-wide)
    echo "Ensuring Edge TPU compiler is installed..."
    # Add Coral package repository
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/google-coral.gpg
    echo "deb [signed-by=/usr/share/keyrings/google-coral.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    sudo apt-get update

    # Install compiler
    sudo apt-get install -y edgetpu-compiler

    echo "✅ Export Environment Setup complete!"
    echo "-------------------------------------------------------"
}

# Check if venv exists, if not run setup
if [ ! -d "$VENV_PATH" ]; then
    setup_environment
fi

# Argument parsing
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    show_help
    exit 0
fi

if [ "$1" == "--setup" ]; then
    setup_environment
    shift # Remove --setup from arguments
fi

# Default values
MODEL=${1:-"yolo12n.pt"}
IMGSZ=${2:-640}
DATA_YAML=${3:-"coco8.yaml"}

# Verify model exists
if [ ! -f "$MODEL" ]; then
    # If the first argument looks like a flag, it might be that they want help
    if [[ "$MODEL" == -* ]]; then
        show_help
        exit 1
    fi
    echo "❌ Error: Model file $MODEL not found."
    exit 1
fi

shift 3
EXTRA_ARGS=$@

echo "-------------------------------------------------------"
echo "🚀 Exporting Model to Edge TPU"
echo "   Environment: $SYS_PYTHON (Venv at $VENV_PATH)"
echo "   Model:       $MODEL"
echo "   Size:        $IMGSZ"
echo "   Data (YAML): $DATA_YAML"
if [ ! -z "$EXTRA_ARGS" ]; then
    echo "   Extra:       $EXTRA_ARGS"
fi
echo "-------------------------------------------------------"

# Run the export using the venv python
$PYTHON_EXE -m ultralytics export model="$MODEL" format=edgetpu imgsz="$IMGSZ" data="$DATA_YAML" $EXTRA_ARGS

echo ""
echo "-------------------------------------------------------"
echo "✅ Export process finished!"
echo "Check the directory for '..._edgetpu.tflite' file."
echo "-------------------------------------------------------"
