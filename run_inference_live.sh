#!/bin/bash

# Configuration
VENV_PATH="./venv_inference"
PYTHON_EXE="$VENV_PATH/bin/python"

# Help message
show_help() {
    echo "Usage: ./run_inference_live.sh [model_path] [source] [labels_yaml] [extra_args...]"
    echo "Example: ./run_inference_live.sh yolo12n_edgetpu.tflite 0 coco.yaml"
    echo ""
    echo "Arguments:"
    echo "  model_path    Path to the .tflite model"
    echo "  source        Camera index or video file (Default: 0)"
    echo "  labels_yaml   Dataset config YAML for class names"
    echo "  extra_args    Additional arguments (e.g., --conf 0.5 --iou 0.4)"
}

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Error: Inference environment not found at $VENV_PATH."
    echo "   Please run ./setup_inference.sh first."
    exit 1
fi

# Check if model path is provided
if [ -z "$1" ]; then
    show_help
    exit 1
fi

MODEL=$1
SOURCE=${2:-0}
LABELS=$3
shift 3
EXTRA_ARGS=$@

echo "-------------------------------------------------------"
echo "🎥 Starting Live TPU Inference"
echo "   Model:  $MODEL"
echo "   Source: $SOURCE"
if [ ! -z "$LABELS" ]; then
    echo "   Labels: $LABELS"
fi
echo "-------------------------------------------------------"

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "❌ Error: Model file $MODEL not found."
    # Try to find it in the yolo12n_saved_model folder as a courtesy
    if [ -f "yolo12n_saved_model/$MODEL" ]; then
        MODEL="yolo12n_saved_model/$MODEL"
        echo "💡 Found model at: $MODEL"
    else
        exit 1
    fi
fi

# Run the inference using the venv python
if [ ! -z "$LABELS" ]; then
    $PYTHON_EXE infer_yolo_litert_live.py --model "$MODEL" --source "$SOURCE" --labels "$LABELS" $EXTRA_ARGS
else
    $PYTHON_EXE infer_yolo_litert_live.py --model "$MODEL" --source "$SOURCE" $EXTRA_ARGS
fi
