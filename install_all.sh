#!/bin/bash

# Hadheq CTYDLE: Full Stack Setup
# This script initializes both Inference and Export environments.

set -e

echo "======================================================="
echo "🏰 Hadheq CTYDLE: Full Stack Setup Starting..."
echo "======================================================="

# Ensure scripts are executable
chmod +x setup_inference.sh
chmod +x setup_export.sh
chmod +x export_model.sh
chmod +x run_inference_live.sh

# Run setups
echo "1/2 :: Initializing Inference Environment..."
./setup_inference.sh

echo "2/2 :: Initializing Export Environment..."
./setup_export.sh

echo "======================================================="
echo "✅ CTYDLE Setup Complete!"
echo "======================================================="
echo "Next Steps:"
echo "1. Export a model: ./export_model.sh your_model.pt"
echo "2. Run live inference: ./run_inference_live.sh your_model_edgetpu.tflite"
echo "======================================================="
