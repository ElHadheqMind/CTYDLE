import os
import sys

# --- AUTO-SWITCH LOGIC ---
script_dir = os.path.dirname(os.path.abspath(__file__))
export_python = os.path.join(script_dir, "venv_export", "bin", "python")

# If we are not using the export venv and it exists, switch to it
if os.path.exists(export_python) and sys.executable != os.path.abspath(export_python):
    # Only switch if we are missing dependencies in the current environment
    try:
        from ultralytics import YOLO
    except ImportError:
        print(f"\n🚀 Switching to Export Environment (Python 3.12)...")
        os.execv(export_python, [export_python] + sys.argv)
        sys.exit(0)
# -------------------------

import argparse
from ultralytics import YOLO

def convert_model():
    # --- AUTO-SWITCH LOGIC ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    export_python = os.path.join(script_dir, "venv_export", "bin", "python")
    
    # If we aren't in the export venv and it exists, switch to it
    if os.path.exists(export_python) and sys.executable != os.path.abspath(export_python):
        print(f"\n🚀 Switching to Export Environment (Python 3.12)...")
        os.execv(export_python, [export_python] + sys.argv)
        sys.exit(0)
    # -------------------------
    parser = argparse.ArgumentParser(description='Convert YOLOv12 model to Edge TPU format')
    parser.add_argument('--model', type=str, required=True, help='Path to the .pt model file')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size (default: 640)')
    parser.add_argument('--output_name', type=str, help='Custom name for the output .tflite file')
    parser.add_argument('--data', type=str, default='coco8.yaml', help='Path to the .yaml data file for calibration (default: coco8.yaml)')
    args = parser.parse_args()

    model_path = args.model
    
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        sys.exit(1)
    
    print(f"Starting export for {model_path} to Edge TPU format...")
    print(f"Using data config: {args.data}")
    print("This will perform INT8 quantization and run the edgetpu_compiler.")
    
    try:
        model = YOLO(model_path)
        
        # Exporting with format="edgetpu" automatically:
        # 1. Converts to TFLite
        # 2. Performs INT8 Quantization
        # 3. Runs the edgetpu_compiler
        export_path = model.export(
            format="edgetpu", 
            imgsz=args.imgsz, 
            data=args.data # Calibration data for quantization
        )
        
        print(f"\nOriginally exported to: {export_path}")

        # Rename if requested
        if args.output_name:
            # Ensure name ends with _edgetpu.tflite if not already
            new_name = args.output_name
            if not new_name.lower().endswith('.tflite'):
                new_name += "_edgetpu.tflite"
            
            new_path = os.path.join(os.path.dirname(export_path), new_name)
            if os.path.exists(export_path):
                os.rename(export_path, new_path)
                print(f"Model renamed to: {new_path}")
            else:
                print(f"Warning: Could not find exported file to rename at {export_path}")
        
    except Exception as e:
        print(f"\nERROR during conversion: {e}")
        print("\nNote: Ensure 'edgetpu_compiler' is installed on your system.")

if __name__ == "__main__":
    convert_model()
