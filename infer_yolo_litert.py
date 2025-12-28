import os
import sys

# --- AUTO-SWITCH LOGIC ---
script_dir = os.path.dirname(os.path.abspath(__file__))
inference_python = os.path.join(script_dir, "venv_inference", "bin", "python")

# Always switch to venv_inference if we aren't using it
if os.path.exists(inference_python) and sys.executable != os.path.abspath(inference_python):
    print(f"\n🔍 Automatically switching to Inference Environment (Python 3.9)...")
    os.execv(inference_python, [inference_python] + sys.argv)
    sys.exit(0)
# -------------------------

import cv2
import numpy as np
import time
import argparse
import subprocess

def check_dependencies():
    """Ensures LiteRT and other dependencies are installed."""
    try:
        import ai_edge_litert
        return
    except ImportError:
        try:
            import litert
            return
        except ImportError:
            print("LiteRT not found. Please ensure you are in the correctly set up environment.")
            print("Run: source venv_inference/bin/activate")
            sys.exit(1)

# Run dependency check before anything else
# check_dependencies() # Removed to prevent conflict

# Now imports can proceed safely using a robust fallback logic
try:
    import ai_edge_litert.interpreter as tflite
    print("Using ai_edge_litert (LiteRT)")
except ImportError:
    try:
        import litert.interpreter as tflite
        print("Using litert (LiteRT)")
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            print("Using tflite_runtime")
        except ImportError:
            try:
                from tensorflow import lite as tflite
                print("Using tensorflow.lite")
            except ImportError:
                print("Error: No TFLite/LiteRT runtime found.")
                sys.exit(1)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class YOLOLiteRT:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45, labels_path=None):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Load Class Names
        self.names = {}
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.dirname(model_path)
        
        # Priority 1: User-specified labels_path
        # Priority 2: metadata.yaml in model directory
        # Priority 3: coco.yaml in script directory
        
        load_path = labels_path
        if not load_path or not os.path.exists(load_path):
            metadata_path = os.path.join(model_dir, "metadata.yaml")
            coco_path = os.path.join(script_dir, "coco.yaml")
            load_path = metadata_path if os.path.exists(metadata_path) else (coco_path if os.path.exists(coco_path) else None)
        
        if load_path and os.path.exists(load_path):
            try:
                import yaml
                with open(load_path, 'r') as f:
                    meta = yaml.safe_load(f)
                    # Support both 'names' key (YOLO format) and direct mapping
                    if isinstance(meta, dict):
                        self.names = meta.get('names', meta)
                print(f"Loaded {len(self.names)} class names from {os.path.basename(load_path)}")
            except Exception as e:
                print(f"Warning: Could not load labels from {load_path}: {e}")
        
        if not self.names:
            print("Warning: No class names loaded. Using IDs as labels.")

        # Initialize Interpreter using Official PyCoral API
        # Reference: https://www.coral.ai/docs/reference/py/
        # Using pycoral.utils.edgetpu.make_interpreter() is the recommended method
        try:
            from pycoral.utils import edgetpu
            self.interpreter = edgetpu.make_interpreter(model_path)
            self.interpreter.allocate_tensors()
            print("✓ Edge TPU Interpreter initialized successfully using PyCoral API")
            print(f"  Using: pycoral.utils.edgetpu.make_interpreter()")
        except ImportError as e:
            print(f"⚠️  PyCoral not available: {e}")
            print("  Trying manual delegate loading...")
            # Manual fallback
            try:
                self.interpreter = tflite.Interpreter(
                    model_path=model_path,
                    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
                )
                self.interpreter.allocate_tensors()
                print("✓ Edge TPU loaded with manual delegate")
            except Exception as e2:
                print(f"⚠️  Manual delegate failed: {e2}")
                print("  Falling back to CPU (will fail for compiled models)")
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
        except Exception as e:
            print(f"⚠️  Edge TPU initialization failed: {e}")
            print("  Checking if we can load with manual delegate...")
            # Manual fallback
            try:
                self.interpreter = tflite.Interpreter(
                    model_path=model_path,
                    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
                )
                self.interpreter.allocate_tensors()
                print("✓ Edge TPU loaded with manual delegate")
            except Exception as e2:
                print(f"⚠️  Manual fallback failed: {e2}")
                print("  Falling back to CPU (will fail for compiled models)")
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.img_size = self.input_shape[1] # Assumes square input

    def get_name(self, cls_id):
        """Returns the class name for a given ID."""
        return self.names.get(int(cls_id), f"ID:{int(cls_id)}")

    def preprocess(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_res, ratio, pad = letterbox(img_rgb, (self.img_size, self.img_size), auto=False)
        
        # Normalize to 0.0 - 1.0 (float32) or keep as int8 if model is quantized
        if self.input_details[0]['dtype'] == np.uint8 or self.input_details[0]['dtype'] == np.int8:
            input_data = img_res.astype(self.input_details[0]['dtype'])
        else:
            input_data = img_res.astype(np.float32) / 255.0
        
        input_data = np.expand_dims(input_data, axis=0)
        return input_data, ratio, pad

    def postprocess(self, outputs, ratio, pad, original_shape):
        # YOLOv8-v12 typically outputs [1, 84, 8400] or similar
        # We need to transpose to [8400, 84]
        output = outputs[0]
        
        # De-quantize if necessary
        out_details = self.output_details[0]
        if out_details['dtype'] == np.int8 or out_details['dtype'] == np.uint8:
            scale, zero_point = out_details['quantization']
            output = (output.astype(np.float32) - zero_point) * scale
            
        predictions = np.squeeze(output)
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        
        # predictions: [N, 4 + num_classes]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        # Get max score and class for each box
        max_scores = np.max(scores, axis=1)
        classes = np.argmax(scores, axis=1)
        
        # Filter by confidence
        mask = max_scores > self.conf_thres
        boxes = boxes[mask]
        scores = max_scores[mask]
        classes = classes[mask]

        if len(boxes) == 0:
            return [], [], []

        # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
        # And rescale to original image size
        dw, dh = pad
        rw, rh = ratio
        
        # Model outputs normalized coordinates [0, 1]
        # We transform them to pixels on the 640x640 canvas first
        x = boxes[:, 0] * self.img_size
        y = boxes[:, 1] * self.img_size
        w = boxes[:, 2] * self.img_size
        h = boxes[:, 3] * self.img_size
        
        # Now convert to original image coordinates
        x1 = (x - w/2 - dw) / rw
        y1 = (y - h/2 - dh) / rh
        x2 = (x + w/2 - dw) / rw
        y2 = (y + h/2 - dh) / rh

        # Clip boxes to image boundaries
        x1 = np.clip(x1, 0, original_shape[1])
        y1 = np.clip(y1, 0, original_shape[0])
        x2 = np.clip(x2, 0, original_shape[1])
        y2 = np.clip(y2, 0, original_shape[0])
        
        final_boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # NMS
        indices = cv2.dnn.NMSBoxes(final_boxes.tolist(), scores.tolist(), self.conf_thres, self.iou_thres)
        
        if len(indices) > 0:
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            return final_boxes[indices], scores[indices], classes[indices]
        return [], [], []

    def run(self, img):
        original_shape = img.shape[:2]
        input_data, ratio, pad = self.preprocess(img)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = (time.perf_counter() - start) * 1000
        
        outputs = [self.interpreter.get_tensor(item['index']) for item in self.output_details]
        
        boxes, scores, classes = self.postprocess(outputs, ratio, pad, original_shape)
        return boxes, scores, classes, inference_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to .tflite model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--show', action='store_true', help='Show output image')
    parser.add_argument('--save', type=str, default='output.jpg', help='Save output image path')
    parser.add_argument('--labels', type=str, help='Path to .yaml file containing class names')
    args = parser.parse_args()

    # Load model
    yolo = YOLOLiteRT(args.model, args.conf, args.iou, labels_path=args.labels)

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not load image {args.image}")
        return

    # Run inference
    boxes, scores, classes, infer_time = yolo.run(img)
    print(f"Inference time: {infer_time:.2f}ms")

    # Draw results
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        name = yolo.get_name(cls)
        cv2.putText(img, f"{name}: {score:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save/Show
    if args.save:
        cv2.imwrite(args.save, img)
        print(f"Result saved to {args.save}")
    
    if args.show:
        cv2.imshow("Detection", img)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
