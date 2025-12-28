import os
import sys

# --- AUTO-SWITCH LOGIC ---
script_dir = os.path.dirname(os.path.abspath(__file__))
inference_python = os.path.join(script_dir, "venv_inference", "bin", "python")

# Always switch to venv_inference if we aren't using it
if os.path.exists(inference_python) and sys.executable != os.path.abspath(inference_python):
    print(f"\n🎥 Automatically switching to Inference Environment (Python 3.9)...")
    os.execv(inference_python, [inference_python] + sys.argv)
    sys.exit(0)
# -------------------------

import subprocess

def check_dependencies():
    """Ensures LiteRT and other dependencies are installed."""
    try:
        import ai_edge_litert
        import cv2
        import numpy
        return
    except ImportError:
        print("Missing dependencies. Please ensure you are in the correct environment.")
        print("Run: source venv_inference/bin/activate")
        sys.exit(1)

# check_dependencies() # Disabled

import cv2
import numpy as np
import time
import argparse
import threading
from infer_yolo_litert import YOLOLiteRT

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                continue
            with self.lock:
                self.frame = frame
                self.ret = ret

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to .tflite model')
    parser.add_argument('--source', type=str, default='0', help='Camera index or video file file')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--labels', type=str, help='Path to .yaml file containing class names')
    args = parser.parse_args()
    
    # Path validation
    if not os.path.exists(args.model):
        print(f"\n❌ Error: Model file '{args.model}' not found.")
        # Try to suggest the correct path
        suggested = os.path.join("yolo12n_saved_model", args.model)
        if os.path.exists(suggested):
            print(f"💡 Did you mean: --model {suggested}")
        sys.exit(1)

    # Load model
    print("Initializing YOLO LiteRT...")
    yolo = YOLOLiteRT(args.model, args.conf, args.iou, labels_path=args.labels)

    # Initialize Video Stream
    src = int(args.source) if args.source.isdigit() else args.source
    vs = VideoStream(src).start()
    time.sleep(1) # Warm up camera

    print("Starting Live Inference. Press 'q' to quit.")
    
    prev_time = time.time()
    while True:
        ret, frame = vs.read()
        if not ret:
            break

        # Run inference
        boxes, scores, classes, infer_time = yolo.run(frame)

        # Draw results
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 3)
            name = yolo.get_name(cls)
            label = f"{name} {score:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Overlay Info
        info = f"Inference: {infer_time:.1f}ms | FPS: {fps:.1f}"
        cv2.rectangle(frame, (5, 5), (350, 45), (0, 0, 0), -1)
        cv2.putText(frame, info, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("YOLO LiteRT Live", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
