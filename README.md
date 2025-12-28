# 🚀 Hadheq CTYDLE: The Flagship Edge AI Pipeline

<div align="center">
  
  <p align="center">
    <b>Coral TPU YOLO Deployment LiteRT Engine</b><br>
    The definitive high-precision bridge from PyTorch to Google Coral Silicon.
  </p>

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.9%20|%203.12-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/Hardware-Google%20Coral%20TPU-red.svg" alt="Hardware Support">
    <img src="https://img.shields.io/badge/Models-YOLOv8--v12-orange.svg" alt="Models Support">
    <img src="https://img.shields.io/badge/Runtime-LiteRT%203.0+-green.svg" alt="Runtime">
    <img src="https://img.shields.io/badge/License-MIT-gray.svg" alt="License">
  </p>
</div>

---

## 💎 The CTYDLE Philosophy

**Hadheq CTYDLE** (pronounced *Citadel*) is a flagship-grade deployment framework designed to solve the "Valley of Death" in Edge AI. It bridges the gap between modern, high-intensity ML models like **YOLOv12** and ultra-efficient hardware like the **Google Coral TPU**.

Deploying on Edge hardware usually requires navigating a nightmare of Bazel build errors, library conflicts, and the surgical requirement of **Full Integer Quantization**. **CTYDLE** automates all of this, providing a zero-manual-touch pipeline that ensures your models run at peak hardware capability in seconds, not weeks.

---

## 🌟 Why YOLO & Ultralytics?

**YOLO (You Only Look Once)** has revolutionized real-time vision. With the advent of **YOLOv12**, the architecture achieves state-of-the-art accuracy with significantly reduced computational overhead—making it the ideal candidate for the Edge TPU's systolic array.

**[Ultralytics](https://docs.ultralytics.com/)** provides the backbone of this engine, offering a unified API for the entire YOLO lineage (v8 to v12). CTYDLE enhances this by specializing the export and inference stages for the architectural nuances of Google AI Edge hardware.

---

## 📐 Architecture: Dual-Engine Strategy

To achieve zero-lag inference while maintaining modern compatibility, CTYDLE utilizes a **Dual-Environment Strategy**:

1.  **Export Engine (`venv_export`)**: Powered by **Python 3.12**. Optimized for high-speed quantization arithmetic, `onnx2tf` translation, and `ultralytics` serialization.
2.  **Inference Engine (`venv_inference`)**: Powered by **Python 3.9**. Optimized for rock-solid stability with official `pycoral` drivers and the latest `ai-edge-litert` (LiteRT) runtime.

---

## 🛠 Prerequisites

### 🖥 Hardware
*   **Google Coral**: USB Accelerator, M.2, or Mini PCIe.
*   **Host**: Raspberry Pi 4/5 (64-bit), Ubuntu workstation, or Industrial x86_64 Edge server.

### 🐧 Software
*   **OS**: Linux (Debian/Ubuntu-based).
*   **Connectivity**: Active USB 3.0 port (for USB Accelerator).

---

## 📦 1. Installation Guide

**Hadheq CTYDLE** is designed for "Zero-Manual-Touch" configuration. Our setup scripts handle GPG keys, Coral repositories, C++ runtimes, and isolated Python environments automatically.

### Initialize the Full Stack (Recommended)
```bash
chmod +x install_all.sh && ./install_all.sh
```

### Manual Environment setup
If you prefer to isolate the stages:
```bash
# Set up the high-stability inference environment (Python 3.9)
chmod +x setup_inference.sh && ./setup_inference.sh

# Set up the high-precision export environment (Python 3.12)
chmod +x setup_export.sh && ./setup_export.sh
```

> **💡 Note**: No `sudo` or manual driver installation is required outside of these scripts.

---

## 🔄 2. Surgical Model Export

Convert your PyTorch `.pt` models into hardware-ready `.tflite` files with a single command. The engine handles the entire transformation chain: **PyTorch** ➡️ **ONNX** ➡️ **TF SavedModel** ➡️ **INT8 TFLite** ➡️ **Edge TPU Compiled**.

```bash
# Syntax: ./export_model.sh [model] [imgsz] [data_yaml]
./export_model.sh yolo12n.pt 640 coco.yaml
```

### 🔬 The Secret Sauce: Quantization & Calibration
The Edge TPU *requires* **Full Integer (INT8)** models. CTYDLE uses your `.yaml` config to perform **Representative Calibration**—running sample data through the model to observe dynamic ranges and map them to 8-bit integers with surgical precision.

### 📦 Export Artifacts
- **`model_edgetpu.tflite`**: The final artifact compiled for the Edge TPU (**USE THIS**).
- **`model_full_integer_quant.tflite`**: The standard INT8 model for CPU-based LiteRT fallbacks.
- **`metadata.yaml`**: Label mappings extracted from the source model.

---

## 🎯 3. High-Performance Inference

CTYDLE features **Smart-Switch Technology**. You can run these commands from any terminal; the system will automatically invoke the correct virtual environment for you.

### Static Image Detection
```bash
python3 infer_yolo_litert.py --model yolo12n_edgetpu.tflite --image test.jpg --labels coco.yaml --show
```

### Real-Time Live Stream
```bash
./run_inference_live.sh yolo12n_edgetpu.tflite 0 coco.yaml
```
*Tip: `--source 0` is your default webcam. Support for RPi Cam modules is native.*

---

### 📊 Technical Specifications
| Stage | Tool / Format | Version / Target |
| :--- | :--- | :--- |
| **Source** | PyTorch (`.pt`) | YOLOv8/v10/v11/v12 |
| **Intermediate** | ONNX | Opset 17+ with Sim-Optimization |
| **Bridge** | TF SavedModel | TensorFlow 2.15+ (v3 Schema) |
| **Quantization** | TFLite | Full INT8 (Representative Calibration) |
| **Final** | Edge TPU TFLite | Compiled with `edgetpu_compiler v16.0` |
| **Runtime** | LiteRT | `ai-edge-litert` (v3.0+) |

---

## 🛠 Diagnostics & Troubleshooting

Always know exactly what your hardware is doing:
```bash
python3 check_tpu.py
```
This utility verifies **USB link speed**, **Udev permissions**, **Driver health**, and **Delegate loading** instantly.

**TPU Not Found?**
1.  **Re-run Setup**: Ensure `setup_inference.sh` finished without errors.
2.  **Groups**: Your user must be in the `plugdev` group (handled by the setup script, but requires a re-login).
3.  **USB Port**: Ensure you are using a USB 3.0 port (typically blue) for maximum data throughput.

---

## 📖 Deep Dive: The Medium Article
For a comprehensive architectural breakdown and a step-by-step masterclass, read our official publication:  
👉 **[CTYDLE: The Ultimate Guide to Run YOLO on Google Coral TPU with LiteRT](https://medium.com/@mezzihoussem/ctydle-the-ultimate-guide-to-run-yolo-on-google-coral-tpu-with-litert-c8045fb8853d)**

---

## 👨‍💻 Developed By
**El Hadheq Mind**  
*Empowering Intelligence at the Edge.*

🌍 [Website](https://elhadheqmind.com/) | 🐙 [GitHub](https://github.com/ElHadheqMind/CTYDLE)

---

## 📄 License
MIT License - Developed for the global Edge AI community.
