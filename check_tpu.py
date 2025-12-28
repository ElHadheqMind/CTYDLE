#!/usr/bin/env python3
"""
Simple TPU connectivity check script.
Verifies that the Edge TPU is connected and ready to use.
"""

import sys
import os
import subprocess

def check_tpu_connection():
    """Check if Edge TPU is connected and accessible."""
    
    # --- AUTO-SWITCH LOGIC ---
    # If we are missing pycoral, try to find the inference venv and run there
    try:
        import pycoral
    except ImportError:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        inference_python = os.path.join(script_dir, "venv_inference", "bin", "python")
        if os.path.exists(inference_python) and sys.executable != inference_python:
            print(f"\n💡 Note: PyCoral not found in current environment ({sys.version.split()[0]}).")
            print(f"   Automatically switching to Inference Environment (Python 3.9)...")
            os.execv(inference_python, [inference_python] + sys.argv)
            sys.exit(0)
    # -------------------------

    print("\n" + "="*70)
    print("  🔍 EDGE TPU SYSTEM DIAGNOSTIC")
    print("="*70 + "\n")

    # Step 0: Check Edge TPU Compiler (Critical for EXPORT)
    print("🏗️  Checking Edge TPU Compiler (needed for model export)...")
    try:
        compiler_version = subprocess.check_output(["edgetpu_compiler", "--version"], text=True).strip()
        print(f"✓ edgetpu_compiler found: {compiler_version}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("✗ edgetpu_compiler NOT FOUND")
        print("  Note: This is required in the EXPORT environment to create .tflite files for TPU.")
    
    # Step 1: Check if pycoral is installed
    try:
        from pycoral.utils import edgetpu
        import pycoral
        
        # Show installation details
        pycoral_path = pycoral.__file__
        print(f"✓ PyCoral library is installed")
        print(f"  Location: {pycoral_path}")
        
        # Try to get version
        try:
            version = pycoral.__version__
            print(f"  Version: {version}")
        except AttributeError:
            print(f"  Version: (version info not available)")
            
    except ImportError as e:
        print(f"✗ PyCoral library not found: {e}")
        print("\n❌ TPU NOT READY - Please install PyCoral first")
        print("   Run: python3 fix_pycoral_install.py")
        return False
    
    # Step 2: Check if TFLite/LiteRT Runtime is installed
    try:
        import ai_edge_litert.interpreter as tflite
        print("✓ ai-edge-litert (LiteRT) is installed")
    except ImportError:
        try:
            import litert.interpreter as tflite
            print("✓ litert is installed")
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite
                print("✓ tflite-runtime is installed")
            except ImportError as e:
                print(f"✗ No TFLite/LiteRT Runtime found: {e}")
                print("\n❌ TPU NOT READY - Please install ai-edge-litert or tflite_runtime")
                return False
    
    # Step 3: Check for libedgetpu library
    print("\n📚 Checking libedgetpu library...")
    import ctypes.util
    
    lib_found = False
    lib_path = None
    
    # Try to find the library
    for lib_name in ["edgetpu", "libedgetpu"]:
        lib_path = ctypes.util.find_library(lib_name)
        if lib_path:
            lib_found = True
            break
    
    # Check common installation paths
    if not lib_found:
        common_paths = [
            "/usr/lib/x86_64-linux-gnu/libedgetpu.so.1",
            "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1",
            "/usr/local/lib/libedgetpu.so.1",
            "/usr/lib/libedgetpu.so.1",
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                lib_path = path
                lib_found = True
                break
    
    if lib_found:
        print(f"✓ libedgetpu found at: {lib_path}")
    else:
        print("✗ libedgetpu library not found")
        print("\n❌ TPU NOT READY - libedgetpu not installed")
        print("   Run: python3 setup_coral.py")
        return False
    
    # Step 4: Detect Edge TPU devices
    print("\n🔌 Detecting Edge TPU devices...")
    
    try:
        devices = edgetpu.list_edge_tpus()
        
        if not devices:
            print("✗ No Edge TPU devices detected")
            print("\n⚠️  TPU NOT CONNECTED")
            print("\nTroubleshooting:")
            print("  1. Make sure your Edge TPU is physically connected via USB")
            print("  2. Check USB connection with: lsusb | grep -i 'google\\|global'")
            print("  3. Verify udev rules are set up correctly")
            print("  4. Try unplugging and replugging the TPU")
            print("  5. Check if you need to add your user to 'plugdev' group:")
            print("     sudo usermod -aG plugdev $USER")
            print("     (then log out and log back in)")
            return False
        
        print(f"✓ Found {len(devices)} Edge TPU device(s):")
        for i, device in enumerate(devices):
            print(f"  [{i}] {device}")
        
    except Exception as e:
        print(f"✗ Error detecting Edge TPU: {e}")
        print("\n❌ TPU CHECK FAILED")
        return False
    
    # Step 5: Try to load the Edge TPU delegate
    print("\n⚙️  Testing Edge TPU delegate...")
    
    try:
        delegate = edgetpu.load_edgetpu_delegate()
        print("✓ Edge TPU delegate loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load Edge TPU delegate: {e}")
        print("\n⚠️  TPU DETECTED BUT NOT ACCESSIBLE")
        print("\nThis might be a permissions issue. Try:")
        print("  1. Check udev rules: ls -l /etc/udev/rules.d/*edgetpu*")
        print("  2. Reload udev rules: sudo udevadm control --reload-rules && sudo udevadm trigger")
        print("  3. Reconnect the TPU device")
        return False
    
    # All checks passed!
    print("\n" + "="*70)
    print("  ✅ TPU IS READY AND CONNECTED!")
    print("="*70)
    print("\n🎉 Your Edge TPU is properly configured and ready to use!")
    print("\nNext steps:")
    print("  • Convert a model: ./export_model.sh yolo12n.pt")
    print("  • Run inference: python infer_yolo_litert.py --model model_edgetpu.tflite --image test.jpg")
    print("  • Live stream: python infer_yolo_litert_live.py --model model_edgetpu.tflite")
    print()
    
    return True

def main():
    """Main entry point."""
    try:
        success = check_tpu_connection()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nCheck cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
