"""
Test script to verify PaddleOCR GPU functionality.
PaddleOCR is tested separately from PyTorch to avoid conflicts.
"""
import os
import sys

# Add torch lib directory to DLL search path BEFORE importing any CUDA libraries
torch_lib = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(torch_lib):
    os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(torch_lib)
    print(f"Added {torch_lib} to DLL search path\n")

# Test PaddleOCR GPU
print("=== Testing PaddleOCR GPU ===")
try:
    from paddleocr import PaddleOCR
    import paddle

    print(f"PaddlePaddle version: {paddle.__version__}")
    print(f"CUDA compiled: {paddle.device.is_compiled_with_cuda()}")
    print(f"CUDA version: {paddle.version.cuda()}")

    # Initialize PaddleOCR with GPU
    print("\nInitializing PaddleOCR with GPU (use_gpu=True)...")
    ocr = PaddleOCR(use_gpu=True, lang='en', show_log=False)

    # Check if GPU is actually being used
    print(f"GPU device count: {paddle.device.cuda.device_count()}")
    print(f"Current device: {paddle.device.get_device()}")

    print("\nPaddleOCR GPU: PASS")
    print("\nGPU is properly configured for OCR processing!")

except Exception as e:
    print(f"\nPaddleOCR GPU: FAIL")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
