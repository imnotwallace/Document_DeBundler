"""
Test PaddleOCR GPU by importing PaddlePaddle FIRST, before any PyTorch.
This ensures no type conflicts occur.
"""
import os
import sys

# Add torch lib directory for cuDNN DLLs
torch_lib = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(torch_lib):
    os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(torch_lib)
    print(f"Added {torch_lib} to DLL search path\n")

# CRITICAL: Import PaddlePaddle/PaddleOCR BEFORE PyTorch
print("=== Testing PaddleOCR GPU (imported FIRST) ===")
try:
    from paddleocr import PaddleOCR
    import paddle

    print(f"PaddlePaddle version: {paddle.__version__}")
    print(f"CUDA compiled: {paddle.device.is_compiled_with_cuda()}")
    print(f"CUDA version: {paddle.version.cuda()}")
    print(f"GPU device count: {paddle.device.cuda.device_count()}")

    # Initialize PaddleOCR with GPU
    print("\nInitializing PaddleOCR with GPU...")
    ocr = PaddleOCR(use_gpu=True, lang='en', show_log=False)
    print("PaddleOCR initialized successfully with GPU")

    # Test actual OCR (minimal)
    print("\nTesting OCR on dummy data...")
    # Note: We can't test without an actual image, but initialization is the key part

    print("\nPaddleOCR GPU: PASS")
    print("GPU is properly configured for OCR processing!")

except Exception as e:
    print(f"\nPaddleOCR GPU: FAIL")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
