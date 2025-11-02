"""
Test script to verify GPU initialization for PaddlePaddle and PyTorch.
Properly initializes DLL paths before importing libraries.
"""
import os
import sys

# Add torch lib directory to DLL search path BEFORE importing any CUDA libraries
torch_lib = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(torch_lib):
    # Add to PATH environment variable (fallback for older Python)
    os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
    # Add to DLL directory (Python 3.8+)
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(torch_lib)
    print(f"Added {torch_lib} to DLL search path")

# Test PyTorch CUDA
print("\n=== Testing PyTorch CUDA ===")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    print("PyTorch GPU: PASS")
except Exception as e:
    print(f"PyTorch GPU: FAIL - {e}")

# Test PaddlePaddle CUDA
print("\n=== Testing PaddlePaddle CUDA ===")
try:
    import paddle
    print(f"PaddlePaddle version: {paddle.__version__}")
    print(f"CUDA compiled: {paddle.device.is_compiled_with_cuda()}")
    print(f"CUDA version: {paddle.version.cuda()}")

    # Run check
    print("\nRunning paddle.utils.run_check()...")
    paddle.utils.run_check()
    print("PaddlePaddle GPU: PASS")
except Exception as e:
    print(f"PaddlePaddle GPU: FAIL - {e}")

# Test sentence-transformers GPU support
print("\n=== Testing sentence-transformers GPU ===")
try:
    from sentence_transformers import SentenceTransformer
    print(f"sentence-transformers can use CUDA: {torch.cuda.is_available()}")
    print("sentence-transformers GPU: PASS")
except Exception as e:
    print(f"sentence-transformers GPU: FAIL - {e}")
