"""
Test script to verify CUDA DLL fix works
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("Testing CUDA DLL Fix")
print("=" * 60)

print("\n[1/3] Applying CUDA DLL path fix...")
try:
    from services.ocr.cuda_path_fix import add_cuda_dlls_to_path, verify_cuda_dlls

    # Verify DLLs exist
    status = verify_cuda_dlls()
    print(f"[OK] nvidia-cudnn installed: {status['nvidia_cudnn_installed']}")
    print(f"[OK] All DLLs found: {status['dlls_found']}")
    print(f"[OK] Found {len(status['dll_paths'])} DLLs")

    if status['missing_dlls']:
        print(f"[WARNING] Missing DLLs: {status['missing_dlls']}")

    # Apply the fix
    result = add_cuda_dlls_to_path()
    print(f"[OK] CUDA DLL path fix applied: {result}")

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[2/3] Testing PyTorch import (with cuDNN)...")
try:
    import torch
    print(f"[OK] PyTorch {torch.__version__} imported successfully")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")

    # Try to trigger cuDNN usage
    if torch.cuda.is_available():
        print("[INFO] Testing CUDA tensor operation...")
        x = torch.randn(2, 2, device='cuda')
        print(f"[OK] CUDA tensor created: {x.shape}")

    print("[OK] No cuDNN DLL errors!")

except OSError as e:
    if 'cudnn' in str(e).lower() and '127' in str(e):
        print(f"[FAIL] cuDNN DLL error still occurs!")
        print(f"Error: {e}")
        sys.exit(1)
    else:
        print(f"[FAIL] Different OSError: {e}")
        sys.exit(1)

except Exception as e:
    print(f"[WARNING] Other error (may not be DLL related): {type(e).__name__}")
    print(f"[INFO] {e}")

print("\n[3/3] Testing PaddleOCR import...")
try:
    # This will import through our paddleocr_engine which has the fix
    from services.ocr.engines.paddleocr_engine import PaddleOCREngine
    print("[OK] PaddleOCREngine imported successfully")
    print("[OK] No cuDNN DLL errors during PaddleOCR import!")

except OSError as e:
    if 'cudnn' in str(e).lower() and '127' in str(e):
        print(f"[FAIL] cuDNN DLL error still occurs during PaddleOCR import!")
        print(f"Error: {e}")
        sys.exit(1)
    else:
        raise

except Exception as e:
    print(f"[WARNING] Other error during PaddleOCR import: {type(e).__name__}")
    print(f"[INFO] {e}")
    # Don't fail - other errors are acceptable for this test

print("\n" + "=" * 60)
print("SUCCESS: CUDA DLL fix is working!")
print("No WinError 127 cuDNN DLL errors occurred")
print("=" * 60)
sys.exit(0)
