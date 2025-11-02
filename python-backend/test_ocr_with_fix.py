"""
Test OCR initialization with explicit cuda_path_fix import
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("OCR Initialization Test with CUDA Fix")
print("=" * 60)

# IMPORTANT: Import CUDA path fix FIRST, BEFORE any paddle/paddleocr imports
print("\n[Step 0] Importing CUDA path fix...")
from services.ocr.cuda_path_fix import add_cuda_dlls_to_path
add_cuda_dlls_to_path()
print("[OK] CUDA paths added to process PATH\n")

print("[Step 1] Pre-importing paddle...")
try:
    import paddle
    print(f"[OK] Paddle {paddle.__version__} imported\n")
except Exception as e:
    print(f"[FAIL] Paddle import failed: {e}")
    sys.exit(1)

print("[2/3] Importing PaddleOCR...")
try:
    from paddleocr import PaddleOCR
    print("[OK] PaddleOCR imported")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

print("\n[3/3] Initializing PaddleOCR...")
try:
    ocr = PaddleOCR(
        use_textline_orientation=False,  # Using new parameter name
        lang='en',
        device='cpu',  # Explicit CPU mode
    )
    print("[OK] PaddleOCR initialized successfully in CPU mode")

    # Cleanup
    del ocr
    print("[OK] Cleanup complete")

    print("\n" + "=" * 60)
    print("SUCCESS: PaddleOCR works with explicit cuda_path_fix import!")
    print("=" * 60)
    sys.exit(0)

except Exception as e:
    print(f"[FAIL] Initialization failed: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
