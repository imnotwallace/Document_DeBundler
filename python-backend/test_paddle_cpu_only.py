"""
Test PaddleOCR initialization in CPU-only mode
"""

import os
import sys
from pathlib import Path

# Force CPU mode BEFORE any paddle imports
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide all CUDA devices
os.environ['CPU_NUM'] = '1'  # Use single CPU thread

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("PaddleOCR CPU-Only Mode Test")
print("=" * 60)

print("\n[INFO] Environment set for CPU-only mode")
print(f"CUDA_VISIBLE_DEVICES: '{os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}'")

print("\n[1/2] Importing PaddleOCR...")
try:
    from paddleocr import PaddleOCR
    print("[OK] PaddleOCR imported")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

print("\n[2/2] Initializing PaddleOCR in CPU mode...")
try:
    ocr = PaddleOCR(
        use_angle_cls=False,
        lang='en',
        device='cpu',  # Explicit CPU mode
    )
    print("[OK] PaddleOCR initialized successfully in CPU mode")

    # Cleanup
    del ocr
    print("[OK] Cleanup complete")

    print("\n" + "=" * 60)
    print("SUCCESS: CPU-only mode works!")
    print("=" * 60)
    sys.exit(0)

except Exception as e:
    print(f"[FAIL] Initialization failed: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
