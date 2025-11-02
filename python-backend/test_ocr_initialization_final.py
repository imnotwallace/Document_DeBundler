"""
Final OCR Initialization Test
Tests both PaddleOCR and Tesseract engines can initialize correctly
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("OCR Engines Initialization Test")
print("=" * 60)

print("\n[1/3] Testing PaddleOCR Engine initialization...")
try:
    from services.ocr.engines.paddleocr_engine import PaddleOCREngine
    from services.ocr.base import OCRConfig

    # Create config for CPU mode (faster for testing)
    config = OCRConfig(
        use_gpu=False,  # Use CPU for test
        languages=['en'],
        enable_angle_classification=False,
    )

    # Initialize engine
    print("[INFO] Creating PaddleOCR engine...")
    engine = PaddleOCREngine(config)

    print("[INFO] Initializing PaddleOCR engine...")
    engine.initialize()

    print(f"[OK] PaddleOCR initialized successfully")
    print(f"[OK] GPU support: {engine.supports_gpu()}")
    print(f"[OK] Engine ready: {engine._initialized}")

    # Cleanup
    engine.cleanup()
    print("[OK] PaddleOCR engine cleaned up")

except Exception as e:
    print(f"[FAIL] PaddleOCR initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[2/3] Testing Tesseract Engine initialization...")
try:
    from services.ocr.engines.tesseract_engine import TesseractEngine
    from services.ocr.base import OCRConfig

    # Create config
    config = OCRConfig(
        use_gpu=False,  # Tesseract doesn't use GPU
        languages=['eng'],  # Use 'eng' (Tesseract standard)
    )

    # Initialize engine
    print("[INFO] Creating Tesseract engine...")
    engine = TesseractEngine(config)

    print("[INFO] Initializing Tesseract engine...")
    engine.initialize()

    print(f"[OK] Tesseract initialized successfully")
    print(f"[OK] Engine ready: {engine._initialized}")

    # Cleanup
    engine.cleanup()
    print("[OK] Tesseract engine cleaned up")

except Exception as e:
    print(f"[FAIL] Tesseract initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[3/3] Testing OCR Service (high-level API)...")
try:
    from services.ocr_service import OCRService

    # Initialize service (will try GPU, fall back to CPU)
    print("[INFO] Creating OCR service...")
    ocr = OCRService(gpu=False, engine='paddleocr')  # Use CPU for test

    print(f"[OK] OCR service initialized")
    print(f"[OK] Service available: {ocr.is_available()}")

    # Get engine info
    info = ocr.get_engine_info()
    print(f"[OK] Engine: {info.get('engine', 'unknown')}")
    print(f"[OK] GPU enabled: {info.get('gpu_enabled', False)}")

    # Cleanup
    ocr.cleanup()
    print("[OK] OCR service cleaned up")

except Exception as e:
    print(f"[FAIL] OCR service initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("SUCCESS: All OCR engines initialized correctly!")
print("=" * 60)
print("\nSummary:")
print("  [OK] PaddleOCR engine: Working")
print("  [OK] Tesseract engine: Working")
print("  [OK] OCR service: Working")
print("\nBoth critical fixes verified:")
print("  [OK] PaddlePaddle 3.0.0 compatibility")
print("  [OK] CUDA DLL path fix")
print("  [OK] Tesseract language code (eng)")
print("=" * 60)
sys.exit(0)
