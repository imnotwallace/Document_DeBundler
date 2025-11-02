"""
Test script to verify PaddleOCR initialization with PaddlePaddle 3.0
This tests that the set_optimization_level error is resolved.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_paddle_version():
    """Check PaddlePaddle version"""
    try:
        import paddle
        print(f"[OK] PaddlePaddle version: {paddle.__version__}")
        print(f"[OK] CUDA available: {paddle.device.is_compiled_with_cuda()}")
        if paddle.device.is_compiled_with_cuda():
            print(f"[OK] CUDA device count: {paddle.device.cuda.device_count()}")
        return True
    except Exception as e:
        print(f"[FAIL] PaddlePaddle check failed: {e}")
        return False

def test_paddleocr_import():
    """Test PaddleOCR import"""
    try:
        from paddleocr import PaddleOCR
        print("[OK] PaddleOCR imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] PaddleOCR import failed: {e}")
        return False

def test_paddleocr_initialization():
    """Test PaddleOCR initialization (the critical test)"""
    try:
        from paddleocr import PaddleOCR

        print("\nInitializing PaddleOCR with CPU mode...")
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            device='cpu',  # Use CPU to avoid GPU memory issues during test
        )
        print("[OK] PaddleOCR initialized successfully with CPU")

        # Cleanup
        del ocr

        return True
    except AttributeError as e:
        if 'set_optimization_level' in str(e):
            print(f"[FAIL] FAILED: Still getting set_optimization_level error!")
            print(f"   Error: {e}")
            return False
        else:
            print(f"[FAIL] AttributeError (different issue): {e}")
            return False
    except Exception as e:
        print(f"[FAIL] PaddleOCR initialization failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def test_simple_ocr():
    """Test OCR on a simple test image"""
    try:
        from paddleocr import PaddleOCR
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        print("\nCreating test image...")
        # Create a simple test image with text
        img = Image.new('RGB', (300, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.load_default()
        except:
            font = None

        draw.text((10, 40), "Test OCR 123", fill=(0, 0, 0), font=font)

        # Convert to numpy array
        img_array = np.array(img)

        print("Initializing PaddleOCR for test...")
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            device='cpu',
        )

        print("Running OCR on test image...")
        result = ocr.ocr(img_array, cls=True)

        if result and result[0]:
            print(f"[OK] OCR succeeded! Detected {len(result[0])} text regions")
            for line in result[0]:
                if line:
                    bbox, (text, conf) = line
                    print(f"   Text: '{text}' (confidence: {conf:.2f})")
        else:
            print("[OK] OCR ran without error (no text detected, but that's OK)")

        # Cleanup
        del ocr

        return True
    except Exception as e:
        print(f"[FAIL] OCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("PaddleOCR Initialization Test")
    print("Testing fix for set_optimization_level error")
    print("=" * 60)

    results = []

    print("\n[1/4] Checking PaddlePaddle version...")
    results.append(("PaddlePaddle version", test_paddle_version()))

    print("\n[2/4] Testing PaddleOCR import...")
    results.append(("PaddleOCR import", test_paddleocr_import()))

    print("\n[3/4] Testing PaddleOCR initialization (CRITICAL TEST)...")
    results.append(("PaddleOCR init", test_paddleocr_initialization()))

    print("\n[4/4] Testing simple OCR operation...")
    results.append(("Simple OCR", test_simple_ocr()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[OK] SUCCESS: All tests passed! PaddleOCR is working correctly.")
        print("[OK] The set_optimization_level error is FIXED!")
        sys.exit(0)
    else:
        print("\n[FAIL] FAILURE: Some tests failed. Check errors above.")
        sys.exit(1)
