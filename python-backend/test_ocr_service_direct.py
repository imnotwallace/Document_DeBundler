"""
Test OCR Service directly to see if PaddleOCR can initialize through our wrapper
"""
import sys
sys.path.insert(0, '.')

print("=" * 80)
print("OCR Service Direct Test")
print("=" * 80)

# Test 1: Import OCR Service
print("\n[1] Importing OCR Service...")
try:
    from services.ocr_service import OCRService
    print("    [OK] OCR Service imported")
except Exception as e:
    print(f"    [ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Initialize with CPU
print("\n[2] Initializing OCR Service (CPU mode)...")
try:
    ocr = OCRService(gpu=False)
    print("    [OK] OCR Service initialized (CPU)")
except Exception as e:
    print(f"    [ERROR] Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check availability
print("\n[3] Checking OCR availability...")
if ocr.is_available():
    info = ocr.get_engine_info()
    print(f"    [OK] OCR available")
    print(f"    Engine: {info.get('engine', 'unknown')}")
    print(f"    GPU: {info.get('gpu_enabled', False)}")
else:
    print(f"    [ERROR] OCR not available")
    sys.exit(1)

# Test 4: Simple OCR test
print("\n[4] Running simple OCR test...")
try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    # Create test image
    img = Image.new('RGB', (300, 100), color='white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    draw.text((10, 30), "Test OCR 123", fill='black', font=font)

    # Convert to numpy
    img_array = np.array(img)

    # Process with OCR service
    result = ocr.ocr_service.process_image(img_array)

    print(f"    [OK] OCR successful!")
    print(f"    Text: '{result.text}'")
    print(f"    Confidence: {result.confidence:.2f}")

except Exception as e:
    print(f"    [ERROR] OCR test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Try GPU mode
print("\n[5] Testing GPU mode...")
try:
    ocr_gpu = OCRService(gpu=True)
    if ocr_gpu.is_available():
        info_gpu = ocr_gpu.get_engine_info()
        print(f"    [OK] GPU mode available")
        print(f"    Engine: {info_gpu.get('engine', 'unknown')}")
        print(f"    GPU: {info_gpu.get('gpu_enabled', False)}")

        # Quick GPU test
        result_gpu = ocr_gpu.ocr_service.process_image(img_array)
        print(f"    [OK] GPU OCR successful!")
        print(f"    Text: '{result_gpu.text}'")
        print(f"    Confidence: {result_gpu.confidence:.2f}")
    else:
        print(f"    [WARN] GPU mode not available (using CPU fallback)")

except Exception as e:
    print(f"    [ERROR] GPU test failed: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
print("\n[6] Cleanup...")
try:
    ocr.cleanup()
    if 'ocr_gpu' in locals():
        ocr_gpu.cleanup()
    print("    [OK] Cleanup complete")
except Exception as e:
    print(f"    [WARN] Cleanup issue: {e}")

print("\n" + "=" * 80)
print("TEST COMPLETE!")
print("=" * 80)
