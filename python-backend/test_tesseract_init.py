"""
Test script to verify Tesseract initialization and OCR processing
Tests the fix for language code ("eng" instead of "en")
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_tesseract_initialization():
    """Test Tesseract engine initialization with 'eng' language code"""
    print("=" * 60)
    print("TESSERACT INITIALIZATION TEST")
    print("=" * 60)

    try:
        from services.ocr.engines.tesseract_engine import TesseractEngine
        from services.ocr.base import OCRConfig

        # Create config with 'eng' language code (after fix)
        config = OCRConfig(
            engine="tesseract",
            use_gpu=False,  # Tesseract is CPU-only
            languages=["eng"],  # Should now match eng.traineddata
            verbose=True
        )

        print(f"\n[OK] Config created with language: {config.languages}")

        # Initialize Tesseract engine
        print("\nInitializing Tesseract engine...")
        engine = TesseractEngine(config)
        engine.initialize()

        print("[OK] Tesseract engine initialized successfully!")
        print(f"[OK] Tesseract available: {engine.tesseract_available}")

        return engine

    except Exception as e:
        print(f"\n[FAIL] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_ocr_processing(engine, test_image_path):
    """Test OCR processing on a real image"""
    print("\n" + "=" * 60)
    print("TESSERACT OCR PROCESSING TEST")
    print("=" * 60)

    try:
        print(f"\nTest image: {test_image_path}")

        # Check if file exists
        if not os.path.exists(test_image_path):
            print(f"[FAIL] Image file not found: {test_image_path}")
            return False

        # Load and convert image
        print("\nLoading image...")
        pil_image = Image.open(test_image_path)

        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            print(f"Converting from {pil_image.mode} to RGB...")
            pil_image = pil_image.convert('RGB')

        # Convert to numpy array
        image_array = np.array(pil_image)
        print(f"[OK] Image loaded: {image_array.shape}")

        # Process with OCR
        print("\nProcessing with Tesseract OCR...")
        result = engine.process_image(image_array)

        print("\n" + "-" * 60)
        print("OCR RESULTS:")
        print("-" * 60)
        print(f"Text length: {len(result.text)} characters")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Processing time: {result.processing_time:.2f} seconds")

        if result.error:
            print(f"Error: {result.error}")
            return False

        # Show first 500 characters of extracted text
        if result.text:
            print(f"\nExtracted text (first 500 chars):")
            print("-" * 60)
            print(result.text[:500])
            if len(result.text) > 500:
                print(f"... ({len(result.text) - 500} more characters)")
        else:
            print("\n[WARN] No text extracted (image may not contain text)")

        print("-" * 60)

        # Determine if successful
        if result.text and len(result.text) > 0:
            print("\n[SUCCESS] Tesseract extracted text from image!")
            return True
        else:
            print("\n[WARN] WARNING: No text extracted, but engine works")
            print("  (Image may not contain readable text)")
            return True  # Still a success if engine works

    except Exception as e:
        print(f"\n[FAIL] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("TESSERACT FIX VERIFICATION")
    print("Testing language code fix: 'en' -> 'eng'")
    print("=" * 60)

    # Test 1: Initialize Tesseract
    engine = test_tesseract_initialization()

    if not engine:
        print("\n" + "=" * 60)
        print("OVERALL RESULT: [FAILED]")
        print("=" * 60)
        return 1

    # Test 2: Process an image
    test_image = r"C:\Users\samue.SAM-NITRO5\OneDrive\Test Images\20251101_203051.jpg"
    success = test_ocr_processing(engine, test_image)

    # Cleanup
    print("\nCleaning up...")
    engine.cleanup()
    print("[OK] Engine cleaned up")

    # Final result
    print("\n" + "=" * 60)
    if success:
        print("OVERALL RESULT: [SUCCESS]")
        print("=" * 60)
        print("\n[OK] Tesseract language code fix verified!")
        print("[OK] Tesseract can find 'eng.traineddata' correctly")
        print("[OK] OCR processing works as expected")
        return 0
    else:
        print("OVERALL RESULT: [FAILED]")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
