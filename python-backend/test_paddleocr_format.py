"""
Quick test to see PaddleOCR 3.x result format
"""
import sys
import logging
sys.path.insert(0, '.')

logging.basicConfig(level=logging.DEBUG)

from services.ocr_service import OCRService
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Create test image
print("Creating test image...")
img = Image.new('RGB', (800, 200), color='white')
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype("arial.ttf", 48)
except:
    font = ImageFont.load_default()

draw.text((50, 75), "Hello World 123", fill='black', font=font)

# Convert to numpy
img_array = np.array(img)

print("\nInitializing PaddleOCR...")
ocr = OCRService(gpu=True)

if ocr.is_available():
    engine_info = ocr.get_engine_info()
    print(f"Engine: {engine_info['engine']}")
    print(f"GPU: {engine_info['gpu_enabled']}")

    print("\nRunning OCR...")
    # Access the underlying engine directly to see raw results
    if hasattr(ocr, 'ocr_service') and hasattr(ocr.ocr_service, 'ocr_manager'):
        engine = ocr.ocr_service.ocr_manager.engine
        if hasattr(engine, 'ocr'):
            raw_result = engine.ocr.ocr(img_array)
            print(f"\nRaw result type: {type(raw_result)}")
            print(f"Raw result: {raw_result}")

    # Now try through the service
    result = ocr.ocr_service.process_image(img_array)
    print(f"\nProcessed result:")
    print(f"Text: '{result.text}'")
    print(f"Confidence: {result.confidence:.2f}")
else:
    print("OCR not available!")

ocr.cleanup()
print("\nDone!")
