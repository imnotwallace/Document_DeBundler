"""
Test PaddleOCR 3.x predict() API
"""
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
import fitz

# Load image
pdf_path = r"C:\Users\samue.SAM-NITRO5\Downloads\PDF testing\Photo-bundle.pdf"
doc = fitz.open(pdf_path)
page = doc[0]
pix = page.get_pixmap(dpi=300)
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
img_array = np.array(img)

print(f"Image size: {img_array.shape}")

# Initialize PaddleOCR
print("Initializing PaddleOCR...")
ocr = PaddleOCR(use_angle_cls=True, lang='en', device='gpu')

# Test predict() API
print("Testing predict() API...")
result = ocr.predict(img_array)

print(f"Result type: {type(result)}")
print(f"Result keys: {result.keys() if hasattr(result, 'keys') else 'Not a dict'}")
print()
print("Full result:")
print(result)

doc.close()
