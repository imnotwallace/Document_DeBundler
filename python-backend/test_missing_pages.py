"""
Test the specific pages that showed NO TEXT FOUND: 7, 9, 11, 12, 15
"""
import sys
import logging
sys.path.insert(0, '.')

logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

from services.ocr_service import OCRService
from services.pdf_processor import PDFProcessor

pdf_path = r"C:\Users\samue.SAM-NITRO5\Downloads\PDF testing\Photo-bundle.pdf"
missing_pages = [6, 8, 10, 11, 14]  # 0-indexed: pages 7, 9, 11, 12, 15

print("Testing pages that showed NO TEXT FOUND")
print("=" * 80)

ocr = OCRService(gpu=True)

with PDFProcessor(pdf_path) as pdf:
    for page_idx in missing_pages:
        print(f"\nPage {page_idx + 1}:")
        try:
            image = pdf.render_page_to_image(page_idx, dpi=300)
            text = ocr.extract_text_from_array(image)
            print(f"  Text length: {len(text)} chars")
            if text:
                print(f"  Preview: '{text[:60]}...'")
            else:
                print(f"  NO TEXT EXTRACTED!")
        except Exception as e:
            print(f"  EXCEPTION: {e}")

ocr.cleanup()
print("\nDone!")
