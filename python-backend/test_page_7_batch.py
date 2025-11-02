"""
Test page 7 through the FULL batch processing pipeline
to see where the text gets lost
"""
import sys
import logging
sys.path.insert(0, '.')

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(message)s'
)

from services.ocr_batch_service import OCRBatchService
from services.pdf_processor import PDFProcessor

pdf_path = r"C:\Users\samue.SAM-NITRO5\Downloads\PDF testing\Photo-bundle.pdf"
test_page = 6  # Page 7 (0-indexed)

print("=" * 80)
print(f"Testing page {test_page + 1} through FULL batch processing pipeline")
print("=" * 80)

# Initialize batch service
service = OCRBatchService(use_gpu=True)

# Open PDF
with PDFProcessor(pdf_path) as pdf:
    # Render page
    print(f"\n1. Rendering page {test_page + 1}...")
    image = pdf.render_page_to_image(test_page, dpi=300)
    print(f"   Image shape: {image.shape}")

    # Process through batch OCR
    print(f"\n2. Processing through batch OCR...")
    texts = service.ocr_service.process_batch([image])
    print(f"   Extracted {len(texts)} text results")
    print(f"   Text length: {len(texts[0])} characters")
    print(f"   Text preview: '{texts[0][:100]}...'")

    # Validate
    print(f"\n3. Validating OCR output...")
    is_valid, reason = service._validate_ocr_output(texts[0], test_page)
    print(f"   Valid: {is_valid}")
    print(f"   Reason: {reason}")

    if not is_valid:
        print(f"\n   VALIDATION FAILED - This is why text was lost!")
        print(f"   Calculating alphanumeric ratio...")
        alphanumeric_count = sum(c.isalnum() for c in texts[0])
        alphanumeric_ratio = alphanumeric_count / len(texts[0]) if len(texts[0]) > 0 else 0
        print(f"   Alphanumeric ratio: {alphanumeric_ratio:.1%}")
        print(f"   Threshold: 30%")

service.cleanup()
print("\nDone!")
