"""
Test page 7 with debug logging to see what's happening
"""
import sys
import logging
import os
sys.path.insert(0, '.')

# Enable DEBUG logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(message)s'
)

from services.ocr_batch_service import OCRBatchService

pdf_path = r"C:\Users\samue.SAM-NITRO5\Downloads\PDF testing\Photo-bundle.pdf"
output_dir = r"F:\Document-De-Bundler\testing_output\debug"
os.makedirs(output_dir, exist_ok=True)

print("Testing page 7 with DEBUG logging enabled")
print("=" * 80)

service = OCRBatchService(use_gpu=True)

try:
    result = service.process_batch(
        files=[pdf_path],
        output_dir=output_dir
    )

    print("\nRESULTS:")
    print("=" * 80)
    if result['successful']:
        file_result = result['successful'][0]
        print(f"SUCCESS: {file_result['file']}")
        print(f"Pages processed: {file_result.get('pages_processed', 'N/A')}")
        print(f"Pages OCR'd: {file_result.get('pages_ocr', 0)}")
        print(f"Output: {file_result['output_path']}")

    if result['failed']:
        print(f"FAILED: {result['failed']}")

finally:
    service.cleanup()

print("\nDone! Check logs above for debug output.")
