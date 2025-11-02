"""
Test OCR processing with real PDF file
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, '.')

from services.ocr_batch_service import OCRBatchService

# Input file
input_pdf = r"C:\Users\samue.SAM-NITRO5\Downloads\PDF testing\Photo-bundle.pdf"

# Output directory
output_dir = r"F:\Document-De-Bundler\testing_output"
os.makedirs(output_dir, exist_ok=True)

print(f"Testing real PDF: {input_pdf}")
print(f"Output directory: {output_dir}")
print("=" * 80)

# Check if file exists
if not os.path.exists(input_pdf):
    print(f"ERROR: File not found: {input_pdf}")
    sys.exit(1)

# Get file size
file_size_mb = os.path.getsize(input_pdf) / (1024 * 1024)
print(f"File size: {file_size_mb:.2f} MB")

# Process with OCR batch service
print("\nInitializing OCR batch service...")
service = OCRBatchService(use_gpu=True)  # Use GPU with PaddleOCR

# Show engine info
if hasattr(service, 'ocr_service') and service.ocr_service:
    engine_info = service.ocr_service.get_engine_info()
    print(f"OCR Engine: {engine_info.get('engine', 'unknown')}")
    print(f"GPU Enabled: {engine_info.get('gpu_enabled', False)}")

try:
    print("\nStarting OCR processing...")
    print("This may take several minutes for large PDFs...")

    result = service.process_batch(
        files=[input_pdf],
        output_dir=output_dir
    )

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE!")
    print("=" * 80)

    # Display results
    if result['successful']:
        file_result = result['successful'][0]
        print(f"\n[SUCCESS] SUCCESS")
        print(f"   Input:  {file_result['file']}")
        print(f"   Output: {file_result['output_path']}")
        print(f"\n   Statistics:")
        print(f"   - Pages processed:    {file_result.get('pages_processed', 'N/A')}")
        print(f"   - Pages with text layer: {file_result.get('pages_text_layer', 0)}")
        print(f"   - Pages OCR'd:        {file_result.get('pages_ocr', 0)}")

        print(f"\n   Output file saved at:")
        print(f"   {file_result['output_path']}")

    if result['failed']:
        print(f"\n[FAILED] FAILED")
        for failure in result['failed']:
            print(f"   File: {failure['file']}")
            print(f"   Error: {failure['error']}")

    print("\n" + "=" * 80)

finally:
    service.cleanup()
    print("\nOCR service cleaned up.")

print("\nTest complete! Check the output file in:")
print(f"   {output_dir}")
