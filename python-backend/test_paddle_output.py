"""
Test what PaddleOCR is actually returning
"""
import sys
sys.path.insert(0, '.')

from services.ocr_service import OCRService
from services.pdf_processor import PDFProcessor

pdf_path = r"C:\Users\samue.SAM-NITRO5\Downloads\PDF testing\Photo-bundle.pdf"

print("Initializing OCR...")
ocr = OCRService(gpu=True)

print(f"OCR Engine: {ocr.get_engine_info()}")

print("\nProcessing first page...")
with PDFProcessor(pdf_path) as pdf:
    image = pdf.render_page_to_image(0, dpi=300)

    # Get raw OCR result
    print("\nCalling OCR on page 1...")
    if hasattr(ocr, 'ocr_service') and hasattr(ocr.ocr_service, 'ocr_manager'):
        engine = ocr.ocr_service.ocr_manager.engine
        if hasattr(engine, 'ocr'):
            raw_result = engine.ocr.ocr(image)
            print(f"\nRaw result type: {type(raw_result)}")
            print(f"Raw result length: {len(raw_result) if raw_result else 0}")

            if raw_result and raw_result[0]:
                print(f"\nFirst page has {len(raw_result[0])} text regions")
                print("\nFirst 3 regions:")
                for i, line in enumerate(raw_result[0][:3]):
                    print(f"\n  Region {i+1}:")
                    print(f"    Bbox: {line[0]}")
                    print(f"    Text: '{line[1][0]}'")
                    print(f"    Confidence: {line[1][1]:.2f}")

    # Get processed result
    print("\nProcessed result:")
    text = ocr.extract_text_from_array(image)
    print(f"Text: '{text[:200]}...'")
    print(f"Total length: {len(text)} characters")

ocr.cleanup()
print("\nDone!")
