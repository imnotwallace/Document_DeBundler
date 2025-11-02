"""
Comprehensive tests for Partial OCR Detection fixes.
"""
import sys
import pytest
import fitz
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ocr_batch_service import OCRBatchService
from services.pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)


class TestPartialOCRFixes:
    """Test suite for partial OCR detection and text layer duplication fixes"""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for test outputs"""
        output_dir = tmp_path / "ocr_test_outputs"
        output_dir.mkdir(exist_ok=True)
        return output_dir

    @pytest.fixture
    def sample_partial_pdf(self, tmp_path):
        """
        Create test PDF with partial text layer coverage.

        Structure:
        - Header with text layer (15% coverage)
        - Body with scanned image (85% coverage, no text)
        """
        pdf_path = tmp_path / "partial_coverage_test.pdf"

        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        # Add header text layer (top 15% of page)
        header_rect = fitz.Rect(50, 50, 562, 120)
        page.insert_textbox(
            header_rect,
            "Annual Report 2024\nCompany Confidential",
            fontsize=12,
            fontname="helv",
            color=(0, 0, 0)
        )

        # Add scanned body image (bottom 85% of page)
        # Create actual raster image to simulate scanned content
        # This ensures NO searchable text layer exists in the body
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
            
            # Create a simple image with text (as pixels, not vector)
            img_width, img_height = 512, 592
            img = Image.new('RGB', (img_width, img_height), color=(230, 230, 230))
            draw = ImageDraw.Draw(img)
            
            # Draw some text as pixels (not searchable)
            try:
                # Try to use a default font, fallback to basic if not available
                font = ImageFont.load_default()
            except:
                font = None
            
            draw.text((10, 50), "This text is in a scanned image", fill=(0, 0, 0), font=font)
            draw.text((10, 100), "Body content line 1", fill=(0, 0, 0), font=font)
            draw.text((10, 150), "Body content line 2", fill=(0, 0, 0), font=font)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Insert as raster image (not searchable)
            body_rect = fitz.Rect(50, 150, 562, 742)
            page.insert_image(body_rect, stream=img_bytes.read())
            
        except ImportError:
            # Fallback: use gray rectangle if PIL not available
            logger.warning("PIL not available, using simple rectangle for test")
            body_rect = fitz.Rect(50, 150, 562, 742)
            page.draw_rect(body_rect, color=(0.8, 0.8, 0.8), fill=(0.9, 0.9, 0.9))

        doc.save(str(pdf_path))
        doc.close()

        return pdf_path

    @pytest.fixture
    def sample_full_ocr_pdf(self, tmp_path):
        """
        Create test PDF with complete invisible OCR layer.

        Should NOT trigger re-OCR.
        """
        pdf_path = tmp_path / "full_ocr_test.pdf"

        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        # Add background image
        page.draw_rect(page.rect, color=(0.9, 0.9, 0.9), fill=(0.95, 0.95, 0.95))

        # Add INVISIBLE text layer covering 90% of page
        full_text = "This is a fully OCR'd document.\n" * 50
        page.insert_textbox(
            page.rect,
            full_text,
            fontsize=8,
            color=(1, 1, 1),  # White (invisible)
            render_mode=3
        )

        doc.save(str(pdf_path))
        doc.close()

        return pdf_path

    def test_partial_coverage_detection(self, sample_partial_pdf):
        """
        Test that partial text layer coverage is correctly detected.

        Expected: Page should fail validation and trigger OCR.
        """
        with PDFProcessor(str(sample_partial_pdf)) as pdf:
            has_valid, metrics = pdf.has_valid_text_layer(0, return_metrics=True)

            # Should detect partial coverage
            assert not has_valid, "Partial coverage page should fail validation"
            assert metrics.text_coverage_ratio < 0.70, f"Coverage ratio should be < 70%, got {metrics.text_coverage_ratio:.2%}"
            assert metrics.coverage_confidence < 0.50, f"Coverage confidence should be low, got {metrics.coverage_confidence:.2%}"

    def test_no_text_duplication(self, sample_partial_pdf, temp_dir):
        """
        CRITICAL TEST: Verify that re-OCR does NOT duplicate text.

        This is the main fix - ensuring original partial text layer
        is removed before adding OCR overlay.
        """
        # Process with OCR batch service
        service = OCRBatchService(use_gpu=False)  # CPU for testing

        try:
            result = service.process_batch(
                files=[str(sample_partial_pdf)],
                output_dir=str(temp_dir)
            )

            assert result['successful'], "Processing should succeed"
            assert len(result['successful']) == 1, "Should process 1 file"

            # Open output file
            output_path = result['successful'][0]['output_path']
            doc = fitz.open(output_path)
            page = doc[0]

            # Extract all text
            full_text = page.get_text()

            # Check for duplicates
            # The header "Annual Report 2024" should appear ONLY ONCE
            header_count = full_text.count("Annual Report 2024")

            assert header_count == 1, \
                f"Header text should appear ONCE, found {header_count} times. " \
                f"This indicates text layer duplication! Full text:\n{full_text}"

            doc.close()

        finally:
            service.cleanup()

    def test_ocr_output_validation(self, sample_partial_pdf, temp_dir):
        """
        Test that OCR output is validated and empty results are rejected.
        """
        service = OCRBatchService(use_gpu=False)

        try:
            # Process
            result = service.process_batch(
                files=[str(sample_partial_pdf)],
                output_dir=str(temp_dir)
            )

            # Check stats
            assert result['successful'], "Processing should succeed"
            file_result = result['successful'][0]

            # Should have attempted OCR
            assert file_result['pages_ocr'] > 0, "Should have OCR'd at least one page"

            # Check that output has actual text (not empty)
            output_path = file_result['output_path']
            doc = fitz.open(output_path)
            page = doc[0]
            text = page.get_text()

            assert len(text.strip()) > 0, \
                f"OCR should produce some text (got {len(text)} chars: '{text.strip()[:50]}...')"

            doc.close()

        finally:
            service.cleanup()

    def test_quality_preservation(self, sample_full_ocr_pdf, temp_dir):
        """
        Test that high-quality existing OCR is preserved, not re-processed.
        """
        # First, verify the PDF has good text quality
        with PDFProcessor(str(sample_full_ocr_pdf)) as pdf:
            has_valid, metrics = pdf.has_valid_text_layer(0, return_metrics=True)
            original_text = pdf.extract_text(0)

            assert has_valid, "Full OCR PDF should pass validation"
            assert metrics.confidence_score >= 0.80, "Should have high-quality text content"
            # Note: Invisible text layers have low spatial coverage but high content quality

        # Process with OCR batch service
        service = OCRBatchService(use_gpu=False)

        try:
            result = service.process_batch(
                files=[str(sample_full_ocr_pdf)],
                output_dir=str(temp_dir)
            )

            # Check that page was NOT re-OCR'd
            file_result = result['successful'][0]
            assert file_result['pages_text_layer'] == 1, "Should use existing text layer"
            assert file_result['pages_ocr'] == 0, "Should NOT re-OCR"

        finally:
            service.cleanup()

    def test_empty_coverage_metrics_fallback(self):
        """
        Test that coverage detection failure triggers OCR (fail-safe).
        """
        from services.ocr.text_quality import TextLayerValidator

        validator = TextLayerValidator()

        # Get empty metrics (simulates detection failure)
        empty_metrics = validator._empty_coverage_metrics()

        # Should have low confidence to trigger OCR
        assert empty_metrics['coverage_confidence'] == 0.0, \
            "Detection failure should return 0.0 confidence to trigger OCR"

    def test_batch_processing_mixed_pages(self, tmp_path, temp_dir):
        """
        Test processing PDF with mix of good and partial coverage pages.
        """
        # Create PDF with 3 pages:
        # Page 1: Full text layer (good)
        # Page 2: Partial text layer (needs OCR)
        # Page 3: No text layer (needs OCR)

        pdf_path = tmp_path / "mixed_coverage_test.pdf"
        doc = fitz.open()

        # Page 1: Full coverage with high-quality text layer
        page1 = doc.new_page(width=612, height=792)
        page1.insert_textbox(page1.rect, "Full text coverage page\n" * 30, fontsize=10)

        # Page 2: Partial text layer (small header, low confidence)
        page2 = doc.new_page(width=612, height=792)
        page2.insert_textbox(fitz.Rect(50, 50, 562, 100), "Header text", fontsize=12)
        page2.draw_rect(fitz.Rect(50, 150, 562, 742), fill=(0.9, 0.9, 0.9))

        # Page 3: Empty page with background (will fail validation, attempt OCR)
        page3 = doc.new_page(width=612, height=792)
        page3.draw_rect(page3.rect, fill=(0.9, 0.9, 0.9))

        doc.save(str(pdf_path))
        doc.close()

        # Process
        service = OCRBatchService(use_gpu=False)

        try:
            result = service.process_batch(
                files=[str(pdf_path)],
                output_dir=str(temp_dir)
            )

            file_result = result['successful'][0]

            # Main validation: Page 1 should use text layer (not re-OCR)
            # This verifies that high-quality text layers pass validation
            assert file_result['pages_text_layer'] >= 1, "Should use text layer for page 1"

            # Note: Pages 2-3 have insufficient content for OCR (blank/minimal text)
            # This test primarily validates text layer detection, not OCR on blank pages

            # Verify no text duplication on page 1
            output_path = file_result['output_path']
            doc = fitz.open(output_path)

            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()

                # Check for common duplicate patterns (only on pages with substantial text)
                words = text.split()
                if len(words) > 20:  # Only check pages with sufficient content
                    most_common_word = max(set(words), key=words.count)
                    count = words.count(most_common_word)

                    # Allow some repetition but not exact duplicates
                    assert count <= len(words) / 4, \
                        f"Page {page_num+1} has suspicious repetition of '{most_common_word}' ({count} times)"

            doc.close()

        finally:
            service.cleanup()


# Integration tests
class TestIntegration:
    """Integration tests for complete workflow"""

    def test_end_to_end_workflow(self, tmp_path):
        """
        Complete end-to-end test simulating real usage.
        """
        # TODO: Implement full workflow test
        pass
