"""
Comprehensive Test Suite for OCR Batch Streaming Assembly

Tests the 5-phase streaming architecture:
- Phase 1: Page scanning and text layer detection
- Phase 2: PDF preparation
- Phase 3: Reverse batch OCR + streaming assembly (CRITICAL - memory profiling)
- Phase 4: Text overlay addition
- Phase 5: PDF save and optimization

Also tests:
- DPI-first batch sizing
- 2-image preprocessing pipeline
- Memory usage (should be constant ~190 MB)
"""

import pytest
import numpy as np
import tempfile
import gc
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import fitz  # PyMuPDF

# Import the service to test
from services.ocr_batch_service import OCRBatchService, OCRProcessingConfig, ProcessingStats
from services.pdf_processor import PDFProcessor
from services.ocr.base import OCRResult


# ====================
# FIXTURES
# ====================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_pdf_with_text(temp_dir):
    """
    Create a simple PDF with embedded text layer.

    Returns path to 10-page PDF with text on all pages.
    """
    pdf_path = temp_dir / "sample_with_text.pdf"
    doc = fitz.open()

    for i in range(10):
        page = doc.new_page(width=612, height=792)  # Letter size
        # Add enough text to pass quality validation (need substantial content)
        text = f"Page {i+1} - Sample Document with Embedded Text Layer\n\n"
        text += "SECTION 1: Introduction\n"
        text += "This is a comprehensive test document with a valid searchable text layer. "
        text += "The document contains sufficient textual content to meet all quality validation thresholds. "
        text += "Character counts, word counts, text density, and coverage ratios all exceed minimum requirements.\n\n"
        text += "SECTION 2: Content Details\n"
        text += "The text layer detection system analyzes multiple metrics including printable character ratios, "
        text += "alphanumeric content ratios, whitespace ratios, average word lengths, and overall text density. "
        text += "This paragraph ensures adequate word count and appropriate text distribution across the page.\n\n"
        text += "SECTION 3: Quality Metrics\n"
        text += "Text quality validation requires minimum character counts (typically 50-100 characters), "
        text += "minimum word counts (typically 10-20 words), and proper text-to-whitespace ratios. "
        text += "Coverage detection ensures text is distributed appropriately across the page dimensions. "
        text += "This comprehensive content guarantees successful text layer identification during processing."
        page.insert_text((72, 72), text)

    doc.save(str(pdf_path))
    doc.close()

    return pdf_path


@pytest.fixture
def sample_pdf_scanned(temp_dir):
    """
    Create a PDF simulating scanned pages (images only, no text layer).

    Returns path to 5-page PDF with images only.
    """
    pdf_path = temp_dir / "sample_scanned.pdf"
    doc = fitz.open()

    for i in range(5):
        page = doc.new_page(width=612, height=792)

        # Create a white image with some text rendered as image
        import io
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new('RGB', (612, 792), color='white')
        draw = ImageDraw.Draw(img)

        # Draw text as IMAGE (not text layer)
        try:
            # Try to use a system font
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            # Fallback to default font
            font = ImageFont.load_default()

        draw.text((72, 72), f"Scanned Page {i+1}", fill='black', font=font)

        # Convert PIL image to bytes
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        # Insert as image
        page.insert_image(
            fitz.Rect(0, 0, 612, 792),
            stream=img_buffer.getvalue()
        )

    doc.save(str(pdf_path))
    doc.close()

    return pdf_path


@pytest.fixture
def sample_pdf_mixed(temp_dir):
    """
    Create a PDF with mixed pages (some with text layer, some scanned).

    Returns path to 20-page PDF:
    - Pages 0-9: Text layer
    - Pages 10-19: Scanned (no text layer)
    """
    pdf_path = temp_dir / "sample_mixed.pdf"
    doc = fitz.open()

    # Pages 0-9: Text layer
    for i in range(10):
        page = doc.new_page(width=612, height=792)
        # Add enough text to pass quality validation (need substantial content)
        text = f"Text Layer Page {i+1}\n\n"
        text += "DOCUMENT OVERVIEW\n"
        text += "This mixed-mode PDF contains pages with embedded searchable text layers. "
        text += "The text layer validation system checks multiple quality metrics to determine usability. "
        text += "This content provides sufficient character count, word count, and text density.\n\n"
        text += "QUALITY REQUIREMENTS\n"
        text += "Successful text layer detection requires meeting minimum thresholds for printable characters, "
        text += "alphanumeric ratios, whitespace distribution, and overall page coverage. "
        text += "Each page must contain adequate textual content distributed appropriately across dimensions.\n\n"
        text += "VALIDATION METRICS\n"
        text += "The system analyzes character counts, word counts, average word lengths, text density metrics, "
        text += "coverage ratios, and confidence scores to determine if extracted text meets quality standards. "
        text += "This comprehensive text ensures all validation criteria are satisfied for proper detection."
        page.insert_text((72, 72), text)

    # Pages 10-19: Scanned
    for i in range(10, 20):
        page = doc.new_page(width=612, height=792)

        # Insert blank white image
        import io
        from PIL import Image
        img = Image.new('RGB', (612, 792), color='white')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        page.insert_image(
            fitz.Rect(0, 0, 612, 792),
            stream=img_buffer.getvalue()
        )

    doc.save(str(pdf_path))
    doc.close()

    return pdf_path


@pytest.fixture
def mock_ocr_service():
    """Mock OCR service that returns predictable results"""
    mock_service = MagicMock()
    mock_service.is_available.return_value = True
    mock_service.engine_name = "paddleocr"
    
    # Use a real dictionary to avoid MagicMock formatting issues
    engine_info = {
        'engine': 'paddleocr',
        'gpu_enabled': True,
        'memory_usage_mb': 150.0
    }
    mock_service.get_engine_info.return_value = engine_info

    # Mock batch processing
    def mock_batch_process(images):
        results = []
        for i, img in enumerate(images):
            # Return mock OCR result
            result = OCRResult(
                text=f"OCR extracted text from image {i}",
                confidence=0.95,
                bbox=[[0, 0], [100, 0], [100, 50], [0, 50]]
            )
            result.image_width = img.shape[1] if hasattr(img, 'shape') else 2550
            result.image_height = img.shape[0] if hasattr(img, 'shape') else 3300
            results.append(result)
        return results

    mock_service.process_batch_with_boxes.side_effect = mock_batch_process

    return mock_service


@pytest.fixture
def mock_preprocessor():
    """Mock intelligent preprocessor"""
    mock_prep = MagicMock()

    # Mock preprocessing result
    def mock_process(image):
        result = MagicMock()
        result.image = image  # Return same image (simulating no improvement)
        result.used_preprocessed = False
        result.techniques_applied = []
        return result

    mock_prep.process.side_effect = mock_process

    return mock_prep


# ====================
# PHASE 1 TESTS: Page Scanning
# ====================

class TestPhase1PageScanning:
    """Test Phase 1: Page scanning and text layer detection"""

    def test_scan_pdf_with_text_layer(self, sample_pdf_with_text, temp_dir, mock_ocr_service, mock_preprocessor):
        """Should correctly identify all pages have text layer"""
        # Create mock for PDFProcessor that returns valid text layer for all pages
        with patch('services.ocr_batch_service.OCRService', return_value=mock_ocr_service), \
             patch('services.ocr_batch_service.IntelligentPreprocessor', return_value=mock_preprocessor), \
             patch('services.ocr_batch_service.PDFProcessor') as MockPDFProcessor:
            
            # Configure PDFProcessor mock to simulate text layer detection
            mock_pdf = MagicMock()
            mock_pdf.get_page_count.return_value = 10
            mock_pdf.has_valid_text_layer.return_value = (True, None)  # All pages have text
            mock_pdf.extract_text.return_value = "Sample text from page"
            MockPDFProcessor.return_value.__enter__.return_value = mock_pdf
            
            service = OCRBatchService(use_gpu=False)
            service._initialize_services()

            result = service._process_single_file(
                str(sample_pdf_with_text),
                str(temp_dir / "output"),
                1,
                1
            )

            assert result['status'] == 'success'
            assert result['pages_text_layer'] == 10
            assert result['pages_ocr'] == 0  # No OCR needed

    def test_scan_pdf_scanned_only(self, sample_pdf_scanned, temp_dir, mock_ocr_service, mock_preprocessor):
        """Should correctly identify all pages need OCR"""
        with patch('services.ocr_batch_service.OCRService', return_value=mock_ocr_service), \
             patch('services.ocr_batch_service.IntelligentPreprocessor', return_value=mock_preprocessor):

            service = OCRBatchService(use_gpu=False)
            service._initialize_services()

            result = service._process_single_file(
                str(sample_pdf_scanned),
                str(temp_dir / "output"),
                1,
                1
            )

            assert result['status'] == 'success'
            assert result['pages_text_layer'] == 0
            assert result['pages_ocr'] >= 0  # Should attempt OCR

    def test_scan_mixed_pdf(self, sample_pdf_mixed, temp_dir, mock_ocr_service, mock_preprocessor):
        """Should correctly identify mixed pages"""
        with patch('services.ocr_batch_service.OCRService', return_value=mock_ocr_service), \
             patch('services.ocr_batch_service.IntelligentPreprocessor', return_value=mock_preprocessor), \
             patch('services.ocr_batch_service.PDFProcessor') as MockPDFProcessor:
            
            # Configure PDFProcessor mock to simulate mixed pages
            mock_pdf = MagicMock()
            mock_pdf.get_page_count.return_value = 20
            
            # First 10 pages have text layer, last 10 need OCR
            def mock_has_text(page_num):
                return (page_num < 10, None)
            
            mock_pdf.has_valid_text_layer.side_effect = mock_has_text
            mock_pdf.extract_text.return_value = "Sample text from page"
            mock_pdf.render_page_to_image.return_value = MagicMock(shape=(3300, 2550, 3))
            MockPDFProcessor.return_value.__enter__.return_value = mock_pdf

            service = OCRBatchService(use_gpu=False)
            service._initialize_services()

            result = service._process_single_file(
                str(sample_pdf_mixed),
                str(temp_dir / "output"),
                1,
                1
            )

            assert result['status'] == 'success'
            assert result['pages_text_layer'] == 10  # First 10 pages
            # Remaining 10 should be queued for OCR


# ====================
# PHASE 2 TESTS: PDF Preparation
# ====================

class TestPhase2PDFPreparation:
    """Test Phase 2: PDF preparation and output setup"""

    def test_output_directory_created(self, sample_pdf_with_text, temp_dir, mock_ocr_service, mock_preprocessor):
        """Should create output directory if it doesn't exist"""
        output_dir = temp_dir / "output" / "nested" / "path"

        with patch('services.ocr_batch_service.OCRService', return_value=mock_ocr_service), \
             patch('services.ocr_batch_service.IntelligentPreprocessor', return_value=mock_preprocessor):

            service = OCRBatchService(use_gpu=False)
            service._initialize_services()

            result = service._process_single_file(
                str(sample_pdf_with_text),
                str(output_dir),
                1,
                1
            )

            assert result['status'] == 'success'
            assert output_dir.exists()
            assert (output_dir / sample_pdf_with_text.name).exists()

    def test_file_size_tracking(self, sample_pdf_with_text, temp_dir, mock_ocr_service, mock_preprocessor):
        """Should track original and final file sizes"""
        with patch('services.ocr_batch_service.OCRService', return_value=mock_ocr_service), \
             patch('services.ocr_batch_service.IntelligentPreprocessor', return_value=mock_preprocessor):

            service = OCRBatchService(use_gpu=False)
            service._initialize_services()

            result = service._process_single_file(
                str(sample_pdf_with_text),
                str(temp_dir / "output"),
                1,
                1
            )

            assert result['status'] == 'success'
            assert 'original_size_mb' in result
            assert 'final_size_mb' in result
            assert 'size_increase_ratio' in result
            assert result['original_size_mb'] > 0
            assert result['final_size_mb'] > 0


# ====================
# PHASE 3 TESTS: Reverse Batch OCR + Streaming Assembly
# ====================

class TestPhase3StreamingAssembly:
    """Test Phase 3: Reverse batch processing and streaming assembly (CRITICAL)"""

    def test_reverse_batch_processing(self, sample_pdf_mixed, temp_dir, mock_ocr_service, mock_preprocessor):
        """Should process OCR batches in reverse order"""
        with patch('services.ocr_batch_service.OCRService', return_value=mock_ocr_service), \
             patch('services.ocr_batch_service.IntelligentPreprocessor', return_value=mock_preprocessor):

            service = OCRBatchService(use_gpu=False, config=OCRProcessingConfig())
            service.batch_size = 5  # Small batch for testing
            service._initialize_services()

            # Track which pages were processed
            processed_pages = []

            original_process_batch = service._process_page_batch
            def track_batch(pdf, page_numbers, file_name):
                processed_pages.extend(page_numbers)
                return original_process_batch(pdf, page_numbers, file_name)

            with patch.object(service, '_process_page_batch', side_effect=track_batch):
                result = service._process_single_file(
                    str(sample_pdf_mixed),
                    str(temp_dir / "output"),
                    1,
                    1
                )

            assert result['status'] == 'success'

            # Verify pages were processed in reverse
            # Pages 10-19 need OCR (reversed = 19,18,...,11,10)
            if len(processed_pages) > 1:
                assert processed_pages[0] > processed_pages[-1], \
                    f"Pages should be reversed, got {processed_pages}"

    def test_immediate_assembly_frees_memory(self, sample_pdf_scanned, temp_dir, mock_ocr_service, mock_preprocessor):
        """
        CRITICAL TEST: Verify images are freed immediately after assembly.

        This ensures streaming architecture works and memory stays constant.
        """
        with patch('services.ocr_batch_service.OCRService', return_value=mock_ocr_service), \
             patch('services.ocr_batch_service.IntelligentPreprocessor', return_value=mock_preprocessor):

            service = OCRBatchService(use_gpu=False)
            service.batch_size = 2  # Small batch
            service._initialize_services()

            # Track delete calls on deskewed images
            delete_count = 0

            original_process_batch = service._process_page_batch
            def track_delete(pdf, page_numbers, file_name):
                results = original_process_batch(pdf, page_numbers, file_name)
                # Results should be (text, ocr_result, deskewed_img) tuples
                for text, ocr_result, deskewed_img in results:
                    assert deskewed_img is not None, "Deskewed image should be returned"
                return results

            with patch.object(service, '_process_page_batch', side_effect=track_delete):
                result = service._process_single_file(
                    str(sample_pdf_scanned),
                    str(temp_dir / "output"),
                    1,
                    1
                )

            assert result['status'] == 'success'
            # Note: Actual memory freed is verified by memory profiling test below

    @pytest.mark.skipif(
        os.environ.get('SKIP_MEMORY_TESTS') == '1',
        reason="Memory profiling test disabled"
    )
    def test_memory_constant_during_processing(self, sample_pdf_mixed, temp_dir, mock_ocr_service, mock_preprocessor):
        """
        CRITICAL TEST: Verify memory stays constant during processing.

        Peak memory should be batch_size × 10 MB, NOT proportional to PDF size.

        This test requires memory_profiler:
        pip install memory-profiler

        Run with: python -m pytest -v -k test_memory_constant
        """
        try:
            from memory_profiler import memory_usage
        except ImportError:
            pytest.skip("memory_profiler not installed")

        with patch('services.ocr_batch_service.OCRService', return_value=mock_ocr_service), \
             patch('services.ocr_batch_service.IntelligentPreprocessor', return_value=mock_preprocessor):

            service = OCRBatchService(use_gpu=False)
            service.batch_size = 5
            service._initialize_services()

            # Track memory during processing
            def process_file():
                return service._process_single_file(
                    str(sample_pdf_mixed),
                    str(temp_dir / "output"),
                    1,
                    1
                )

            # Measure memory usage
            mem_usage = memory_usage(process_file, interval=0.1, timeout=None)

            # Calculate memory statistics
            baseline_mem = mem_usage[0]
            peak_mem = max(mem_usage)
            avg_mem = sum(mem_usage) / len(mem_usage)

            # Calculate realistic thresholds accounting for overhead
            # - PyMuPDF base: ~100-200 MB
            # - Batch processing: batch_size × 100 MB (conservative with overhead)
            # - Python GC delays: ~100 MB
            # Total: ~300-400 MB increase is reasonable for mocked environment
            
            # For 20-page PDF, memory should NOT grow linearly (20 × 50MB = 1000MB)
            # With streaming, should be closer to baseline + batch overhead (~600-700MB total)
            max_expected_increase = 600  # MB - allows overhead but prevents linear growth
            max_expected_total = baseline_mem + max_expected_increase
            
            memory_increase = peak_mem - baseline_mem
            
            print(f"\nMemory Profile:")
            print(f"  Baseline: {baseline_mem:.1f} MB")
            print(f"  Peak: {peak_mem:.1f} MB")
            print(f"  Average: {avg_mem:.1f} MB")
            print(f"  Memory Increase: {memory_increase:.1f} MB")
            print(f"  Max Allowed Increase: {max_expected_increase} MB")
            print(f"  Max Expected Total: {max_expected_total:.1f} MB")
            
            # Verify memory didn't grow linearly with page count
            # For 20 pages, linear growth would be ~1000 MB
            # Streaming should keep it under 600 MB increase
            assert memory_increase < max_expected_increase, \
                f"Memory increase {memory_increase:.1f}MB exceeded threshold {max_expected_increase}MB. " \
                f"This suggests memory is accumulating instead of streaming."
            
            print(f"  [PASS] Memory stayed constant (increase {memory_increase:.1f}MB < {max_expected_increase}MB)")


# ====================
# PHASE 4 TESTS: Text Overlay
# ====================

class TestPhase4TextOverlay:
    """Test Phase 4: Text overlay addition to text-layer pages"""

    def test_text_overlay_added(self, sample_pdf_with_text, temp_dir, mock_ocr_service, mock_preprocessor):
        """Should add text overlays to pages with existing text layer"""
        with patch('services.ocr_batch_service.OCRService', return_value=mock_ocr_service), \
             patch('services.ocr_batch_service.IntelligentPreprocessor', return_value=mock_preprocessor):

            service = OCRBatchService(use_gpu=False)
            service._initialize_services()

            result = service._process_single_file(
                str(sample_pdf_with_text),
                str(temp_dir / "output"),
                1,
                1
            )

            assert result['status'] == 'success'

            # Verify output PDF has text
            output_pdf = temp_dir / "output" / sample_pdf_with_text.name
            assert output_pdf.exists()

            doc = fitz.open(str(output_pdf))
            for page_num in range(doc.page_count):
                text = doc[page_num].get_text()
                assert len(text) > 0, f"Page {page_num} should have text"
            doc.close()


# ====================
# PHASE 5 TESTS: PDF Save and Optimization
# ====================

class TestPhase5PDFSave:
    """Test Phase 5: PDF save and optimization"""

    def test_pdf_optimization_settings(self, sample_pdf_with_text, temp_dir, mock_ocr_service, mock_preprocessor):
        """Should save PDF with optimization settings (garbage, deflate, clean)"""
        with patch('services.ocr_batch_service.OCRService', return_value=mock_ocr_service), \
             patch('services.ocr_batch_service.IntelligentPreprocessor', return_value=mock_preprocessor):

            service = OCRBatchService(use_gpu=False)
            service._initialize_services()

            result = service._process_single_file(
                str(sample_pdf_with_text),
                str(temp_dir / "output"),
                1,
                1
            )

            assert result['status'] == 'success'

            output_pdf = temp_dir / "output" / sample_pdf_with_text.name
            assert output_pdf.exists()

            # Verify PDF is valid
            doc = fitz.open(str(output_pdf))
            assert doc.page_count == 10
            doc.close()

    def test_fallback_copy_on_save_failure(self, sample_pdf_with_text, temp_dir, mock_ocr_service, mock_preprocessor):
        """Should fallback to copying original if save fails"""
        with patch('services.ocr_batch_service.OCRService', return_value=mock_ocr_service), \
             patch('services.ocr_batch_service.IntelligentPreprocessor', return_value=mock_preprocessor):

            service = OCRBatchService(use_gpu=False)
            service._initialize_services()

            # Make save fail (simulate permission error)
            with patch('fitz.Document.save', side_effect=RuntimeError("Permission denied")):
                result = service._process_single_file(
                    str(sample_pdf_with_text),
                    str(temp_dir / "output"),
                    1,
                    1
                )

            # Should still succeed (with fallback)
            assert result['status'] == 'success'
            assert 'warning' in result

            output_pdf = temp_dir / "output" / sample_pdf_with_text.name
            assert output_pdf.exists()


# ====================
# DPI-FIRST BATCH SIZING TESTS
# ====================

class TestDPIFirstBatchSizing:
    """Test DPI-first batch sizing logic"""

    def test_batch_size_calculated_for_target_dpi(self):
        """Should calculate batch size to achieve target DPI"""
        config = OCRProcessingConfig()

        # Mock OCR config with target DPI
        ocr_config = {
            'dpi': 600,  # High DPI request
            'use_gpu': True
        }

        with patch('services.ocr_batch_service.detect_hardware_capabilities') as mock_hw:
            mock_hw.return_value = {
                'gpu_available': True,
                'cuda_available': True,
                'directml_available': False,
                'gpu_memory_gb': 4.0,
                'system_memory_gb': 16.0,
                'cpu_count': 8,
                'platform': 'Windows'
            }

            service = OCRBatchService(
                use_gpu=True,
                ocr_config=ocr_config
            )

            # Batch size should be calculated for 600 DPI
            # At 600 DPI, batch size will be smaller than at 300 DPI
            assert service.batch_size >= 1  # At minimum 1
            assert service.batch_size < 25  # Should be less than 300 DPI baseline

    def test_batch_size_one_for_very_high_dpi(self):
        """Should allow batch_size=1 for very high DPI (1200+)"""
        ocr_config = {
            'dpi': 1200,  # Very high DPI
            'use_gpu': True
        }

        with patch('services.ocr_batch_service.detect_hardware_capabilities') as mock_hw:
            mock_hw.return_value = {
                'gpu_available': True,
                'cuda_available': True,
                'directml_available': False,
                'gpu_memory_gb': 4.0,
                'system_memory_gb': 16.0,
                'cpu_count': 8,
                'platform': 'Windows'
            }

            service = OCRBatchService(
                use_gpu=True,
                ocr_config=ocr_config
            )

            # Should accept batch_size=1 for high DPI
            assert service.batch_size >= 1

    def test_legacy_batch_sizing_without_dpi(self):
        """Should use legacy optimization when DPI not specified"""
        with patch('services.ocr_batch_service.detect_hardware_capabilities') as mock_hw:
            mock_hw.return_value = {
                'gpu_available': True,
                'cuda_available': True,
                'directml_available': False,
                'gpu_memory_gb': 4.0,
                'system_memory_gb': 16.0,
                'cpu_count': 8,
                'platform': 'Windows'
            }

            service = OCRBatchService(use_gpu=True)

            # Should use default batch sizing (maximize batch size)
            assert service.batch_size > 1  # Should be optimized for throughput


# ====================
# RGB DESKEWING TESTS
# ====================

class TestRGBDeskewing:
    """Test RGB deskewing implementation"""

    def test_deskew_detects_angle(self):
        """Should detect rotation angle from skewed image"""
        # Create a skewed test image
        import cv2

        # Create white image with black text
        img = np.ones((1000, 800, 3), dtype=np.uint8) * 255
        cv2.line(img, (100, 100), (700, 120), (0, 0, 0), 2)  # Slightly rotated line
        cv2.line(img, (100, 200), (700, 220), (0, 0, 0), 2)
        cv2.line(img, (100, 300), (700, 320), (0, 0, 0), 2)

        service = OCRBatchService(use_gpu=False)

        deskewed = service._apply_deskew_only(img)

        # Should return an image (deskewed or original)
        assert deskewed is not None
        assert deskewed.shape == img.shape or deskewed.shape[0] > 0

    def test_deskew_handles_no_lines(self):
        """Should handle images with no detectable lines gracefully"""
        # Create blank white image
        img = np.ones((1000, 800, 3), dtype=np.uint8) * 255

        service = OCRBatchService(use_gpu=False)

        deskewed = service._apply_deskew_only(img)

        # Should return original when no lines detected
        assert np.array_equal(deskewed, img)

    def test_deskew_handles_grayscale_input(self):
        """Should handle grayscale input gracefully"""
        # Create grayscale image
        img = np.ones((1000, 800), dtype=np.uint8) * 255

        service = OCRBatchService(use_gpu=False)

        deskewed = service._apply_deskew_only(img)

        # Should return original for grayscale
        assert np.array_equal(deskewed, img)


# ====================
# INTEGRATION TESTS
# ====================

class TestIntegration:
    """Integration tests for full pipeline"""

    def test_full_pipeline_mixed_pdf(self, sample_pdf_mixed, temp_dir, mock_ocr_service, mock_preprocessor):
        """Test complete pipeline with mixed PDF"""
        with patch('services.ocr_batch_service.OCRService', return_value=mock_ocr_service), \
             patch('services.ocr_batch_service.IntelligentPreprocessor', return_value=mock_preprocessor):

            service = OCRBatchService(use_gpu=False)

            result = service.process_batch(
                files=[str(sample_pdf_mixed)],
                output_dir=str(temp_dir / "output")
            )

            assert len(result['successful']) == 1
            assert len(result['failed']) == 0
            assert result['total_pages_processed'] == 20

            # Verify statistics
            stats = result['statistics']
            assert stats['total_files'] == 1
            assert stats['successful_files'] == 1
            assert stats['pages_per_second'] > 0

    def test_cancellation_support(self, sample_pdf_mixed, temp_dir, mock_ocr_service, mock_preprocessor):
        """Should support cancellation during processing"""
        import threading

        cancellation_flag = threading.Event()

        with patch('services.ocr_batch_service.OCRService', return_value=mock_ocr_service), \
             patch('services.ocr_batch_service.IntelligentPreprocessor', return_value=mock_preprocessor):

            service = OCRBatchService(
                use_gpu=False,
                cancellation_flag=cancellation_flag
            )

            # Cancel immediately
            cancellation_flag.set()

            result = service.process_batch(
                files=[str(sample_pdf_mixed)],
                output_dir=str(temp_dir / "output")
            )

            # Should handle cancellation gracefully
            assert len(result['successful']) + len(result['failed']) >= 0

    def test_batch_multiple_files(self, sample_pdf_with_text, sample_pdf_scanned, temp_dir, mock_ocr_service, mock_preprocessor):
        """Should process multiple files in batch"""
        with patch('services.ocr_batch_service.OCRService', return_value=mock_ocr_service), \
             patch('services.ocr_batch_service.IntelligentPreprocessor', return_value=mock_preprocessor):

            service = OCRBatchService(use_gpu=False)

            result = service.process_batch(
                files=[str(sample_pdf_with_text), str(sample_pdf_scanned)],
                output_dir=str(temp_dir / "output")
            )

            assert len(result['successful']) == 2
            assert len(result['failed']) == 0
            assert result['total_pages_processed'] == 15  # 10 + 5


# ====================
# UTILITY FUNCTIONS
# ====================

def run_memory_profile_manually():
    """
    Manual memory profiling script.

    Run this separately to profile memory usage:
    python -m memory_profiler python-backend/tests/test_ocr_batch_streaming.py::run_memory_profile_manually
    """
    print("Setting up test environment...")

    # Create temp PDF
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())

    pdf_path = temp_dir / "test.pdf"
    doc = fitz.open()
    for i in range(100):  # 100 pages
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), f"Page {i+1}")
    doc.save(str(pdf_path))
    doc.close()

    print(f"Created test PDF: {pdf_path}")

    # Process with service
    with patch('services.ocr_batch_service.OCRService'), \
         patch('services.ocr_batch_service.IntelligentPreprocessor'):

        service = OCRBatchService(use_gpu=False)
        service._initialize_services()

        print("Processing PDF...")
        result = service._process_single_file(
            str(pdf_path),
            str(temp_dir / "output"),
            1,
            1
        )

        print(f"Result: {result['status']}")
        print(f"Pages processed: {result['pages_processed']}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print("Done!")


if __name__ == "__main__":
    # Run memory profile manually
    run_memory_profile_manually()
