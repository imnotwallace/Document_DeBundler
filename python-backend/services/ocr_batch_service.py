"""
OCR Batch Service
Production-ready OCR batch processing with adaptive memory management,
error recovery, and progress tracking for large PDF processing.
"""

import logging
import time
import gc
import hashlib
import threading
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from .ocr_service import OCRService
from .pdf_processor import PDFProcessor
from .ocr.config import detect_hardware_capabilities, get_optimal_batch_size
from .ocr.vram_monitor import VRAMMonitor

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for batch processing"""
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_pages_processed: int = 0
    total_pages_ocr: int = 0
    total_pages_text_layer: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        duration = self.end_time - self.start_time if self.end_time > 0 else 0
        return {
            'total_files': self.total_files,
            'successful_files': self.successful_files,
            'failed_files': self.failed_files,
            'total_pages_processed': self.total_pages_processed,
            'total_pages_ocr': self.total_pages_ocr,
            'total_pages_text_layer': self.total_pages_text_layer,
            'duration_seconds': round(duration, 2),
            'pages_per_second': round(self.total_pages_processed / duration, 2) if duration > 0 else 0
        }


class OCRBatchService:
    """
    Production-ready OCR batch processing service.

    Features:
    - Adaptive memory management with VRAM monitoring
    - Exponential backoff retry logic
    - Graceful error recovery
    - Detailed progress tracking with ETA
    - Cancellation support
    - Text layer detection to skip unnecessary OCR
    """

    # Retry configuration
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 1.0  # seconds
    RETRY_BACKOFF_FACTOR = 2.0

    # Progress update interval
    MIN_PROGRESS_INTERVAL = 2.0  # seconds

    # Memory cleanup interval
    CLEANUP_PAGE_INTERVAL = 10  # pages

    def __init__(
        self,
        progress_callback: Optional[Callable[[int, int, str, float, float], None]] = None,
        cancellation_flag: Optional[threading.Event] = None,
        use_gpu: bool = True,
        ocr_language: str = 'en',
        max_dpi: Optional[int] = None
    ):
        """
        Initialize OCR batch service.

        Args:
            progress_callback: Callable(current, total, message, percent, eta)
                - current: Current item index
                - total: Total items
                - message: Status message
                - percent: Completion percentage (0-100)
                - eta: Estimated time remaining in seconds
            cancellation_flag: threading.Event for cancellation signaling
            use_gpu: Whether to use GPU acceleration if available
            ocr_language: Language code for OCR (e.g., 'en', 'ch', 'french')
            max_dpi: Maximum DPI for rendering (None = use system recommendation)
        """
        self.progress_callback = progress_callback
        self.cancellation_flag = cancellation_flag
        self.use_gpu = use_gpu
        self.ocr_language = ocr_language
        self.max_dpi = max_dpi

        # Services (lazy initialized)
        self.ocr_service: Optional[OCRService] = None
        self.vram_monitor: Optional[VRAMMonitor] = None

        # Hardware detection
        self.capabilities = detect_hardware_capabilities()
        self.batch_size = self._detect_batch_size()

        # Progress tracking
        self.last_progress_time = 0.0
        self.processing_start_time = 0.0

        # Statistics
        self.stats = ProcessingStats()

        logger.info(
            f"OCR Batch Service initialized: "
            f"GPU={self.use_gpu and self.capabilities['gpu_available']}, "
            f"language={self.ocr_language}, "
            f"max_dpi={self.max_dpi or 'auto'}, "
            f"batch_size={self.batch_size}, "
            f"VRAM={self.capabilities['gpu_memory_gb']:.1f}GB, "
            f"RAM={self.capabilities['system_memory_gb']:.1f}GB"
        )

    def _detect_batch_size(self) -> int:
        """
        Detect optimal batch size based on GPU VRAM and system RAM.

        Returns:
            Optimal batch size for current hardware
        """
        use_gpu = self.use_gpu and self.capabilities['gpu_available']

        batch_size = get_optimal_batch_size(
            use_gpu=use_gpu,
            gpu_memory_gb=self.capabilities['gpu_memory_gb'],
            system_memory_gb=self.capabilities['system_memory_gb']
        )

        logger.info(f"Detected optimal batch size: {batch_size}")
        return batch_size

    def _initialize_services(self) -> None:
        """Initialize OCR service and VRAM monitor (lazy initialization)"""
        if self.ocr_service is None:
            logger.info("Initializing OCR service...")
            self.ocr_service = OCRService(
                gpu=self.use_gpu,
                engine="paddleocr",  # Use PaddleOCR as primary
                fallback_enabled=True,
                language=self.ocr_language,
                max_dpi=self.max_dpi
            )

            if not self.ocr_service.is_available():
                raise RuntimeError("Failed to initialize OCR service")

            # Log engine info
            engine_info = self.ocr_service.get_engine_info()
            logger.info(
                f"OCR engine ready: {engine_info['engine']}, "
                f"GPU={engine_info['gpu_enabled']}, "
                f"memory={engine_info['memory_usage_mb']:.1f}MB, "
                f"language={self.ocr_language}, "
                f"max_dpi={self.max_dpi or 'auto'}"
            )

        if self.vram_monitor is None and self.use_gpu and self.capabilities['gpu_available']:
            logger.info("Initializing VRAM monitor...")
            self.vram_monitor = VRAMMonitor(check_interval=1.0)

    def _is_cancelled(self) -> bool:
        """Check if processing has been cancelled"""
        if self.cancellation_flag is None:
            return False
        return self.cancellation_flag.is_set()

    def _should_report_progress(self) -> bool:
        """Check if enough time has passed to report progress"""
        current_time = time.time()
        if current_time - self.last_progress_time >= self.MIN_PROGRESS_INTERVAL:
            self.last_progress_time = current_time
            return True
        return False

    def _report_progress(
        self,
        current: int,
        total: int,
        message: str,
        force: bool = False
    ) -> None:
        """
        Report progress to callback if available.

        Args:
            current: Current item number
            total: Total items
            message: Status message
            force: Force update regardless of time interval
        """
        if self.progress_callback is None:
            return

        if not force and not self._should_report_progress():
            return

        # Calculate percentage
        percent = (current / total * 100) if total > 0 else 0

        # Estimate ETA
        elapsed = time.time() - self.processing_start_time
        if current > 0 and elapsed > 0:
            rate = current / elapsed
            remaining_items = total - current
            eta = remaining_items / rate if rate > 0 else 0
        else:
            eta = 0

        try:
            self.progress_callback(current, total, message, percent, eta)
        except Exception as e:
            logger.error(f"Progress callback failed: {e}")

    def _retry_with_backoff(
        self,
        operation: Callable[[], Any],
        operation_name: str,
        max_retries: Optional[int] = None
    ) -> Any:
        """
        Retry operation with exponential backoff.

        Args:
            operation: Callable to execute
            operation_name: Human-readable operation name for logging
            max_retries: Maximum retry attempts (default: MAX_RETRIES)

        Returns:
            Result from successful operation

        Raises:
            Exception: If all retries exhausted
        """
        if max_retries is None:
            max_retries = self.MAX_RETRIES

        last_exception = None
        delay = self.INITIAL_RETRY_DELAY

        for attempt in range(max_retries):
            try:
                result = operation()
                if attempt > 0:
                    logger.info(f"{operation_name} succeeded on attempt {attempt + 1}")
                return result

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}"
                )

                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    delay *= self.RETRY_BACKOFF_FACTOR

        # All retries exhausted
        logger.error(f"{operation_name} failed after {max_retries} attempts")
        raise last_exception

    def _validate_ocr_output(self, text: str, page_num: int) -> tuple:
        """
        Validate that OCR produced usable text.

        Args:
            text: OCR extracted text
            page_num: Page number (for logging)

        Returns:
            (is_valid, reason) tuple
        """
        # Check for empty text
        if len(text.strip()) == 0:
            return False, "No text extracted"

        # Check for garbage (mostly non-alphanumeric)
        alphanumeric_count = sum(c.isalnum() for c in text)
        alphanumeric_ratio = alphanumeric_count / len(text) if len(text) > 0 else 0

        if alphanumeric_ratio < 0.30:  # Less than 30% alphanumeric = likely garbage
            return False, f"Low alphanumeric ratio ({alphanumeric_ratio:.1%})"

        # Check for common OCR failure patterns
        if text.strip() in ['|||||||', '###', '...', '---']:
            return False, "OCR failure pattern detected"

        return True, "Valid OCR output"

    def _ocr_with_settings(
        self,
        pdf: PDFProcessor,
        page_num: int,
        engine: str,
        dpi: int
    ) -> str:
        """
        Process single page with specific OCR engine and DPI.

        Args:
            pdf: PDFProcessor instance
            page_num: Page number
            engine: "paddleocr" or "tesseract"
            dpi: DPI setting (300, 400, etc.)

        Returns:
            Extracted text
        """
        # Create temporary OCR service with specific engine
        temp_ocr = OCRService(
            gpu=self.use_gpu,
            engine=engine,
            fallback_enabled=False,
            language=self.ocr_language,
            max_dpi=self.max_dpi
        )

        try:
            image = pdf.render_page_to_image(page_num, dpi=dpi)
            text = temp_ocr.extract_text_from_array(image)
            del image
            return text
        finally:
            temp_ocr.cleanup()

    def _retry_ocr_with_fallback(
        self,
        pdf: PDFProcessor,
        page_num: int,
        file_name: str
    ) -> tuple:
        """
        Retry OCR with fallback engines and DPI settings.

        Strategy:
        1. Try PaddleOCR at 300 DPI (default)
        2. Try PaddleOCR at 400 DPI (higher quality)
        3. Try Tesseract at 300 DPI (fallback engine)
        4. Try Tesseract at 400 DPI (last resort)

        Args:
            pdf: PDFProcessor instance
            page_num: Page number to process
            file_name: File name (for logging)

        Returns:
            (text, method) tuple - extracted text and method used
        """
        strategies = [
            ("PaddleOCR 400 DPI", lambda: self._ocr_with_settings(pdf, page_num, "paddleocr", 400)),
            ("Tesseract 300 DPI", lambda: self._ocr_with_settings(pdf, page_num, "tesseract", 300)),
            ("Tesseract 400 DPI", lambda: self._ocr_with_settings(pdf, page_num, "tesseract", 400)),
        ]

        for strategy_name, strategy_func in strategies:
            try:
                logger.info(f"Retrying page {page_num+1} with {strategy_name}")
                text = strategy_func()

                is_valid, reason = self._validate_ocr_output(text, page_num)
                if is_valid:
                    logger.info(f"Retry succeeded with {strategy_name}")
                    return text, strategy_name
                else:
                    logger.warning(f"{strategy_name} failed validation: {reason}")

            except Exception as e:
                logger.warning(f"{strategy_name} failed: {e}")
                continue

        # All strategies failed
        logger.error(f"All OCR retry strategies failed for page {page_num+1}")
        return "", "ALL_FAILED"

    def _calculate_quality_score(self, text: str, page: fitz.Page) -> float:
        """
        Calculate text quality score using existing TextLayerValidator.

        Args:
            text: Text to evaluate
            page: PyMuPDF page object

        Returns:
            Quality score (0.0-1.0)
        """
        from .ocr.text_quality import TextLayerValidator, TextQualityThresholds

        try:
            # Create temporary validator
            validator = TextLayerValidator(TextQualityThresholds())

            # Calculate metrics (reuse existing validation logic)
            metrics = validator._calculate_metrics(text, page)

            return metrics.confidence_score
        
        except Exception as e:
            # If quality calculation fails, return neutral score
            logger.warning(f"Quality score calculation failed: {e}")
            return 0.5  # Neutral score - neither high nor low quality

    def _should_replace_with_ocr(
        self,
        original_text: str,
        ocr_text: str,
        page: fitz.Page,
        page_num: int
    ) -> tuple:
        """
        Decide whether to replace original text with OCR result.

        Compares quality scores and only replaces if OCR is better.

        Args:
            original_text: Original text from partial layer
            ocr_text: New OCR result
            page: PyMuPDF page object
            page_num: Page number (for logging)

        Returns:
            (should_replace, reason) tuple
        """
        # Calculate quality scores
        original_score = self._calculate_quality_score(original_text, page)
        ocr_score = self._calculate_quality_score(ocr_text, page)

        logger.debug(
            f"Page {page_num+1}: Quality comparison - "
            f"Original: {original_score:.2%}, OCR: {ocr_score:.2%}"
        )

        # Decision logic
        QUALITY_THRESHOLD = 0.70  # Minimum acceptable quality
        IMPROVEMENT_MARGIN = 0.05  # Require 5% improvement to justify replacement

        # If original is already high quality, require significant improvement
        if original_score >= QUALITY_THRESHOLD:
            if ocr_score > original_score + IMPROVEMENT_MARGIN:
                return True, f"OCR improved quality ({original_score:.2%} → {ocr_score:.2%})"
            else:
                return False, f"Original quality sufficient ({original_score:.2%}), OCR not better ({ocr_score:.2%})"

        # If original is low quality, accept any improvement
        if ocr_score > original_score:
            return True, f"OCR improved quality ({original_score:.2%} → {ocr_score:.2%})"
        else:
            return False, f"OCR did not improve quality ({original_score:.2%} → {ocr_score:.2%})"

    def _clean_rebuild_page_with_ocr(
        self,
        doc: fitz.Document,
        page_num: int,
        ocr_text: str
    ) -> None:
        """
        Rebuild page cleanly with OCR text layer, removing all existing text.

        This prevents text layer duplication by:
        1. Rendering the page to an image (preserves visual appearance)
        2. Creating a new blank page
        3. Inserting the rendered image
        4. Adding OCR text as invisible overlay
        5. Replacing the original page

        Args:
            doc: PyMuPDF document
            page_num: Page number to rebuild
            ocr_text: OCR text to embed
        """
        try:
            # Get original page
            original_page = doc[page_num]
            page_rect = original_page.rect

            logger.debug(f"Rebuilding page {page_num+1} with clean OCR layer")

            # Step 1: Render page to high-quality image
            # This captures ALL visual content (text, images, drawings)
            # but loses the text layer (which we want)
            pix = original_page.get_pixmap(dpi=300)

            # Step 2: Create new blank page with same dimensions
            new_page = doc.new_page(
                width=page_rect.width,
                height=page_rect.height
            )

            # Step 3: Insert rendered image to preserve visual appearance
            new_page.insert_image(page_rect, pixmap=pix)

            # Step 4: Add OCR text as invisible overlay
            # This creates searchable text without visible rendering
            new_page.insert_textbox(
                page_rect,
                ocr_text,
                fontsize=8,
                color=(1, 1, 1),  # White (invisible)
                overlay=True,
                render_mode=3  # Invisible text mode
            )

            # Step 5: Replace original page with clean rebuilt page
            # IMPORTANT: Delete original FIRST, then insert new page at same position
            # This avoids index shifting issues with move_page
            
            # Delete the original page first
            doc.delete_page(page_num)
            
            # Now insert the new page at the correct position
            # (new_page is currently at the end of the document)
            doc.move_page(doc.page_count - 1, page_num)
            
            # Clean up pixmap (explicit deletion for memory management)
            del pix
            import gc
            gc.collect()

            logger.debug(f"Page {page_num+1} rebuilt successfully")

        except Exception as e:
            logger.error(f"Failed to rebuild page {page_num+1}: {e}", exc_info=True)
            raise

    def _process_page_batch(
        self,
        pdf: PDFProcessor,
        page_numbers: List[int],
        file_name: str
    ) -> List[str]:
        """
        Process a batch of PDF pages with OCR and retry logic.

        Args:
            pdf: PDFProcessor instance
            page_numbers: List of page numbers to process
            file_name: Name of file being processed (for logging)

        Returns:
            List of extracted text for each page
        """
        # Check VRAM pressure and adjust batch size if needed
        if self.vram_monitor:
            if self.vram_monitor.should_reduce_batch_size():
                pressure = self.vram_monitor.get_memory_pressure_level()
                logger.warning(
                    f"High VRAM pressure detected ({pressure}), "
                    f"processing pages individually"
                )
                # Process one at a time under memory pressure
                texts = []
                for page_num in page_numbers:
                    text = self._process_single_page(pdf, page_num, file_name)
                    texts.append(text)
                return texts

        # Normal batch processing
        def process_batch_operation():
            # Render pages to images - use configured DPI or system recommendation
            dpi = self.max_dpi if self.max_dpi else 300
            images = []
            for page_num in page_numbers:
                image = pdf.render_page_to_image(page_num, dpi=dpi)
                images.append(image)

            # Process with OCR
            texts = self.ocr_service.process_batch(images)

            # Verify results match input count
            if len(texts) != len(images):
                logger.error(
                    f"OCR batch result mismatch: expected {len(images)}, "
                    f"got {len(texts)}"
                )
                # Pad with empty strings
                while len(texts) < len(images):
                    texts.append("")

            # Cleanup images
            del images

            return texts

        # Execute with retry
        texts = self._retry_with_backoff(
            process_batch_operation,
            f"OCR batch ({len(page_numbers)} pages from {file_name})"
        )

        # VALIDATE AND RETRY
        validated_texts = []
        for idx, (page_num, text) in enumerate(zip(page_numbers, texts)):
            # DEBUG: Log what OCR extracted before validation
            logger.debug(f"Page {page_num+1}: Raw OCR extracted {len(text)} chars: '{text[:100]}...'")

            # Validate OCR output
            is_valid, reason = self._validate_ocr_output(text, page_num)

            if is_valid:
                validated_texts.append(text)
                logger.debug(f"Page {page_num+1}: OCR valid ({len(text)} chars) - {reason}")
            else:
                logger.warning(f"Page {page_num+1}: OCR validation FAILED - {reason} (extracted {len(text)} chars)")

                # Retry with fallback
                retry_text, method = self._retry_ocr_with_fallback(pdf, page_num, file_name)

                if retry_text:
                    validated_texts.append(retry_text)
                    logger.info(f"Page {page_num+1}: Retry succeeded with {method}")
                else:
                    # All retries failed - use empty text
                    logger.error(f"Page {page_num+1}: All OCR attempts failed")
                    validated_texts.append("")

        return validated_texts

    def _process_single_page(
        self,
        pdf: PDFProcessor,
        page_num: int,
        file_name: str
    ) -> str:
        """
        Process a single page with OCR and retry logic.

        Args:
            pdf: PDFProcessor instance
            page_num: Page number to process
            file_name: Name of file being processed (for logging)

        Returns:
            Extracted text
        """
        def process_page_operation():
            # Use configured DPI or system recommendation
            dpi = self.max_dpi if self.max_dpi else 300
            image = pdf.render_page_to_image(page_num, dpi=dpi)
            text = self.ocr_service.extract_text_from_array(image)
            del image
            return text

        try:
            text = self._retry_with_backoff(
                process_page_operation,
                f"OCR page {page_num + 1} from {file_name}"
            )
            return text
        except Exception as e:
            logger.error(f"Failed to process page {page_num + 1} after retries: {e}")
            return ""  # Return empty string on failure

    def _process_single_file(
        self,
        file_path: str,
        output_dir: str,
        file_idx: int,
        total_files: int
    ) -> Dict[str, Any]:
        """
        Process a single PDF file.

        Args:
            file_path: Path to PDF file
            output_dir: Directory to save processed file
            file_idx: Current file index (1-based)
            total_files: Total number of files

        Returns:
            Dictionary with processing results:
                - status: "success" or "failed"
                - pages_processed: Number of pages processed
                - pages_ocr: Number of pages that required OCR
                - pages_text_layer: Number of pages with text layer
                - error: Error message (if failed)
                - output_path: Path to output file (if successful)
        """
        file_name = Path(file_path).name
        logger.info(f"Processing file {file_idx}/{total_files}: {file_name}")

        result = {
            'status': 'failed',
            'pages_processed': 0,
            'pages_ocr': 0,
            'pages_text_layer': 0,
            'error': None,
            'output_path': None
        }

        try:
            # Open PDF
            with PDFProcessor(file_path) as pdf:
                total_pages = pdf.get_page_count()
                logger.info(f"File has {total_pages} pages")

                # Track pages needing OCR and original texts
                pages_needing_ocr = []
                page_texts = {}
                original_texts = {}  # Store original text for quality comparison

                # Check each page for text layer
                for page_num in range(total_pages):
                    # Check cancellation
                    if self._is_cancelled():
                        logger.info(f"Processing cancelled at page {page_num + 1}")
                        result['error'] = "Processing cancelled by user"
                        return result

                    # Report progress
                    self._report_progress(
                        current=page_num,
                        total=total_pages,
                        message=f"File {file_idx}/{total_files}: {file_name} - Checking page {page_num + 1}/{total_pages}"
                    )

                    # Check for text layer
                    has_text, _ = pdf.has_valid_text_layer(page_num)

                    if has_text:
                        # Extract from text layer (fast)
                        text = pdf.extract_text(page_num)
                        page_texts[page_num] = text
                        original_texts[page_num] = text  # Save for quality comparison
                        result['pages_text_layer'] += 1
                        logger.debug(f"Page {page_num + 1}: Using text layer")
                    else:
                        # Queue for OCR
                        pages_needing_ocr.append(page_num)
                        # Save any partial text for quality comparison
                        original_texts[page_num] = pdf.extract_text(page_num)
                        logger.debug(f"Page {page_num + 1}: Needs OCR")

                # Process pages needing OCR in batches
                if pages_needing_ocr:
                    logger.info(f"{len(pages_needing_ocr)} pages need OCR")

                    for batch_start_idx in range(0, len(pages_needing_ocr), self.batch_size):
                        # Check cancellation
                        if self._is_cancelled():
                            logger.info("Processing cancelled during OCR")
                            result['error'] = "Processing cancelled by user"
                            return result

                        batch_end_idx = min(batch_start_idx + self.batch_size, len(pages_needing_ocr))
                        batch_page_nums = pages_needing_ocr[batch_start_idx:batch_end_idx]

                        # Report progress
                        self._report_progress(
                            current=batch_start_idx,
                            total=len(pages_needing_ocr),
                            message=f"File {file_idx}/{total_files}: {file_name} - OCR batch {batch_start_idx + 1}-{batch_end_idx}/{len(pages_needing_ocr)}"
                        )

                        # Process batch
                        texts = self._process_page_batch(pdf, batch_page_nums, file_name)

                        # Store results and track success/failure
                        for page_num, text in zip(batch_page_nums, texts):
                            if text:  # Only store non-empty text
                                page_texts[page_num] = text
                                result['pages_ocr'] += 1
                            else:  # Track failed OCR attempts
                                logger.warning(f"Page {page_num+1}: OCR failed, no text extracted")
                                result['pages_ocr_failed'] = result.get('pages_ocr_failed', 0) + 1

                        # Memory cleanup every N pages
                        if (batch_end_idx % self.CLEANUP_PAGE_INTERVAL) == 0:
                            gc.collect()
                            if self.vram_monitor:
                                self.vram_monitor.log_stats()

                result['pages_processed'] = total_pages

                # Create output directory if needed
                output_path = Path(output_dir) / file_name
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Save PDF with embedded OCR text layers
                try:
                    # Open the original PDF with fitz
                    doc = fitz.open(file_path)

                    # Add invisible text layers to OCR'd pages with clean rebuild
                    # IMPORTANT: Process in REVERSE order to avoid index shifting issues
                    # When we rebuild page 0, indices for pages 5, 10 shift
                    # By processing from end to beginning, indices remain stable
                    if pages_needing_ocr:
                        for page_num in reversed(pages_needing_ocr):
                            if page_num not in page_texts:
                                continue

                            ocr_text = page_texts[page_num]

                            # Skip if no text (failed OCR)
                            if not ocr_text:
                                logger.warning(f"Skipping page {page_num+1}: No OCR text available")
                                continue

                            # Quality comparison (from Step 2)
                            if page_num in original_texts and original_texts[page_num]:
                                should_replace, reason = self._should_replace_with_ocr(
                                    original_texts[page_num],
                                    ocr_text,
                                    doc[page_num],
                                    page_num
                                )

                                if not should_replace:
                                    logger.info(f"Page {page_num+1}: Keeping original - {reason}")
                                    continue

                            # CLEAN REBUILD WITH OCR
                            try:
                                self._clean_rebuild_page_with_ocr(doc, page_num, ocr_text)
                                logger.info(f"Page {page_num+1}: Embedded clean OCR layer")
                            except Exception as e:
                                logger.error(f"Page {page_num+1}: Failed to embed OCR - {e}")
                                # Continue with other pages

                    # Save the modified PDF
                    doc.save(str(output_path), garbage=4, deflate=True)
                    doc.close()

                    result['output_path'] = str(output_path)
                    result['status'] = 'success'

                    logger.info(f"Saved PDF with OCR text layers to {output_path}")

                except Exception as save_error:
                    logger.error(f"Failed to save PDF {file_name}: {save_error}", exc_info=True)
                    # Fall back to copying original if save fails
                    try:
                        import shutil
                        shutil.copy2(file_path, output_path)
                        result['output_path'] = str(output_path)
                        result['status'] = 'success'
                        result['warning'] = f"Copied original PDF (OCR text layer save failed: {save_error})"
                        logger.warning(f"Fell back to copying original PDF for {file_name}")
                    except Exception as copy_error:
                        logger.error(f"Failed to copy PDF {file_name}: {copy_error}", exc_info=True)
                        result['error'] = f"Failed to save PDF: {save_error}"
                        result['status'] = 'failed'
                        return result

                result['status'] = 'success'

                logger.info(
                    f"Successfully processed {file_name}: "
                    f"{result['pages_text_layer']} from text layer, "
                    f"{result['pages_ocr']} from OCR"
                )

        except Exception as e:
            logger.error(f"Failed to process {file_name}: {e}", exc_info=True)
            result['error'] = str(e)

        finally:
            # Cleanup memory after each file
            gc.collect()
            if self.use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

        return result

    def process_batch(
        self,
        files: List[str],
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Process multiple PDF files sequentially.

        Args:
            files: List of PDF file paths
            output_dir: Directory to save processed files

        Returns:
            Dictionary with batch results:
                - successful: List of successfully processed files
                - failed: List of failed files with error info
                - total_pages_processed: Total pages across all files
                - statistics: Detailed processing statistics
        """
        self.processing_start_time = time.time()
        self.stats = ProcessingStats(
            total_files=len(files),
            start_time=self.processing_start_time
        )

        logger.info(f"Starting batch processing: {len(files)} files")

        # Initialize services
        try:
            self._initialize_services()
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            return {
                'successful': [],
                'failed': [{'file': f, 'error': 'Service initialization failed'} for f in files],
                'total_pages_processed': 0,
                'statistics': self.stats.to_dict()
            }

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        successful = []
        failed = []

        # Process each file
        for file_idx, file_path in enumerate(files, start=1):
            # Check cancellation
            if self._is_cancelled():
                logger.info("Batch processing cancelled")
                break

            # Process file
            result = self._process_single_file(
                file_path=file_path,
                output_dir=output_dir,
                file_idx=file_idx,
                total_files=len(files)
            )

            # Update statistics
            self.stats.total_pages_processed += result['pages_processed']
            self.stats.total_pages_ocr += result['pages_ocr']
            self.stats.total_pages_text_layer += result['pages_text_layer']

            # Categorize result
            if result['status'] == 'success':
                successful.append({
                    'file': file_path,
                    'pages_processed': result['pages_processed'],
                    'pages_ocr': result['pages_ocr'],
                    'pages_text_layer': result['pages_text_layer'],
                    'output_path': result['output_path']
                })
                self.stats.successful_files += 1
            else:
                failed.append({
                    'file': file_path,
                    'error': result['error']
                })
                self.stats.failed_files += 1

        self.stats.end_time = time.time()

        # Final progress report
        self._report_progress(
            current=len(files),
            total=len(files),
            message="Batch processing complete",
            force=True
        )

        # Summary
        duration = self.stats.end_time - self.stats.start_time
        logger.info(
            f"Batch processing complete: "
            f"{self.stats.successful_files} successful, "
            f"{self.stats.failed_files} failed, "
            f"{self.stats.total_pages_processed} pages processed "
            f"({self.stats.total_pages_ocr} OCR, {self.stats.total_pages_text_layer} text layer) "
            f"in {duration:.1f}s"
        )

        return {
            'successful': successful,
            'failed': failed,
            'total_pages_processed': self.stats.total_pages_processed,
            'statistics': self.stats.to_dict()
        }

    def cleanup(self) -> None:
        """Release OCR resources and cleanup memory"""
        logger.info("Cleaning up OCR batch service...")

        if self.ocr_service is not None:
            self.ocr_service.cleanup()
            self.ocr_service = None

        # Force garbage collection
        gc.collect()

        # Clear GPU cache if available
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("GPU cache cleared")
            except ImportError:
                pass

        logger.info("Cleanup complete")

    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()
