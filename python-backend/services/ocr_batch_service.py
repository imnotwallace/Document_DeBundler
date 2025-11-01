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
        use_gpu: bool = True
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
        """
        self.progress_callback = progress_callback
        self.cancellation_flag = cancellation_flag
        self.use_gpu = use_gpu

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
                fallback_enabled=True
            )

            if not self.ocr_service.is_available():
                raise RuntimeError("Failed to initialize OCR service")

            # Log engine info
            engine_info = self.ocr_service.get_engine_info()
            logger.info(
                f"OCR engine ready: {engine_info['engine']}, "
                f"GPU={engine_info['gpu_enabled']}, "
                f"memory={engine_info['memory_usage_mb']:.1f}MB"
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
            # Render pages to images
            images = []
            for page_num in page_numbers:
                image = pdf.render_page_to_image(page_num, dpi=300)
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

        return texts

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
            image = pdf.render_page_to_image(page_num, dpi=300)
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

                # Track pages needing OCR
                pages_needing_ocr = []
                page_texts = {}

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
                        result['pages_text_layer'] += 1
                        logger.debug(f"Page {page_num + 1}: Using text layer")
                    else:
                        # Queue for OCR
                        pages_needing_ocr.append(page_num)
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

                        # Store results
                        for page_num, text in zip(batch_page_nums, texts):
                            page_texts[page_num] = text
                            result['pages_ocr'] += 1

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
                    
                    # Add invisible text layers to OCR'd pages
                    if pages_needing_ocr:
                        for page_num in pages_needing_ocr:
                            if page_num in page_texts:
                                page = doc[page_num]
                                text = page_texts[page_num]
                                
                                # Add text as invisible overlay
                                # Use the full page rectangle
                                rect = page.rect
                                
                                # Insert text block with invisible font (white on white)
                                # This makes the text searchable but not visible
                                page.insert_textbox(
                                    rect,
                                    text,
                                    fontsize=8,
                                    color=(1, 1, 1),  # White text (invisible on white background)
                                    overlay=True,
                                    render_mode=3  # Invisible text (render mode 3)
                                )
                    
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
