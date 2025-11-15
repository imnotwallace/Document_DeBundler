"""
OCR Batch Service

Production-ready OCR batch processing with:
- STREAMING ASSEMBLY: Constant memory usage (190 MB) regardless of PDF size
- DPI-FIRST SIZING: Prioritize target DPI over batch size (batch_size=1 if needed)
- Adaptive memory management with VRAM monitoring
- Error recovery and progress tracking
- 2-image preprocessing pipeline (deskewed + preprocessed)

Memory optimization:
- Traditional: 5000 pages × 7.5 MB = 37.5 GB RAM (OOM!)
- Streaming: batch_size × 10 MB = 190 MB RAM (constant)
"""

import logging
import time
import gc
import hashlib
import threading
import unicodedata
import fitz  # PyMuPDF
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from .ocr_service import OCRService
from .pdf_processor import PDFProcessor
from .ocr.config import detect_hardware_capabilities, get_optimal_batch_size
from .ocr.vram_monitor import VRAMMonitor
from .ocr.base import OCRResult
from .ocr.intelligent_preprocessing import IntelligentPreprocessor
from .memory_monitor import MemoryMonitor

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


@dataclass
class OCRProcessingConfig:
    """Configuration for OCR processing behavior and quality thresholds"""
    
    # Quality validation thresholds
    min_alphanumeric_ratio: float = 0.10  # Lowered from 0.15 to support symbol-heavy content (receipts, forms, separators)
    min_quality_improvement_margin: float = 0.02  # Lowered from 0.05 for more sensitive comparison
    min_quality_threshold: float = 0.70  # Threshold for "good enough" quality
    
    # Size optimization
    image_compression_quality: int = 85  # JPEG quality for scanned pages (0-100)
    enable_compression: bool = True  # Apply compression to images
    
    # Processing modes
    processing_mode: str = "hybrid"  # "hybrid", "selective", "full"
    # - hybrid: Analyze each page, only OCR what needs improvement
    # - selective: Only OCR pages with no text layer
    # - full: OCR all pages regardless
    
    # Coordinate mapping
    use_coordinate_mapping: bool = True  # Position text using OCR bounding boxes
    
    # Intelligent preprocessing (NEW)
    enable_intelligent_preprocessing: bool = True  # Auto-optimize images for OCR
    preprocessing_allow_destructive: bool = True  # Allow binarization/morphological ops
    preprocessing_enable_validation: bool = True  # Validate quality improvements
    preprocessing_min_quality_improvement: float = 4.0  # Minimum improvement to use preprocessed (tuned for photo documents)
    preprocessing_min_ssim: float = 0.85  # Minimum structural similarity threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'min_alphanumeric_ratio': self.min_alphanumeric_ratio,
            'min_quality_improvement_margin': self.min_quality_improvement_margin,
            'min_quality_threshold': self.min_quality_threshold,
            'image_compression_quality': self.image_compression_quality,
            'enable_compression': self.enable_compression,
            'processing_mode': self.processing_mode,
            'use_coordinate_mapping': self.use_coordinate_mapping,
            'enable_intelligent_preprocessing': self.enable_intelligent_preprocessing,
            'preprocessing_allow_destructive': self.preprocessing_allow_destructive,
            'preprocessing_enable_validation': self.preprocessing_enable_validation,
            'preprocessing_min_quality_improvement': self.preprocessing_min_quality_improvement,
            'preprocessing_min_ssim': self.preprocessing_min_ssim,
        }


class OCRBatchService:
    """
    Production-ready OCR batch processing service with STREAMING ASSEMBLY.

    MEMORY ARCHITECTURE:
    ====================
    Uses streaming assembly to achieve constant memory usage regardless of PDF size.
    
    Traditional approach (MEMORY BOMB):
    - Process all batches, accumulate images in memory
    - Peak RAM: total_pages × 7.5 MB (37.5 GB for 5000 pages!)
    
    Streaming approach (CONSTANT MEMORY):
    - OCR batch → immediately assemble to PDF → free images
    - Peak RAM: batch_size × 10 MB (~190 MB constant)
    
    PROCESSING FLOW:
    ================
    Phase 1: Scan all pages (identify text layer vs OCR needed)
    Phase 2: Process OCR batches in REVERSE order
    Phase 3: IMMEDIATELY assemble each page to output PDF
    Phase 4: Add text overlays to text-layer pages
    Phase 5: Save final PDF
    
    Why reverse order?
    - Rebuilding page 0 shifts indices for pages 1,2,3...
    - Processing backwards (999→0) keeps indices stable
    
    DPI-FIRST BATCH SIZING:
    =======================
    When user specifies target DPI from frontend:
    - Calculate batch_size to achieve that DPI (even if batch_size=1)
    - Prioritize quality over speed
    - Memory is managed by streaming, not batch size
    
    Features:
    - Adaptive memory management with VRAM monitoring
    - Exponential backoff retry logic
    - Graceful error recovery
    - Detailed progress tracking with ETA
    - Cancellation support
    - Text layer detection to skip unnecessary OCR
    - Streaming assembly (constant memory)
    - DPI-first batch sizing
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
        file_status_callback: Optional[Callable] = None,
        cancellation_flag: Optional[threading.Event] = None,
        use_gpu: bool = True,
        config: Optional[OCRProcessingConfig] = None,
        ocr_config: Optional[dict] = None,
        ocr_service_cache: Optional['OCRService'] = None
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
            file_status_callback: Callable for detailed file status updates with timing
            cancellation_flag: threading.Event for cancellation signaling
            use_gpu: Whether to use GPU acceleration if available
            config: OCR processing configuration object (uses defaults if not provided)
            ocr_config: Raw OCR configuration dictionary from frontend (converted to OCRProcessingConfig)
            ocr_service_cache: Pre-initialized OCR service from module initialization (reused for performance)
        """
        self.progress_callback = progress_callback
        self.file_status_callback = file_status_callback
        self.cancellation_flag = cancellation_flag

        # Handle ocr_config from frontend - convert dict to OCRProcessingConfig
        if ocr_config:
            logger.info("Converting frontend OCR config to OCRProcessingConfig")
            self.config = self._convert_frontend_config(ocr_config)
            # Override use_gpu if specified in config
            self.use_gpu = ocr_config.get('use_gpu', use_gpu)
        else:
            self.use_gpu = use_gpu
            self.config = config or OCRProcessingConfig()

        # Services (lazy initialized)
        self.ocr_service: Optional[OCRService] = None
        self.ocr_service_cache = ocr_service_cache  # Store cached service for reuse
        self.vram_monitor: Optional[VRAMMonitor] = None
        self.preprocessor: Optional[IntelligentPreprocessor] = None
        self.memory_monitor: Optional[MemoryMonitor] = None  # Memory leak detection

        # Hardware detection
        self.capabilities = detect_hardware_capabilities()
        self.batch_size = self._detect_batch_size()

        # Progress tracking
        self.last_progress_time = 0.0
        self.processing_start_time = 0.0

        # Statistics
        self.stats = ProcessingStats()
        
        # File timing tracking
        self.file_queue_times = {}  # {file_path: queued_at_timestamp}
        self.file_start_times = {}  # {file_path: started_at_timestamp}

        logger.info(
            f"OCR Batch Service initialized: "
            f"GPU={self.use_gpu and self.capabilities['gpu_available']}, "
            f"batch_size={self.batch_size}, "
            f"VRAM={self.capabilities['gpu_memory_gb']:.1f}GB, "
            f"RAM={self.capabilities['system_memory_gb']:.1f}GB, "
            f"mode={self.config.processing_mode}, "
            f"coordinate_mapping={self.config.use_coordinate_mapping}, "
            f"cached_service={'yes' if ocr_service_cache else 'no'}"
        )

    def _convert_frontend_config(self, ocr_config: dict) -> OCRProcessingConfig:
        """
        Convert frontend OCR configuration dictionary to OCRProcessingConfig.

        Frontend config includes:
        - use_gpu, dpi, languages, batch_size: OCR engine settings
        - processing_mode: Maps to OCRProcessingConfig
        - confidence_threshold: Maps to min_quality_threshold

        Args:
            ocr_config: Dictionary from frontend with OCR settings

        Returns:
            OCRProcessingConfig object with mapped settings
        """
        config = OCRProcessingConfig()

        # Map processing mode (hybrid/selective/full)
        if 'processing_mode' in ocr_config:
            config.processing_mode = ocr_config['processing_mode']
            logger.info(f"Using processing mode: {config.processing_mode}")

        # Map confidence threshold to quality threshold
        if 'confidence_threshold' in ocr_config:
            config.min_quality_threshold = ocr_config['confidence_threshold']
            logger.info(f"Min quality threshold: {config.min_quality_threshold}")

        # Enable coordinate mapping by default (can be disabled in frontend config)
        if 'use_coordinate_mapping' in ocr_config:
            config.use_coordinate_mapping = ocr_config['use_coordinate_mapping']

        # Note: DPI, languages, and other OCR engine settings are stored separately
        # and will be passed to OCRService initialization
        # Store them in instance variables for later use
        if 'dpi' in ocr_config:
            self.override_dpi = ocr_config['dpi']
            logger.info(f"DPI override: {self.override_dpi}")
        else:
            self.override_dpi = None

        if 'languages' in ocr_config:
            self.override_languages = ocr_config['languages']
            logger.info(f"Language override: {self.override_languages}")
        else:
            self.override_languages = None

        if 'batch_size' in ocr_config:
            # Override the auto-detected batch size
            self.batch_size = ocr_config['batch_size']
            logger.info(f"Batch size override: {self.batch_size}")

        return config

    def _detect_batch_size(self) -> int:
        """
        Detect optimal batch size based on target DPI (if provided) or GPU VRAM.
        
        DPI-FIRST APPROACH: If user specifies DPI, calculate batch size to achieve that DPI.
        Otherwise, use legacy optimization (maximize batch size, reduce DPI if needed).
        
        Automatically adjusts for preprocessing pipeline memory overhead:
        - Without preprocessing: 1 image per page (original only)
        - With preprocessing: 2 images per page (deskewed + fully preprocessed, original eliminated)
        
        Returns:
            Optimal batch size for current hardware and target DPI
        """
        use_gpu = self.use_gpu and self.capabilities['gpu_available']
        
        # Detect model type (mobile or server)
        from .ocr.config import detect_model_type, calculate_batch_size_for_dpi
        model_type = detect_model_type()
        
        # DPI-FIRST: If user specified DPI, calculate batch size for that DPI
        if hasattr(self, 'override_dpi') and self.override_dpi:
            target_dpi = self.override_dpi
            
            batch_size = calculate_batch_size_for_dpi(
                target_dpi=target_dpi,
                use_gpu=use_gpu,
                gpu_memory_gb=self.capabilities['gpu_memory_gb'],
                system_memory_gb=self.capabilities['system_memory_gb'],
                preprocessing_enabled=self.config.enable_intelligent_preprocessing,
                model_type=model_type
            )
            
            logger.info(
                f"DPI-first batch sizing: target_dpi={target_dpi}, "
                f"batch_size={batch_size} "
                f"({'GPU ' + str(self.capabilities['gpu_memory_gb']) + 'GB' if use_gpu else 'CPU'})"
            )
            
            return batch_size
        
        # LEGACY: No DPI specified, use old optimization (maximize batch size)
        # Get base batch size (assumes 300 DPI baseline)
        base_batch_size = get_optimal_batch_size(
            use_gpu=use_gpu,
            gpu_memory_gb=self.capabilities['gpu_memory_gb'],
            system_memory_gb=self.capabilities['system_memory_gb'],
            dpi=300,  # Baseline DPI
            model_type=model_type
        )
        
        # Apply memory multiplier if preprocessing is enabled
        if self.config.enable_intelligent_preprocessing:
            # Memory footprint increases from 1 image to 2 images per page
            # OPTIMIZED: Original images eliminated, only deskewed + preprocessed kept
            # Without preprocessing: 1 image = 7.5 MB per page (original RGB only)
            # With preprocessing: 2 images = 10 MB per page (7.5 MB deskewed RGB + 2.5 MB preprocessed grayscale)
            # Reduction factor: 7.5 / 10 = 0.75 (~25% reduction)
            memory_multiplier = 0.75
            adjusted_batch_size = max(1, int(base_batch_size * memory_multiplier))
            
            logger.info(
                f"Detected optimal batch size: {base_batch_size} (base) → "
                f"{adjusted_batch_size} (adjusted for 2-image preprocessing pipeline, "
                f"{memory_multiplier:.0%} multiplier)"
            )
            return adjusted_batch_size
        else:
            logger.info(f"Detected optimal batch size: {base_batch_size} (no preprocessing)")
            return base_batch_size

    def _calculate_safe_dpi_for_page(self, pdf: PDFProcessor, page_num: int, target_dpi: int = 300) -> int:
        """
        Calculate safe DPI for rendering a page based on its physical dimensions and available memory.

        This prevents GPU/RAM OOM errors from oversized PDFs with high-resolution embedded images.

        Args:
            pdf: PDFProcessor instance
            page_num: Page number (0-indexed)
            target_dpi: Desired DPI (default 300)

        Returns:
            Safe DPI that won't cause OOM (may be lower than target_dpi)
        """
        # Use already-open document if available, otherwise open it
        if hasattr(pdf, 'doc') and pdf.doc is not None:
            doc = pdf.doc
            page = doc[page_num]
            rect = page.rect
        else:
            # Fallback: open the document (for cases where pdf.doc isn't available)
            with fitz.open(pdf.file_path) as doc:
                page = doc[page_num]
                rect = page.rect

        # Calculate expected image dimensions at target DPI
        expected_width = int(rect.width * target_dpi / 72)
        expected_height = int(rect.height * target_dpi / 72)
        expected_pixels = expected_width * expected_height
        expected_gb = (expected_pixels * 3) / (1024 ** 3)

        # Get available memory (80% threshold for safety)
        if self.use_gpu:
            available_gb = self.capabilities.get('gpu_memory_gb', 4) * 0.8
        else:
            # For CPU, use 50% of system RAM as safe limit
            available_gb = self.capabilities.get('system_memory_gb', 16) * 0.5

        # If expected memory exceeds safe threshold, reduce DPI
        if expected_gb > available_gb:
            # Calculate DPI that would fit in memory
            safe_scale = (available_gb / expected_gb) ** 0.5
            safe_dpi = max(150, int(target_dpi * safe_scale))  # Minimum 150 DPI

            logger.warning(
                f"Page {page_num + 1} at {target_dpi} DPI would use {expected_gb:.2f}GB "
                f"({expected_width}x{expected_height}px). Reducing to {safe_dpi} DPI "
                f"to fit in {available_gb:.2f}GB available memory."
            )

            return safe_dpi

        return target_dpi

    def _downsample_for_ocr_if_needed(self, image: np.ndarray) -> np.ndarray:
        """
        Downsample image if it exceeds safe dimensions for PaddleOCR processing.

        PaddleOCR's internal memory requirements are ~80× the input image size.
        A 4382×4166 image (18.3 MP, 55MB input) requires 4.4GB for processing.

        Args:
            image: Input image as numpy array

        Returns:
            Downsampled image if needed, or original image
        """
        height, width = image.shape[:2]
        megapixels = (height * width) / 1_000_000

        # Calculate safe megapixel limit based on available memory
        # Empirical: ~0.24 GB per megapixel for PaddleOCR processing
        if self.use_gpu:
            available_gb = self.capabilities.get('gpu_memory_gb', 4) * 0.8
        else:
            available_gb = self.capabilities.get('system_memory_gb', 16) * 0.5

        safe_megapixels = available_gb / 0.24  # GB per megapixel from empirical testing
        max_dimension = int((safe_megapixels * 1_000_000) ** 0.5)  # Assume square for max per side

        # Check if image exceeds safe dimensions
        if width > max_dimension or height > max_dimension or megapixels > safe_megapixels:
            # Calculate scale factor to fit within safe dimensions
            scale_factor = min(
                max_dimension / width if width > max_dimension else 1.0,
                max_dimension / height if height > max_dimension else 1.0,
                (safe_megapixels / megapixels) ** 0.5
            )

            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            logger.warning(
                f"Image too large for OCR processing: {width}x{height} ({megapixels:.1f}MP). "
                f"Downsampling to {new_width}x{new_height} ({(new_width*new_height)/1_000_000:.1f}MP) "
                f"to fit in {available_gb:.1f}GB memory limit."
            )

            # Downsample using high-quality interpolation
            downsampled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return downsampled

        return image

    def _initialize_services(self) -> None:
        """Initialize OCR service and VRAM monitor (lazy initialization)"""
        if self.ocr_service is None:
            # Check if we have a cached service from module initialization
            if self.ocr_service_cache is not None:
                logger.info("Using pre-initialized OCR service from cache")
                self.ocr_service = self.ocr_service_cache
                
                # Service already has session started and engines loaded
                engine_info = self.ocr_service.get_engine_info()
                logger.info(
                    f"OCR engine ready (from cache): {engine_info['engine']}, "
                    f"GPU={engine_info['gpu_enabled']}, "
                    f"memory={engine_info['memory_usage_mb']:.1f}MB"
                )
            else:
                # Fallback: Create new service (original behavior)
                logger.info("Initializing OCR service with engine pooling...")
                self.ocr_service = OCRService(
                    gpu=self.use_gpu,
                    engine="paddleocr",  # Use PaddleOCR as primary
                    fallback_enabled=True,
                    use_pooling=True  # Enable engine pooling for reuse
                )

                if not self.ocr_service.is_available():
                    raise RuntimeError("Failed to initialize OCR service")
                
                # Start batch session (keeps engines alive for reuse)
                self.ocr_service.start_session()

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

        if self.preprocessor is None and self.config.enable_intelligent_preprocessing:
            logger.info("Initializing intelligent preprocessor...")
            self.preprocessor = IntelligentPreprocessor(
                allow_destructive=self.config.preprocessing_allow_destructive,
                enable_validation=self.config.preprocessing_enable_validation,
                min_quality_improvement=self.config.preprocessing_min_quality_improvement,
                min_ssim=self.config.preprocessing_min_ssim
            )

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

    def _apply_deskew_only(self, image: np.ndarray) -> np.ndarray:
        """
        Apply ONLY deskew to an RGB image (visual improvement for final output).
        
        This is separate from full preprocessing (denoise/sharpen/binarize)
        which are OCR-only enhancements not appropriate for final output.
        
        Uses Hough line transform to detect document rotation and correct it.
        
        Args:
            image: Input image (RGB numpy array)
            
        Returns:
            Deskewed image (RGB) or original if deskew fails/not needed
        """
        try:
            import cv2
            
            # Make a copy to avoid modifying original
            if len(image.shape) != 3:
                logger.debug("Deskew requires RGB image, received grayscale")
                return image
            
            # Step 1: Convert to grayscale for angle detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Step 2: Apply binary threshold (Otsu's method)
            # This helps with edge detection
            _, binary = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # Step 3: Detect edges using Canny
            edges = cv2.Canny(binary, 50, 150, apertureSize=3)
            
            # Step 4: Use Hough line transform to detect dominant lines
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=100,
                minLineLength=100,
                maxLineGap=10
            )
            
            if lines is None or len(lines) == 0:
                logger.debug("No lines detected for deskew, using original")
                return image
            
            # Step 5: Calculate angles of all detected lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)
            
            # Step 6: Find median angle (more robust than mean)
            median_angle = np.median(angles)
            
            # Step 7: Normalize angle to [-45, 45] range
            # Documents are typically rotated by small angles
            if median_angle < -45:
                median_angle = 90 + median_angle
            elif median_angle > 45:
                median_angle = median_angle - 90
            
            # Step 8: Only deskew if angle is significant (> 0.5 degrees)
            if abs(median_angle) < 0.5:
                logger.debug(f"Skew angle {median_angle:.2f}° too small, no deskew needed")
                return image
            
            logger.debug(f"Detected skew angle: {median_angle:.2f}°, applying deskew")
            
            # Step 9: Rotate RGB image by detected angle
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            
            # Calculate new bounding box size to avoid cropping
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_width = int((height * sin) + (width * cos))
            new_height = int((height * cos) + (width * sin))
            
            # Adjust rotation matrix for new size
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            # Apply rotation to RGB image
            deskewed = cv2.warpAffine(
                image,
                rotation_matrix,
                (new_width, new_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)  # White background
            )
            
            logger.debug(f"Deskewed image by {median_angle:.2f}°")
            return deskewed
                
        except Exception as e:
            logger.warning(f"Deskew failed: {e}, using original image")
            return image
    
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

        # Use configurable threshold (lowered from 0.30 to support receipts/forms with special chars)
        if alphanumeric_ratio < self.config.min_alphanumeric_ratio:
            logger.debug(
                f"Page {page_num}: Alphanumeric ratio {alphanumeric_ratio:.1%} below threshold "
                f"{self.config.min_alphanumeric_ratio:.1%}"
            )
            return False, f"Low alphanumeric ratio ({alphanumeric_ratio:.1%})"

        # Check for common OCR failure patterns
        # Only reject if ENTIRE text is garbage AND text is very short
        # This prevents rejecting legitimate content like "--- Page 5 of 10 ---"
        stripped_text = text.strip()
        if len(stripped_text) < 10 and stripped_text in ['|||||||', '###', '......', '------', '||||', '####']:
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

        Now uses engine pooling to avoid redundant initialization.

        Args:
            pdf: PDFProcessor instance
            page_num: Page number
            engine: "paddleocr" or "tesseract"
            dpi: DPI setting (300, 400, etc.)

        Returns:
            Extracted text
        """
        # Switch to requested engine using pooling (no re-initialization!)
        if self.ocr_service.engine_name != engine:
            logger.info(f"Switching from {self.ocr_service.engine_name} to {engine}")
            self.ocr_service.switch_engine(engine)

        try:
            # Render at requested DPI
            image = pdf.render_page_to_image(page_num, dpi=dpi)
            text = self.ocr_service.extract_text_from_array(image)
            del image
            return text
        except Exception as e:
            logger.error(f"OCR failed with {engine} at {dpi} DPI: {e}")
            return ""

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

        # Use configurable thresholds (lowered from 0.70 and 0.05 for more sensitive comparison)
        quality_threshold = self.config.min_quality_threshold
        improvement_margin = self.config.min_quality_improvement_margin

        # If original is already high quality, require improvement to justify replacement
        if original_score >= quality_threshold:
            if ocr_score > original_score + improvement_margin:
                logger.info(
                    f"Page {page_num+1}: Replacing with OCR - quality improved "
                    f"({original_score:.2%} → {ocr_score:.2%})"
                )
                return True, f"OCR improved quality ({original_score:.2%} → {ocr_score:.2%})"
            else:
                logger.debug(
                    f"Page {page_num+1}: Keeping original - quality sufficient "
                    f"({original_score:.2%}), OCR not significantly better ({ocr_score:.2%})"
                )
                return False, f"Original quality sufficient ({original_score:.2%}), OCR not better ({ocr_score:.2%})"

        # If original is low quality, accept any improvement
        if ocr_score > original_score:
            logger.info(
                f"Page {page_num+1}: Replacing with OCR - improved low-quality text "
                f"({original_score:.2%} → {ocr_score:.2%})"
            )
            return True, f"OCR improved quality ({original_score:.2%} → {ocr_score:.2%})"
        else:
            logger.debug(
                f"Page {page_num+1}: Keeping original - OCR did not improve "
                f"({original_score:.2%} → {ocr_score:.2%})"
            )
            return False, f"OCR did not improve quality ({original_score:.2%} → {ocr_score:.2%})"

    def _clean_rebuild_page_with_ocr(
        self,
        doc: fitz.Document,
        page_num: int,
        ocr_text: str,
        ocr_result: Optional[object] = None,
        rebuild_strategy: str = "full",
        deskewed_image: Optional[np.ndarray] = None
    ) -> None:
        """
        Add OCR text layer to page using specified strategy.

        Strategies:
        - "overlay": Add text overlay only, preserve original content (minimal size increase)
        - "full": Full rebuild with image re-rendering and compression (for scanned pages)

        Args:
            doc: PyMuPDF document
            page_num: Page number to process
            ocr_text: OCR text to embed
            ocr_result: Optional OCR result with bounding boxes for coordinate mapping
            rebuild_strategy: "overlay" or "full"
        """
        try:
            original_page = doc[page_num]
            page_rect = original_page.rect

            if rebuild_strategy == "overlay":
                # Strategy 1: Text overlay only (no re-rendering)
                # Best for pages with good visual content but poor/no text layer
                logger.debug(f"Page {page_num+1}: Adding text overlay (no re-render)")
                self._add_text_overlay(original_page, page_rect, ocr_text, ocr_result)
                
            else:
                # Strategy 2: Full rebuild with compression (for scanned pages)
                logger.debug(f"Page {page_num+1}: Full rebuild with compression")
                self._full_rebuild_with_compression(
                    doc, page_num, original_page, page_rect, ocr_text, ocr_result, deskewed_image
                )

            logger.debug(f"Page {page_num+1} processed successfully with strategy: {rebuild_strategy}")

        except Exception as e:
            logger.error(f"Failed to process page {page_num+1}: {e}", exc_info=True)
            raise

    def _clean_text_for_pdf(self, text: str) -> str:
        """
        Remove problematic characters from text for PDF insertion.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text safe for PDF insertion
        """
        # Remove control characters except newline/tab
        text = ''.join(ch for ch in text if ch in '\n\t' or not unicodedata.category(ch).startswith('C'))
        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        return text

    def _insert_text_with_bbox(
        self,
        page: fitz.Page,
        text_line: str,
        pdf_bbox: 'PDFBBox',
        page_rect: fitz.Rect
    ) -> bool:
        """
        Insert text at precise position using OCR bounding box.
        
        Args:
            page: PyMuPDF page
            text_line: Text to insert
            pdf_bbox: PDF coordinates from coordinate mapper
            page_rect: Page rectangle (for bounds checking)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Calculate font size from bbox height
            bbox_height = pdf_bbox.height
            fontsize = max(1.0, bbox_height * 0.9)  # 90% of height, minimum 1pt
            
            # Get font for descender calculation
            try:
                font = fitz.Font("helv")
            except Exception as e:
                logger.warning(f"Font 'helv' not available: {e}, using default")
                font = fitz.Font()  # Use default font
            
            # Calculate insertion point (baseline-left)
            # Start at bottom-left, adjust for descenders
            insertion_point = fitz.Point(pdf_bbox.x0, pdf_bbox.y1)
            
            # Adjust for descenders (descender is negative)
            insertion_point = insertion_point + (0, font.descender * fontsize)
            
            # Bounds check - ensure point is within page
            if not page_rect.contains(insertion_point):
                logger.warning(
                    f"Insertion point {insertion_point} outside page rect {page_rect}, "
                    f"clamping to page bounds"
                )
                insertion_point.x = max(page_rect.x0, min(insertion_point.x, page_rect.x1))
                insertion_point.y = max(page_rect.y0, min(insertion_point.y, page_rect.y1))
            
            # Clean text
            cleaned_text = self._clean_text_for_pdf(text_line.strip())
            if not cleaned_text:
                return True  # Empty text after cleaning, nothing to insert
            
            # Insert text with invisible render mode
            page.insert_text(
                insertion_point,
                cleaned_text,
                fontsize=fontsize,
                fontname="helv",
                color=(1, 1, 1),  # White (invisible on white background)
                render_mode=3,    # INVISIBLE - searchable but not visible
                overlay=True
            )
            
            logger.debug(
                f"Inserted text at ({insertion_point.x:.1f}, {insertion_point.y:.1f}), "
                f"fontsize={fontsize:.1f}, bbox_height={bbox_height:.1f}"
            )
            return True
            
        except Exception as e:
            logger.warning(f"Failed to insert text with bbox: {e}")
            return False

    def _insert_text_fallback(
        self,
        page: fitz.Page,
        text: str,
        page_rect: fitz.Rect
    ) -> bool:
        """
        Insert text without coordinate mapping using TextWriter.
        
        Distributes text lines across page height for better searchability.
        
        Args:
            page: PyMuPDF page
            text: Full text to insert
            page_rect: Page rectangle
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clean and validate text
            cleaned_text = self._clean_text_for_pdf(text)
            if not cleaned_text or not cleaned_text.strip():
                logger.debug("Empty text after cleaning, skipping insertion")
                return True  # Not an error, just nothing to insert
            
            # Truncate very long text
            MAX_TEXT_LENGTH = 100_000  # 100KB
            if len(cleaned_text) > MAX_TEXT_LENGTH:
                logger.warning(
                    f"Text too long ({len(cleaned_text)} chars), "
                    f"truncating to {MAX_TEXT_LENGTH}"
                )
                cleaned_text = cleaned_text[:MAX_TEXT_LENGTH] + "... [truncated]"
            
            # Create TextWriter for this page
            tw = fitz.TextWriter(page_rect)
            
            # Split text into lines
            lines = cleaned_text.split('\n')
            if not lines:
                return True  # Empty text, nothing to do
            
            # Calculate line spacing
            margin = 10  # pixels from edge
            available_height = page_rect.height - (2 * margin)
            line_spacing = available_height / max(len(lines), 1)
            
            # Font size
            fontsize = 8  # Small, unobtrusive
            try:
                font = fitz.Font("helv")
            except Exception as e:
                logger.warning(f"Font 'helv' not available: {e}, using default")
                font = fitz.Font()  # Use default font
            
            # Insert each line
            for i, line in enumerate(lines):
                if not line.strip():
                    continue  # Skip empty lines
                
                # Calculate position (left margin, distributed vertically)
                x = page_rect.x0 + margin
                y = page_rect.y0 + margin + (i * line_spacing)
                
                # Adjust for baseline (descender)
                y = y + (fontsize * 0.8)  # Approximate baseline adjustment
                
                # Add to TextWriter
                tw.append(
                    (x, y),
                    line.strip(),
                    font=font,
                    fontsize=fontsize
                )
            
            # Write all text at once with invisible render mode
            tw.write_text(
                page,
                render_mode=3,  # INVISIBLE
                overlay=True
            )
            
            logger.debug(f"Inserted {len(lines)} lines using TextWriter fallback")
            return True
            
        except Exception as e:
            logger.error(f"TextWriter fallback failed: {e}", exc_info=True)
            return False

    def _verify_page_searchability(
        self,
        page: fitz.Page,
        page_num: int,
        expected_text_length: int
    ) -> None:
        """
        Verify that page has searchable text (diagnostic only).
        
        Args:
            page: Page to check
            page_num: Page number (for logging)
            expected_text_length: Expected text length from OCR
        """
        try:
            extracted = page.get_text()
            extracted_len = len(extracted.strip())
            
            if extracted_len == 0:
                logger.error(
                    f"Page {page_num+1}: VERIFICATION FAILED - No searchable text! "
                    f"(Expected ~{expected_text_length} chars)"
                )
            elif extracted_len < expected_text_length * 0.5:
                logger.warning(
                    f"Page {page_num+1}: VERIFICATION WARNING - Only {extracted_len} chars "
                    f"extracted (expected ~{expected_text_length})"
                )
            else:
                logger.debug(
                    f"Page {page_num+1}: VERIFICATION OK - {extracted_len} chars searchable"
                )
                
        except Exception as e:
            logger.warning(f"Page {page_num+1}: Verification failed: {e}")

    def _add_text_overlay(
        self,
        page: fitz.Page,
        page_rect: fitz.Rect,
        ocr_text: str,
        ocr_result: Optional[object] = None
    ) -> None:
        """
        Add invisible text overlay to existing page without re-rendering.

        Uses coordinate mapping if OCR result has bounding boxes.

        Args:
            page: PyMuPDF page to add text to
            page_rect: Page rectangle
            ocr_text: OCR text to add
            ocr_result: Optional OCR result with bounding boxes
        """
        # Check if we can use coordinate mapping
        if self.config.use_coordinate_mapping and ocr_result and hasattr(ocr_result, 'bbox') and ocr_result.bbox:
            # Use positioned text with coordinate mapping
            logger.info(f"Using coordinate mapping for text overlay")
            self._insert_positioned_text(page, page_rect, ocr_text, ocr_result)
        else:
            # FIXED: Use TextWriter fallback
            logger.warning("Using TextWriter fallback for text overlay")
            
            success = self._insert_text_fallback(page, ocr_text, page_rect)
            
            if success:
                logger.info(f"TextWriter fallback succeeded ({len(ocr_text)} chars)")
            else:
                logger.error("TextWriter fallback failed!")
                raise RuntimeError("Failed to add text overlay - TextWriter failed")

    def _full_rebuild_with_compression(
        self,
        doc: fitz.Document,
        page_num: int,
        original_page: fitz.Page,
        page_rect: fitz.Rect,
        ocr_text: str,
        ocr_result: Optional[object] = None,
        deskewed_image: Optional[np.ndarray] = None
    ) -> None:
        """
        Full page rebuild with image compression for scanned pages.
        
        IMPORTANT: Uses original page orientation (no rotation applied).
        Preserves the visual appearance of the source PDF.
        
        Args:
            doc: PyMuPDF document
            page_num: Page number
            original_page: Original page object
            page_rect: Page rectangle
            ocr_text: OCR text
            ocr_result: Optional OCR result with bounding boxes
            deskewed_image: Image for output (name kept for compatibility, but preserves original orientation)
        """
        # Step 1: Get page image (use provided image if available, otherwise render from PDF)
        if deskewed_image is not None:
            # Use pre-rendered image (preserves original orientation)
            logger.debug(f"Page {page_num+1}: Using original image for final output")
            img_for_output = deskewed_image
        else:
            # Fallback: render from original PDF
            logger.debug(f"Page {page_num+1}: Rendering from original PDF (no deskewed image)")
            pix = original_page.get_pixmap(dpi=300)
            # Convert pixmap to numpy array for consistent processing
            import io
            from PIL import Image
            img_data = pix.tobytes("ppm")
            img_for_output = np.array(Image.frombytes("RGB", (pix.width, pix.height), img_data))
            del pix

        # Step 2: Create new blank page
        new_page = doc.new_page(
            width=page_rect.width,
            height=page_rect.height
        )

        # Step 3: Insert image with compression
        if self.config.enable_compression:
            # Convert numpy array to compressed image data
            import io
            from PIL import Image
            
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(img_for_output.astype(np.uint8))
            
            # Compress as JPEG
            img_buffer = io.BytesIO()
            pil_img.save(
                img_buffer,
                format='JPEG',
                quality=self.config.image_compression_quality,
                optimize=True
            )
            img_buffer.seek(0)
            
            # Insert compressed image
            new_page.insert_image(page_rect, stream=img_buffer.getvalue())
            
        else:
            # Insert without compression
            # Convert numpy array to PIL Image, then to bytes
            import io
            from PIL import Image
            pil_img = Image.fromarray(img_for_output.astype(np.uint8))
            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            new_page.insert_image(page_rect, stream=img_buffer.getvalue())

        # Step 4: Add OCR text with coordinate mapping
        logger.info(f"Step 4: Adding OCR text ({len(ocr_text)} chars) to rebuilt page")
        
        if self.config.use_coordinate_mapping and ocr_result and hasattr(ocr_result, 'bbox') and ocr_result.bbox:
            self._insert_positioned_text(new_page, page_rect, ocr_text, ocr_result)
        else:
            # FIXED: Use TextWriter fallback
            logger.warning("Using TextWriter fallback in full rebuild")
            
            success = self._insert_text_fallback(new_page, ocr_text, page_rect)
            
            if success:
                logger.info(f"TextWriter succeeded in rebuilt page ({len(ocr_text)} chars)")
            else:
                logger.error("TextWriter failed in rebuilt page!")
                raise RuntimeError("Failed to add text to rebuilt page - TextWriter failed")

        # Step 5: Replace original page
        doc.delete_page(page_num)
        doc.move_page(doc.page_count - 1, page_num)

        # Cleanup
        del img_for_output
        import gc
        gc.collect()

    def _insert_positioned_text(
        self,
        page: fitz.Page,
        page_rect: fitz.Rect,
        ocr_text: str,
        ocr_result: object
    ) -> None:
        """
        Insert text at precise locations using OCR bounding boxes.

        Args:
            page: PyMuPDF page
            page_rect: Page rectangle
            ocr_text: Full OCR text
            ocr_result: OCR result with bounding boxes
        """
        try:
            from .ocr.coordinate_mapper import CoordinateMapper
            
            mapper = CoordinateMapper()
            
            # Get text lines and bounding boxes from OCR result
            # OCR result format varies by engine, handle both
            if hasattr(ocr_result, 'raw_result') and ocr_result.raw_result:
                text_lines, bboxes = self._extract_text_and_boxes_from_ocr(ocr_result.raw_result)
            elif isinstance(ocr_result.bbox, list) and ocr_result.bbox:
                # Simple case: single bbox for entire text
                text_lines = [ocr_text]
                bboxes = [ocr_result.bbox]
            else:
                # FIXED: Use TextWriter fallback
                logger.warning(f"No bounding boxes available, using TextWriter fallback")
                success = self._insert_text_fallback(page, ocr_text, page_rect)
                if not success:
                    logger.error("TextWriter fallback failed!")
                return
            
            # Get image dimensions from OCR result (actual dimensions used for OCR)
            # This accounts for any preprocessing that may have changed dimensions
            dpi = 300  # Default DPI for coordinate mapping
            if hasattr(ocr_result, 'image_width') and ocr_result.image_width:
                image_width = ocr_result.image_width
                image_height = ocr_result.image_height
                logger.debug(f"Using actual OCR image dimensions: {image_width}x{image_height}")
            else:
                # Fallback: calculate from page_rect (assume 300 DPI rendering)
                image_width = int(page_rect.width * dpi / 72)
                image_height = int(page_rect.height * dpi / 72)
                logger.debug(f"Using calculated dimensions from page_rect: {image_width}x{image_height}")
            
            # CRITICAL FIX: Process ALL text lines, not just those with bounding boxes
            # Previous code used zip() which silently dropped text without matching bboxes
            inserted_count = 0
            failed_count = 0
            skipped_no_bbox = 0

            # Warn if mismatch between text lines and bounding boxes
            if len(text_lines) != len(bboxes):
                logger.warning(
                    f"Mismatch: {len(text_lines)} text lines but {len(bboxes)} bounding boxes. "
                    f"Will process all text, using fallback for lines without bboxes."
                )

            # Process ALL text lines (iterate by index to handle mismatches)
            for i, text_line in enumerate(text_lines):
                if not text_line or not text_line.strip():
                    continue

                # Check if we have a bounding box for this text line
                if i < len(bboxes):
                    bbox = bboxes[i]

                    # Convert bbox to PDF coordinates
                    pdf_bbox = mapper.image_to_pdf_coords(
                        bbox=bbox,
                        image_width=image_width,
                        image_height=image_height,
                        page_rect=page_rect,
                        image_dpi=dpi
                    )

                    # Insert with coordinate mapping
                    success = self._insert_text_with_bbox(page, text_line, pdf_bbox, page_rect)
                    if success:
                        inserted_count += 1
                    else:
                        failed_count += 1
                else:
                    # No bounding box available - this text would have been DROPPED by old code!
                    logger.debug(f"Text line {i+1} has no bbox, will be included in fallback text")
                    skipped_no_bbox += 1

            logger.info(
                f"Coordinate mapping: {inserted_count} positioned text elements inserted, "
                f"{failed_count} failed, {skipped_no_bbox} without bboxes"
            )

            # CRITICAL FIX: If some text had no bounding boxes, append it using fallback
            if skipped_no_bbox > 0:
                # Collect text lines that didn't have bounding boxes
                text_without_bboxes = []
                for i, text_line in enumerate(text_lines):
                    if i >= len(bboxes) and text_line.strip():
                        text_without_bboxes.append(text_line.strip())

                if text_without_bboxes:
                    combined_text = "\n".join(text_without_bboxes)
                    logger.warning(
                        f"Adding {len(text_without_bboxes)} text lines without bboxes "
                        f"({len(combined_text)} chars) using TextWriter fallback"
                    )
                    success = self._insert_text_fallback(page, combined_text, page_rect)
                    if success:
                        logger.info(f"Successfully inserted {len(combined_text)} chars without coordinate mapping")
                    else:
                        logger.error(f"Failed to insert text without bboxes!")
            
        except Exception as e:
            logger.warning(f"Coordinate mapping failed: {e}, using TextWriter fallback")
            # FIXED: Use TextWriter fallback
            success = self._insert_text_fallback(page, ocr_text, page_rect)
            if not success:
                logger.error("TextWriter fallback ALSO failed!")

    def _extract_text_and_boxes_from_ocr(self, raw_result) -> tuple:
        """
        Extract text lines and bounding boxes from raw OCR result.

        Handles PaddleOCR 3.x format.

        Args:
            raw_result: Raw OCR result from engine

        Returns:
            Tuple of (text_lines, bboxes) lists
        """
        text_lines = []
        bboxes = []
        
        try:
            # PaddleOCR 3.x format: dict with 'rec_texts', 'rec_scores', 'rec_polys'
            if isinstance(raw_result, dict):
                texts = raw_result.get('rec_texts', [])
                polys = raw_result.get('rec_polys', [])

                for text, poly in zip(texts, polys):
                    if text and text.strip():
                        text_lines.append(text.strip())
                        # Convert numpy array to list if needed
                        if hasattr(poly, 'tolist'):
                            poly = poly.tolist()
                        bboxes.append(poly)

            # PaddleOCR 3.x wrapped format: list containing dict
            elif isinstance(raw_result, list) and len(raw_result) > 0:
                first_item = raw_result[0]

                # Check if it's the new wrapped format with nested dict
                if isinstance(first_item, dict) and 'rec_texts' in first_item:
                    texts = first_item.get('rec_texts', [])
                    polys = first_item.get('rec_polys', [])

                    for text, poly in zip(texts, polys):
                        if text and text.strip():
                            text_lines.append(text.strip())
                            # Convert numpy array to list if needed
                            if hasattr(poly, 'tolist'):
                                poly = poly.tolist()
                            bboxes.append(poly)

                # PaddleOCR 2.x format: list of [bbox, (text, confidence)]
                elif len(first_item) >= 2 and isinstance(first_item, (list, tuple)):
                    for item in raw_result:
                        if len(item) >= 2:
                            bbox, (text, score) = item[0], item[1]
                            if text and text.strip():
                                text_lines.append(text.strip())
                                bboxes.append(bbox)
                            
        except Exception as e:
            logger.error(
                f"Failed to extract text and boxes from raw_result. "
                f"Exception: {e}, Type: {type(e)}, "
                f"raw_result type: {type(raw_result)}, raw_result value: {raw_result}"
            )
        
        logger.info(f"Extracted {len(text_lines)} text lines and {len(bboxes)} bounding boxes from raw OCR result")
        return text_lines, bboxes

    def _process_page_batch(
        self,
        pdf: PDFProcessor,
        page_numbers: List[int],
        file_name: str
    ) -> List[tuple]:
        """
        Process a batch of PDF pages with OCR and retry logic.
        
        STREAMING ARCHITECTURE:
        Returns (text, ocr_result, original_img) tuples for IMMEDIATE consumption.
        Caller MUST free original_img after use to maintain constant memory.
        
        Memory per batch: batch_size × 10 MB
        - Original RGB: 7.5 MB per page (preserves original orientation)
        - Preprocessed grayscale: 2.5 MB per page (freed before return)

        Args:
            pdf: PDFProcessor instance
            page_numbers: List of page numbers to process
            file_name: Name of file being processed (for logging)

        Returns:
            List of (text, ocr_result, deskewed_img) tuples for each page
            IMPORTANT: Caller must immediately use and free deskewed_img!
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
                results = []
                for page_num in page_numbers:
                    text, ocr_result, deskewed_img = self._process_single_page(pdf, page_num, file_name)
                    results.append((text, ocr_result, deskewed_img))
                return results

        # Normal batch processing
        def process_batch_operation():
            # Render pages to images FROM ORIGINAL PDF
            # Images preserve original orientation for final output
            # Full preprocessing (including grayscale deskew) applied separately for OCR quality
            deskewed_images = []
            for page_num in page_numbers:
                # Calculate safe DPI for this page (prevents OOM from oversized PDFs)
                safe_dpi = self._calculate_safe_dpi_for_page(pdf, page_num, target_dpi=300)
                # Render from PDF with safe DPI
                image = pdf.render_page_to_image(page_num, dpi=safe_dpi)
                
                # KEEP ORIGINAL ORIENTATION for final output
                # Note: Variable name is "deskewed_images" for compatibility,
                # but we now preserve original page orientation
                # Full preprocessing (including grayscale deskew) still applied for OCR quality
                deskewed_images.append(image)

            # STEP 2: Apply FULL PREPROCESSING to deskewed images (OCR quality)
            # This includes denoise, sharpen, contrast, binarize - NOT for final output!
            images_for_ocr = []
            image_dimensions = []  # Track actual dimensions for coordinate mapping
            if self.preprocessor is not None:
                logger.debug(f"Applying full preprocessing to {len(deskewed_images)} images (FOR OCR ONLY)")
                for idx, img in enumerate(deskewed_images):
                    result = self.preprocessor.process(img)
                    # Use preprocessed OR deskewed based on validation
                    final_ocr_image = result.image if result.used_preprocessed else img
                    images_for_ocr.append(final_ocr_image)

                    # Store actual dimensions for coordinate mapping
                    image_dimensions.append({
                        'width': final_ocr_image.shape[1],
                        'height': final_ocr_image.shape[0]
                    })

                    if result.used_preprocessed:
                        logger.debug(
                            f"Page {page_numbers[idx]+1}: Using fully preprocessed for OCR - "
                            f"{result.techniques_applied}"
                        )
                    else:
                        logger.debug(f"Page {page_numbers[idx]+1}: Using deskewed only for OCR (preprocessing rejected)")
                        # CRITICAL: Free rejected preprocessed image immediately
                        if result.image is not None and result.image is not img:
                            del result.image
            else:
                # No preprocessing - use deskewed images directly (no copy needed)
                images_for_ocr = deskewed_images
                for img in images_for_ocr:
                    image_dimensions.append({
                        'width': img.shape[1],
                        'height': img.shape[0]
                    })

            # CRITICAL: Downsample images if they exceed safe dimensions for PaddleOCR
            # PaddleOCR's internal memory is ~80× input size, so a 55MB image can need 4.4GB
            downsampled_for_ocr = []
            downsample_scale_factors = []  # Track scale factors for coordinate mapping
            for idx, img in enumerate(images_for_ocr):
                original_height, original_width = img.shape[:2]
                downsampled = self._downsample_for_ocr_if_needed(img)
                downsampled_for_ocr.append(downsampled)

                # Calculate scale factor if downsampling occurred
                if downsampled is not img:
                    downsampled_height, downsampled_width = downsampled.shape[:2]
                    scale_x = original_width / downsampled_width
                    scale_y = original_height / downsampled_height
                    downsample_scale_factors.append((scale_x, scale_y))
                else:
                    downsample_scale_factors.append((1.0, 1.0))  # No scaling needed

            # Process with OCR - get full results with bounding boxes
            # IMPORTANT: OCR runs on fully preprocessed images for better text quality
            ocr_results = self.ocr_service.process_batch_with_boxes(downsampled_for_ocr)

            # Verify results match input count
            if len(ocr_results) != len(downsampled_for_ocr):
                logger.error(
                    f"OCR batch result mismatch: expected {len(downsampled_for_ocr)}, "
                    f"got {len(ocr_results)}"
                )
                # Pad with empty results
                from .ocr.base import OCRResult
                while len(ocr_results) < len(downsampled_for_ocr):
                    ocr_results.append(OCRResult(text="", confidence=0.0))

            # Scale bounding boxes back to original image coordinates if downsampling occurred
            for idx, (ocr_result, scale_factors) in enumerate(zip(ocr_results, downsample_scale_factors)):
                scale_x, scale_y = scale_factors

                # Only scale if downsampling occurred (scale != 1.0)
                if scale_x != 1.0 or scale_y != 1.0:
                    if ocr_result.bbox:
                        scaled_bboxes = []
                        for bbox in ocr_result.bbox:
                            # Scale each point in the bounding box
                            scaled_bbox = [[int(x * scale_x), int(y * scale_y)] for x, y in bbox]
                            scaled_bboxes.append(scaled_bbox)
                        ocr_result.bbox = scaled_bboxes
                        logger.debug(
                            f"Page {page_numbers[idx]+1}: Scaled {len(scaled_bboxes)} bounding boxes "
                            f"by {scale_x:.2f}x, {scale_y:.2f}y (downsampling correction)"
                        )

            # Attach image dimensions to OCR results for coordinate mapping
            for ocr_result, dims in zip(ocr_results, image_dimensions):
                ocr_result.image_width = dims['width']
                ocr_result.image_height = dims['height']

            # Cleanup fully preprocessed and downsampled images (no longer needed)
            # IMPORTANT: We only keep deskewed_images for final PDF output
            # CRITICAL: Explicit per-item cleanup to prevent list reference holding
            for img in images_for_ocr:
                if img is not None:
                    del img
            images_for_ocr.clear()
            del images_for_ocr

            for img in downsampled_for_ocr:
                if img is not None:
                    del img
            downsampled_for_ocr.clear()
            del downsampled_for_ocr

            # Force garbage collection to free numpy arrays immediately
            gc.collect()

            # Return OCR results WITH original images for final output
            # Images preserve original page orientation
            return ocr_results, deskewed_images

        # Execute with retry
        ocr_results, deskewed_images = self._retry_with_backoff(
            process_batch_operation,
            f"OCR batch ({len(page_numbers)} pages from {file_name})"
        )

        # VALIDATE AND RETRY
        # Results now include: (text, ocr_result, deskewed_image)
        # OPTIMIZED: Only 2 images per page (original eliminated)
        validated_results = []
        for idx, (page_num, ocr_result) in enumerate(zip(page_numbers, ocr_results)):
            text = ocr_result.text
            
            # DEBUG: Log what OCR extracted before validation
            logger.debug(f"Page {page_num+1}: Raw OCR extracted {len(text)} chars: '{text[:100]}...'")

            # Validate OCR output
            is_valid, reason = self._validate_ocr_output(text, page_num)

            if is_valid:
                # Include original image with result (preserves orientation)
                validated_results.append((
                    text, 
                    ocr_result,
                    deskewed_images[idx]
                ))
                logger.debug(f"Page {page_num+1}: OCR valid ({len(text)} chars) - {reason}")
            else:
                logger.warning(f"Page {page_num+1}: OCR validation FAILED - {reason} (extracted {len(text)} chars)")

                # Retry with fallback
                retry_text, method = self._retry_ocr_with_fallback(pdf, page_num, file_name)

                if retry_text:
                    # Create new OCR result with retry text
                    from .ocr.base import OCRResult
                    retry_result = OCRResult(
                        text=retry_text,
                        confidence=0.8,  # Assume decent confidence for retry
                        bbox=None  # No bounding boxes for retry
                    )
                    # Retry doesn't have dimension info, use defaults
                    retry_result.image_width = None
                    retry_result.image_height = None
                    validated_results.append((
                        retry_text, 
                        retry_result,
                        deskewed_images[idx]
                    ))
                    logger.info(f"Page {page_num+1}: Retry succeeded with {method}")
                else:
                    # All retries failed - use empty result
                    logger.error(f"Page {page_num+1}: All OCR attempts failed")
                    from .ocr.base import OCRResult
                    empty_result = OCRResult(text="", confidence=0.0, error="All OCR attempts failed")
                    empty_result.image_width = None
                    empty_result.image_height = None
                    validated_results.append((
                        "", 
                        empty_result,
                        deskewed_images[idx]
                    ))

        # Cleanup images after validation
        del deskewed_images
        
        # Force garbage collection after batch completes
        gc.collect()
        
        return validated_results

    def _process_single_page(
        self,
        pdf: PDFProcessor,
        page_num: int,
        file_name: str
    ) -> tuple:
        """
        Process a single page with OCR and retry logic.

        Args:
            pdf: PDFProcessor instance
            page_num: Page number to process
            file_name: Name of file being processed (for logging)

        Returns:
            Tuple of (text, ocr_result, original_image) - preserves page orientation
        """
        def process_page_operation():
            # Calculate safe DPI for this page (prevents OOM from oversized PDFs)
            safe_dpi = self._calculate_safe_dpi_for_page(pdf, page_num, target_dpi=300)
            # Render original image with safe DPI
            image = pdf.render_page_to_image(page_num, dpi=safe_dpi)
            
            # KEEP ORIGINAL ORIENTATION for final output
            # Variable name is "deskewed_image" for compatibility, but preserves original orientation
            deskewed_image = image
            
            # Apply full preprocessing for OCR (includes grayscale deskew for accuracy)
            if self.preprocessor is not None:
                preprocess_result = self.preprocessor.process(image)
                image_for_ocr = preprocess_result.image if preprocess_result.used_preprocessed else image

                # CRITICAL: Free rejected preprocessed image immediately to prevent leak
                if not preprocess_result.used_preprocessed:
                    if preprocess_result.image is not None and preprocess_result.image is not image:
                        del preprocess_result.image
            else:
                image_for_ocr = image  # No copy needed - just use reference

            # Store original dimensions before downsampling
            original_height, original_width = image_for_ocr.shape[:2]

            # CRITICAL: Downsample image if it exceeds safe dimensions for PaddleOCR
            # PaddleOCR's internal memory is ~80× input size, so a 55MB image can need 4.4GB
            downsampled_for_ocr = self._downsample_for_ocr_if_needed(image_for_ocr)

            # Calculate scale factor if downsampling occurred
            if downsampled_for_ocr is not image_for_ocr:
                downsampled_height, downsampled_width = downsampled_for_ocr.shape[:2]
                scale_x = original_width / downsampled_width
                scale_y = original_height / downsampled_height
            else:
                scale_x, scale_y = 1.0, 1.0

            # Get full OCR result with bounding boxes
            ocr_results = self.ocr_service.process_batch_with_boxes([downsampled_for_ocr])

            # Scale bounding boxes back to original coordinates if downsampling occurred
            if ocr_results and (scale_x != 1.0 or scale_y != 1.0):
                ocr_result = ocr_results[0]
                if ocr_result.bbox:
                    scaled_bboxes = []
                    for bbox in ocr_result.bbox:
                        scaled_bbox = [[int(x * scale_x), int(y * scale_y)] for x, y in bbox]
                        scaled_bboxes.append(scaled_bbox)
                    ocr_result.bbox = scaled_bboxes
                    logger.debug(
                        f"Page {page_num+1}: Scaled {len(scaled_bboxes)} bounding boxes "
                        f"by {scale_x:.2f}x, {scale_y:.2f}y (downsampling correction)"
                    )

            # Attach ORIGINAL image dimensions for coordinate mapping (not downsampled!)
            if ocr_results:
                ocr_results[0].image_width = original_width
                ocr_results[0].image_height = original_height

            # Cleanup OCR images (only delete if they're separate objects)
            if downsampled_for_ocr is not image_for_ocr:
                del downsampled_for_ocr
            del image_for_ocr

            # Return OCR result with original image (preserves orientation)
            return (ocr_results[0] if ocr_results else None, deskewed_image)

        try:
            ocr_result, deskewed_img = self._retry_with_backoff(
                process_page_operation,
                f"OCR page {page_num + 1} from {file_name}"
            )
            
            if ocr_result:
                return (ocr_result.text, ocr_result, deskewed_img)
            else:
                from .ocr.base import OCRResult
                empty_result = OCRResult(text="", confidence=0.0, error="No OCR result")
                empty_result.image_width = None
                empty_result.image_height = None
                return ("", empty_result, deskewed_img)
                
        except Exception as e:
            logger.error(f"Failed to process page {page_num + 1} after retries: {e}")
            from .ocr.base import OCRResult
            empty_result = OCRResult(text="", confidence=0.0, error=str(e))
            empty_result.image_width = None
            empty_result.image_height = None
            # Return empty image as well
            empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
            return ("", empty_result, empty_img)  # Return empty string on failure

    def _process_single_file(
        self,
        file_path: str,
        output_dir: str,
        file_idx: int,
        total_files: int
    ) -> Dict[str, Any]:
        """
        Process a single PDF file with STREAMING ASSEMBLY.
        
        Memory-optimized approach:
        - Phase 1: Scan pages to identify text layer vs OCR needed
        - Phase 2: Process OCR batches in REVERSE order
        - Phase 3: IMMEDIATELY assemble each page to PDF (no accumulation)
        - Peak RAM: batch_size × 10 MB (constant, not proportional to PDF size!)

        Args:
            file_path: Path to PDF file
            output_dir: Directory to save processed file
            file_idx: Current file index (1-based)
            total_files: Total number of files

        Returns:
            Dictionary with processing results
        """
        file_name = Path(file_path).name
        logger.info(f"Processing file {file_idx}/{total_files}: {file_name}")

        # Emit "processing" status
        started_at = time.time()
        self.file_start_times[file_path] = started_at
        queued_at = self.file_queue_times.get(file_path)
        
        if self.file_status_callback:
            self.file_status_callback(
                file_path=file_path,
                file_name=file_name,
                file_index=file_idx,
                total_files=total_files,
                status="processing",
                queued_at=queued_at,
                started_at=started_at
            )
        
        result = {
            'status': 'failed',
            'pages_processed': 0,
            'pages_ocr': 0,
            'pages_text_layer': 0,
            'error': None,
            'output_path': None
        }

        # Track fitz document for cleanup
        doc = None

        # Initialize memory monitor for leak detection
        if self.memory_monitor is None:
            self.memory_monitor = MemoryMonitor(enable_tracemalloc=False)

        # Start memory tracking for this file
        self.memory_monitor.start_tracking(f"file_{file_idx}_{file_name}")

        try:
            # Open PDF for analysis
            with PDFProcessor(file_path) as pdf:
                total_pages = pdf.get_page_count()
                logger.info(f"File has {total_pages} pages")

                # ============================================================
                # PHASE 1: Scan all pages to identify text layer vs OCR needed
                # ============================================================
                pages_needing_ocr = []
                text_layer_pages = {}  # {page_num: text}
                original_texts = {}  # For quality comparison

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
                        message=f"File {file_idx}/{total_files}: {file_name} - Scanning page {page_num + 1}/{total_pages}"
                    )

                    # Check for text layer
                    has_text, validation_info = pdf.has_valid_text_layer(page_num)

                    if has_text:
                        # Extract from text layer (fast)
                        text = pdf.extract_text(page_num)
                        text_layer_pages[page_num] = text
                        original_texts[page_num] = text
                        result['pages_text_layer'] += 1
                        
                        # DIAGNOSTIC: Log detailed text layer info
                        char_count = len(text)
                        word_count = len(text.split())
                        preview = text[:100].replace('\n', ' ') if text else "(empty)"
                        logger.info(
                            f"Page {page_num + 1}: HAS TEXT LAYER - "
                            f"{char_count} chars, {word_count} words, "
                            f"preview: '{preview}...'"
                        )
                    else:
                        # Queue for OCR
                        pages_needing_ocr.append(page_num)
                        # Save any partial text for quality comparison
                        original_texts[page_num] = pdf.extract_text(page_num)
                        
                        # DIAGNOSTIC: Log why page needs OCR
                        partial_text = original_texts[page_num]
                        logger.info(
                            f"Page {page_num + 1}: NEEDS OCR - "
                            f"existing text: {len(partial_text)} chars"
                        )

                logger.info(
                    f"Scan complete: {len(text_layer_pages)} text layer pages, "
                    f"{len(pages_needing_ocr)} need OCR"
                )

                # ============================================================
                # PHASE 2: Prepare output PDF
                # ============================================================
                output_path = Path(output_dir) / file_name
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Track original file size
                original_size = Path(file_path).stat().st_size

                # Open output PDF (will modify in-place)
                doc = fitz.open(file_path)

                # Import analyzer for page type detection
                from .pdf_text_layer_analyzer import PDFTextLayerAnalyzer, PageType
                analyzer = PDFTextLayerAnalyzer()

                # ============================================================
                # PHASE 3: Process OCR batches in REVERSE and assemble immediately
                # ============================================================
                # CRITICAL: Process in REVERSE to avoid index shifting!
                # When we rebuild page 0, indices 1,2,3... shift
                # By going backwards (999→0), indices remain stable
                
                if pages_needing_ocr:
                    logger.info(f"Starting streaming OCR+assembly for {len(pages_needing_ocr)} pages")
                    
                    # Reverse the list for processing
                    pages_needing_ocr_reversed = list(reversed(pages_needing_ocr))
                    
                    # Process in reverse batches
                    for batch_start_idx in range(0, len(pages_needing_ocr_reversed), self.batch_size):
                        # Check cancellation
                        if self._is_cancelled():
                            logger.info("Processing cancelled during OCR")
                            result['error'] = "Processing cancelled by user"
                            doc.close()
                            return result

                        batch_end_idx = min(batch_start_idx + self.batch_size, len(pages_needing_ocr_reversed))
                        batch_page_nums = pages_needing_ocr_reversed[batch_start_idx:batch_end_idx]

                        # Report progress
                        pages_completed = batch_start_idx
                        self._report_progress(
                            current=pages_completed,
                            total=len(pages_needing_ocr),
                            message=f"File {file_idx}/{total_files}: {file_name} - OCR batch {pages_completed}/{len(pages_needing_ocr)}"
                        )

                        logger.debug(f"Processing reverse batch: pages {[p+1 for p in batch_page_nums]}")

                        # OCR batch - returns (text, ocr_result, deskewed_img) tuples
                        # Images are temporarily in memory (190 MB for batch_size=19)
                        batch_results = self._process_page_batch(pdf, batch_page_nums, file_name)

                        try:
                            # IMMEDIATELY ASSEMBLE each page while image is hot in memory
                            for page_num, (ocr_text, ocr_result, deskewed_img) in zip(batch_page_nums, batch_results):

                                # Track OCR results
                                if ocr_text:
                                    result['pages_ocr'] += 1
                                else:
                                    logger.warning(f"Page {page_num+1}: OCR returned empty text")
                                    result['pages_ocr_empty'] = result.get('pages_ocr_empty', 0) + 1

                                # Quality comparison - should we use OCR?
                                should_replace = True
                                if page_num in original_texts and original_texts[page_num]:
                                    should_replace, reason = self._should_replace_with_ocr(
                                        original_texts[page_num],
                                        ocr_text,
                                        doc[page_num],
                                        page_num
                                    )

                                    if not should_replace:
                                        # Check if both are low quality
                                        original_score = self._calculate_quality_score(
                                            original_texts[page_num],
                                            doc[page_num]
                                        )
                                        if original_score < self.config.min_quality_threshold:
                                            logger.warning(
                                                f"Page {page_num+1}: Both original and OCR are low quality "
                                                f"(original: {original_score:.2%}), keeping original"
                                            )
                                            result['pages_low_quality'] = result.get('pages_low_quality', 0) + 1

                                        logger.info(f"Page {page_num+1}: Keeping original - {reason}")

                                        # Free image immediately (don't need it)
                                        del deskewed_img
                                        continue

                                # Analyze page type for rebuild strategy
                                page_analysis = analyzer.analyze_page(doc[page_num], page_num)

                                if page_analysis.page_type == PageType.SCANNED_PAGE:
                                    rebuild_strategy = "full"
                                    logger.debug(f"Page {page_num+1}: Using FULL rebuild (scanned)")
                                else:
                                    rebuild_strategy = "overlay"
                                    logger.debug(f"Page {page_num+1}: Using OVERLAY ({page_analysis.page_type.value})")

                                # ASSEMBLE TO PDF RIGHT NOW (while image is in memory)
                                try:
                                    self._clean_rebuild_page_with_ocr(
                                        doc=doc,
                                        page_num=page_num,
                                        ocr_text=ocr_text,
                                        ocr_result=ocr_result,
                                        rebuild_strategy=rebuild_strategy,
                                        deskewed_image=deskewed_img
                                    )
                                    logger.debug(f"Page {page_num+1}: Assembled with {rebuild_strategy} strategy")

                                    # Track strategy usage
                                    strategy_key = f'pages_{rebuild_strategy}_strategy'
                                    result[strategy_key] = result.get(strategy_key, 0) + 1

                                except Exception as e:
                                    logger.error(f"Page {page_num+1}: Failed to assemble - {e}")
                                    # Continue with other pages

                                # FREE IMAGE IMMEDIATELY after assembly
                                del deskewed_img

                        finally:
                            # CRITICAL: Always free batch results even if exception occurs mid-loop
                            # Free entire batch
                            del batch_results

                            # Aggressive memory cleanup after each batch
                            gc.collect()
                            if self.vram_monitor:
                                self.vram_monitor.log_stats()

                            # Memory checkpoint after batch
                            if self.memory_monitor:
                                self.memory_monitor.checkpoint(f"after_batch_{batch_start_idx}")

                            logger.debug(f"Batch complete, memory freed")

                # ============================================================
                # PHASE 4: Add text overlays to text-layer-only pages
                # ============================================================
                # These pages don't need image re-rendering, just text overlay
                # Process in normal order (no rebuild, so no index shifting)
                
                if text_layer_pages:
                    logger.info(f"Adding text overlays to {len(text_layer_pages)} text layer pages")
                    
                    for page_num, text in text_layer_pages.items():
                        try:
                            page = doc[page_num]
                            page_rect = page.rect
                            
                            # DIAGNOSTIC: Log text being added
                            char_count = len(text)
                            logger.info(
                                f"Page {page_num+1}: Adding text overlay ({char_count} chars)..."
                            )
                            
                            # FIXED: Use TextWriter for text-layer-only pages
                            success = self._insert_text_fallback(page, text, page_rect)
                            
                            if success:
                                logger.info(f"Page {page_num+1}: Text overlay added with TextWriter")
                            else:
                                logger.error(f"Page {page_num+1}: TextWriter failed!")
                                raise RuntimeError(f"Failed to add text overlay to page {page_num+1}")
                            
                        except Exception as e:
                            logger.error(f"Page {page_num+1}: CRITICAL - Failed to add text overlay - {e}", exc_info=True)
                            # Continue with other pages

                # ============================================================
                # PHASE 5: Save final PDF
                # ============================================================
                result['pages_processed'] = total_pages

                try:
                    # Save with optimization
                    doc.save(
                        str(output_path),
                        garbage=4,  # Maximum garbage collection
                        deflate=True,  # Compress streams
                        clean=True,  # Clean structure
                        pretty=False  # Compact output
                    )
                    doc.close()

                    # Track file size changes
                    final_size = output_path.stat().st_size
                    size_increase_ratio = final_size / original_size if original_size > 0 else 1.0
                    size_increase_mb = (final_size - original_size) / (1024 * 1024)

                    result['output_path'] = str(output_path)
                    result['status'] = 'success'
                    result['original_size_mb'] = round(original_size / (1024 * 1024), 2)
                    result['final_size_mb'] = round(final_size / (1024 * 1024), 2)
                    result['size_increase_ratio'] = round(size_increase_ratio, 2)
                    result['size_increase_mb'] = round(size_increase_mb, 2)

                    logger.info(
                        f"Saved PDF with OCR: {output_path} - "
                        f"Size: {result['original_size_mb']:.2f}MB → {result['final_size_mb']:.2f}MB "
                        f"({size_increase_ratio:.2f}x, +{size_increase_mb:.2f}MB)"
                    )

                except Exception as save_error:
                    logger.error(f"Failed to save PDF {file_name}: {save_error}", exc_info=True)
                    
                    # Fallback: copy original
                    try:
                        import shutil
                        shutil.copy2(file_path, output_path)
                        result['output_path'] = str(output_path)
                        result['status'] = 'success'
                        result['warning'] = f"Copied original (save failed: {save_error})"
                        logger.warning(f"Fell back to copying original for {file_name}")
                    except Exception as copy_error:
                        logger.error(f"Failed to copy {file_name}: {copy_error}", exc_info=True)
                        result['error'] = f"Failed to save PDF: {save_error}"
                        result['status'] = 'failed'
                        return result

                result['status'] = 'success'

                logger.info(
                    f"Successfully processed {file_name}: "
                    f"{result['pages_text_layer']} text layer, "
                    f"{result['pages_ocr']} OCR"
                )
                
                # Emit "complete" status with timing
                completed_at = time.time()
                started_at = self.file_start_times.get(file_path, completed_at)
                queued_at = self.file_queue_times.get(file_path)
                elapsed_time = completed_at - started_at
                
                if self.file_status_callback:
                    self.file_status_callback(
                        file_path=file_path,
                        file_name=file_name,
                        file_index=file_idx,
                        total_files=total_files,
                        status="complete",
                        queued_at=queued_at,
                        started_at=started_at,
                        completed_at=completed_at,
                        elapsed_time=elapsed_time,
                        total_pages=result['pages_processed']
                    )

        except Exception as e:
            logger.error(f"Failed to process {file_name}: {e}", exc_info=True)
            result['error'] = str(e)
            
            # Emit "failed" status with timing
            completed_at = time.time()
            started_at = self.file_start_times.get(file_path, completed_at)
            queued_at = self.file_queue_times.get(file_path)
            elapsed_time = completed_at - started_at
            
            if self.file_status_callback:
                self.file_status_callback(
                    file_path=file_path,
                    file_name=file_name,
                    file_index=file_idx,
                    total_files=total_files,
                    status="failed",
                    queued_at=queued_at,
                    started_at=started_at,
                    completed_at=completed_at,
                    elapsed_time=elapsed_time,
                    error=str(e)
                )

        finally:
            # CRITICAL: Close fitz document to prevent file handle leak
            if doc is not None:
                try:
                    doc.close()
                    logger.debug(f"Closed fitz document for {file_name}")
                except Exception as close_error:
                    logger.warning(f"Error closing document: {close_error}")

            # Stop memory tracking and detect leaks
            if self.memory_monitor:
                final_snapshot = self.memory_monitor.stop_tracking()
                if final_snapshot:
                    logger.debug(
                        f"Memory tracking complete for {file_name}: "
                        f"Final RSS={final_snapshot.rss_mb:.2f}MB"
                    )

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

        # Emit "queued" status for all files
        queued_at = time.time()
        if self.file_status_callback:
            for file_idx, file_path in enumerate(files, start=1):
                file_name = Path(file_path).name
                self.file_queue_times[file_path] = queued_at
                self.file_status_callback(
                    file_path=file_path,
                    file_name=file_name,
                    file_index=file_idx,
                    total_files=len(files),
                    status="queued",
                    queued_at=queued_at
                )
        
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
                # Pass through all fields from _process_single_file
                success_result = dict(result)  # Copy all fields
                success_result['file'] = file_path  # Add file path
                successful.append(success_result)
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
        
        # Log engine pool performance summary
        if self.ocr_service and self.ocr_service.use_pooling:
            from .ocr.engine_pool import get_engine_pool
            pool = get_engine_pool()
            pool.log_performance_summary()

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
            # End session and keep engines in pool for future use
            # Set cleanup=False to keep engines in pool (for next batch)
            # Set cleanup=True to force immediate cleanup
            self.ocr_service.end_session(cleanup=False)
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
