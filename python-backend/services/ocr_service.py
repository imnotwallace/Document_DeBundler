"""
OCR Service
Handles optical character recognition for scanned PDFs using the OCR abstraction layer
"""

import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image

from .ocr import OCRManager, OCRConfig, create_ocr_manager, get_default_config, get_model_directory
from .ocr.engine_pool import OCREnginePool, get_engine_pool
from .ocr.post_processor import process_ocr_text

logger = logging.getLogger(__name__)


class OCRService:
    """
    Handles OCR processing for scanned documents.
    Uses the OCR abstraction layer with PaddleOCR/Tesseract engines.
    """

    def __init__(
        self,
        gpu: bool = True,
        engine: Optional[str] = None,
        fallback_enabled: bool = True,
        use_pooling: bool = True,
        language: str = "en",
        model_version: str = "mobile"
    ):
        """
        Initialize OCR service.

        Args:
            gpu: Whether to use GPU if available (auto-detects)
            engine: Specific engine to use ("paddleocr", "tesseract", or "auto")
            fallback_enabled: Enable fallback to alternative engine if primary fails
            use_pooling: Use engine pooling for reuse (recommended for batch processing)
            language: OCR language code (e.g., "en", "ch", "french")
            model_version: Model version ("server" or "mobile")
        """
        self.gpu = gpu
        self.engine_name = engine
        self.fallback_enabled = fallback_enabled
        self.use_pooling = use_pooling
        self.manager: Optional[OCRManager] = None
        self._in_session = False  # Track if we're in a batch session
        self._engine_pool = get_engine_pool() if use_pooling else None

        # Initialize OCR manager
        try:
            config = get_default_config(
                engine=engine,
                prefer_gpu=gpu,
                model_dir=get_model_directory()
            )
            
            # Override language and model version
            config.languages = [language]
            config.model_version = model_version
            
            # Use engine pooling if enabled
            if self.use_pooling and self._engine_pool:
                logger.info(f"Initializing OCR service with engine pooling")
                
                # Don't initialize immediately - use lazy initialization
                # Engine will be created from pool on first use
                self.manager = OCRManager(config=config, fallback_enabled=fallback_enabled)
                # Don't call initialize() - will use pool
                
                logger.info(f"OCR service initialized with pooling (engine: {engine or 'auto'}, lang: {language}, version: {model_version})")
            else:
                # Traditional initialization without pooling
                self.manager = OCRManager(config=config, fallback_enabled=fallback_enabled)
                self.manager.initialize()
                logger.info(f"OCR service initialized with {self.manager.get_engine_name()} (lang: {language}, version: {model_version})")
                
        except Exception as e:
            logger.error(f"Failed to initialize OCR service: {e}", exc_info=True)
            self.manager = None

    def start_session(self) -> None:
        """
        Start a batch processing session.
        
        During a session, the OCR engine is kept alive for reuse,
        avoiding expensive re-initialization between operations.
        
        Example:
            ocr = OCRService(gpu=True)
            ocr.start_session()
            for page in pages:
                text = ocr.extract_text_from_array(page)
            ocr.end_session()
        """
        self._in_session = True
        logger.info("Started OCR batch session (engines will be reused)")
        
        # Warm up engine if using pooling
        if self.use_pooling and self._engine_pool:
            engine_type = self.engine_name or "paddleocr"
            config = get_default_config(
                engine=engine_type,
                prefer_gpu=self.gpu,
                model_dir=get_model_directory()
            )
            
            try:
                engine, stats = self._engine_pool.get_engine(engine_type, config)
                
                # Warm up if this is a cold engine
                if not stats.is_warm:
                    self._engine_pool.warmup_engine(engine_type)
                    
                logger.info(
                    f"Using {engine_type} engine from pool "
                    f"(warm: {stats.is_warm}, inferences: {stats.total_inferences})"
                )
            except Exception as e:
                logger.warning(f"Failed to get engine from pool: {e}, falling back to standard initialization")
    
    def end_session(self, cleanup: bool = False) -> None:
        """
        End a batch processing session.
        
        Args:
            cleanup: If True, cleanup engines immediately. If False (default),
                    keep engines in pool for future sessions.
        
        Example:
            ocr = OCRService(gpu=True)
            ocr.start_session()
            # ... process many pages ...
            ocr.end_session(cleanup=False)  # Keep engines for next batch
        """
        self._in_session = False
        
        if cleanup:
            logger.info("Ending OCR batch session with cleanup")
            self.cleanup()
        else:
            logger.info("Ending OCR batch session (keeping engines in pool for reuse)")
            
        # Log performance summary if using pooling
        if self.use_pooling and self._engine_pool:
            self._engine_pool.log_performance_summary()
    
    def switch_engine(self, engine_name: str) -> None:
        """
        Switch to a different OCR engine without losing session state.
        
        Uses engine pooling to avoid re-initialization overhead.
        
        Args:
            engine_name: Engine to switch to ("paddleocr" or "tesseract")
        
        Example:
            ocr = OCRService(gpu=True, engine="paddleocr")
            ocr.start_session()
            
            # Process with PaddleOCR
            text1 = ocr.extract_text_from_array(page1)
            
            # Switch to Tesseract for fallback
            ocr.switch_engine("tesseract")
            text2 = ocr.extract_text_from_array(page2)
            
            ocr.end_session()
        """
        if not self.use_pooling:
            logger.warning("Engine switching requires pooling to be enabled")
            # Fall back to re-initialization
            self.cleanup()
            self.engine_name = engine_name
            config = get_default_config(
                engine=engine_name,
                prefer_gpu=self.gpu,
                model_dir=get_model_directory()
            )
            self.manager = OCRManager(config=config, fallback_enabled=self.fallback_enabled)
            self.manager.initialize()
            return
        
        logger.info(f"Switching to {engine_name} engine")
        self.engine_name = engine_name
        
        # Get engine from pool
        config = get_default_config(
            engine=engine_name,
            prefer_gpu=self.gpu,
            model_dir=get_model_directory()
        )
        
        try:
            engine, stats = self._engine_pool.get_engine(engine_name, config)
            
            # Update manager to use pooled engine
            if self.manager is None:
                self.manager = OCRManager(config=config, fallback_enabled=False)
            
            self.manager.engine = engine
            self.manager._engine_name = engine_name
            
            logger.info(f"Switched to {engine_name} (warm: {stats.is_warm}, inferences: {stats.total_inferences})")
            
        except Exception as e:
            logger.error(f"Failed to switch engine: {e}")
            raise

    def _ensure_engine_available(self) -> bool:
        """
        Ensure OCR engine is available, using pool if enabled.
        
        Returns:
            True if engine is available, False otherwise
        """
        if self.manager is None:
            return False
            
        # If engine already initialized, we're good
        if self.manager.engine is not None:
            return True
        
        # If pooling enabled, get engine from pool
        if self.use_pooling and self._engine_pool:
            engine_type = self.engine_name or "paddleocr"
            config = get_default_config(
                engine=engine_type,
                prefer_gpu=self.gpu,
                model_dir=get_model_directory()
            )
            
            try:
                engine, stats = self._engine_pool.get_engine(engine_type, config)
                self.manager.engine = engine
                self.manager._engine_name = engine_type
                
                # Mark engine as used
                self._engine_pool.mark_used(engine_type)
                
                logger.debug(
                    f"Got {engine_type} from pool "
                    f"(warm: {stats.is_warm}, inferences: {stats.total_inferences})"
                )
                return True
                
            except Exception as e:
                logger.error(f"Failed to get engine from pool: {e}")
                return False
        
        # No pooling - engine should have been initialized in __init__
        return False

    def extract_text_from_image(self, image_path: Path) -> str:
        """
        Extract text from a single image file.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text as string
        """
        if not self.is_available():
            logger.warning("OCR not initialized")
            return ""
        
        # Ensure engine is available (get from pool if needed)
        if not self._ensure_engine_available():
            logger.error("Failed to get OCR engine")
            return ""

        try:
            # Load image
            image = Image.open(image_path)
            image_array = np.array(image)

            # Process with OCR
            result = self.manager.process_image(image_array)

            if result.error:
                logger.error(f"OCR error: {result.error}")
                return ""

            return result.text

        except Exception as e:
            logger.error(f"Failed to extract text from image: {e}", exc_info=True)
            return ""

    def extract_text_from_array(self, image_array: np.ndarray) -> str:
        """
        Extract text from image as numpy array.

        Args:
            image_array: Image as numpy array

        Returns:
            Extracted text as string
        """
        if not self.is_available():
            logger.warning("OCR not initialized")
            return ""
        
        # Ensure engine is available (get from pool if needed)
        if not self._ensure_engine_available():
            logger.error("Failed to get OCR engine")
            return ""

        try:
            result = self.manager.process_image(image_array)

            if result.error:
                logger.error(f"OCR error: {result.error}")
                return ""

            logger.debug(f"OCR confidence: {result.confidence:.2f}, time: {result.processing_time:.2f}s")

            # Apply post-processing to improve accuracy
            processed_text = process_ocr_text(result.text)
            return processed_text

        except Exception as e:
            logger.error(f"Failed to extract text: {e}", exc_info=True)
            return ""

    def process_pdf_page(self, pdf_path: Path, page_num: int) -> str:
        """
        Extract text from a PDF page using OCR.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)

        Returns:
            Extracted text as string
        """
        if not self.is_available():
            logger.warning("OCR not initialized")
            return ""
        
        # Ensure engine is available (get from pool if needed)
        if not self._ensure_engine_available():
            logger.error("Failed to get OCR engine")
            return ""

        try:
            import fitz  # PyMuPDF

            # Open PDF and get page
            doc = fitz.open(pdf_path)
            page = doc[page_num]

            # Render page to image (higher DPI = better quality)
            pix = page.get_pixmap(dpi=300)

            # Convert to numpy array
            image_array = np.frombuffer(pix.samples, dtype=np.uint8)
            image_array = image_array.reshape(pix.height, pix.width, pix.n)

            # If RGBA, convert to RGB
            if pix.n == 4:
                image_array = image_array[:, :, :3]

            doc.close()

            # Process with OCR
            return self.extract_text_from_array(image_array)

        except Exception as e:
            logger.error(f"Failed to process PDF page: {e}", exc_info=True)
            return ""

    def process_batch(self, images: List[np.ndarray]) -> List[str]:
        """
        Process multiple images in batch for better performance.

        Args:
            images: List of images as numpy arrays

        Returns:
            List of extracted text strings
        """
        if not self.is_available():
            logger.warning("OCR not initialized")
            return [""] * len(images)
        
        # Ensure engine is available (get from pool if needed)
        if not self._ensure_engine_available():
            logger.error("Failed to get OCR engine")
            return [""] * len(images)

        try:
            results = self.manager.process_batch(images)
            return [r.text for r in results]

        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            return [""] * len(images)

    def process_batch_with_boxes(self, images: List[np.ndarray]):
        """
        Process multiple images in batch, returning full OCR results with bounding boxes.

        Args:
            images: List of images as numpy arrays

        Returns:
            List of OCRResult objects with text and bounding boxes
        """
        if not self.is_available():
            logger.warning("OCR not initialized")
            # Return empty OCRResult objects
            from .ocr.base import OCRResult
            return [OCRResult(text="", confidence=0.0) for _ in images]
        
        # Ensure engine is available (get from pool if needed)
        if not self._ensure_engine_available():
            logger.error("Failed to get OCR engine")
            from .ocr.base import OCRResult
            return [OCRResult(text="", confidence=0.0, error="Failed to get engine") for _ in images]

        try:
            # Return full OCRResult objects
            results = self.manager.process_batch(images)
            return results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            from .ocr.base import OCRResult
            return [OCRResult(text="", confidence=0.0, error=str(e)) for _ in images]

    def is_available(self) -> bool:
        """Check if OCR is available and initialized"""
        # If using pooling, manager existence is enough (engine is lazy-initialized)
        if self.use_pooling:
            return self.manager is not None
        # Otherwise, check if engine is initialized
        return self.manager is not None and self.manager.engine is not None

    def get_engine_info(self) -> dict:
        """
        Get information about the current OCR engine.

        Returns:
            Dictionary with engine information
        """
        if not self.is_available():
            return {"available": False}

        return {
            "available": True,
            "engine": self.manager.get_engine_name(),
            "gpu_enabled": self.manager.supports_gpu(),
            "memory_usage_mb": self.manager.get_memory_usage() / (1024 * 1024),
        }

    def cleanup(self) -> None:
        """
        Release OCR resources.
        
        Note: If using engine pooling and in a session, engines are kept
        in the pool for reuse. Call end_session(cleanup=True) to force cleanup.
        """
        # If in session and using pooling, don't cleanup (keep for reuse)
        if self._in_session and self.use_pooling:
            logger.debug("Skipping cleanup: in batch session with pooling enabled")
            return
        
        if self.manager is not None:
            # If using pooling, engines are managed by the pool
            if not self.use_pooling:
                self.manager.cleanup()
            self.manager = None
            logger.info("OCR service cleaned up")

    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()
