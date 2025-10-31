"""
OCR Manager
Factory and management for OCR engines with fallback support
"""

import logging
from typing import Optional, List
from pathlib import Path

from .base import OCREngine, OCRConfig, OCRResult
from .engines.paddleocr_engine import PaddleOCREngine
from .engines.tesseract_engine import TesseractEngine
from .config import get_default_config, get_model_directory

logger = logging.getLogger(__name__)


class OCRManager:
    """
    Manages OCR engine lifecycle with automatic fallback.

    Features:
    - Automatic engine selection based on configuration
    - Fallback to alternative engine if primary fails
    - Resource management and cleanup
    - Progress reporting
    """

    def __init__(
        self,
        config: Optional[OCRConfig] = None,
        fallback_enabled: bool = True
    ):
        """
        Initialize OCR manager.

        Args:
            config: OCR configuration, or None for auto-detection
            fallback_enabled: Enable automatic fallback to alternative engine
        """
        if config is None:
            config = get_default_config(model_dir=get_model_directory())

        self.config = config
        self.fallback_enabled = fallback_enabled
        self.engine: Optional[OCREngine] = None
        self._engine_name: Optional[str] = None

    def initialize(self) -> None:
        """Initialize the OCR engine"""
        if self.engine is not None:
            logger.warning("OCR engine already initialized")
            return

        # Determine which engine to use
        engine_name = self.config.engine

        if engine_name == "auto":
            # Auto-select: Try PaddleOCR first, fallback to Tesseract
            engine_name = "paddleocr"

        # Try to create and initialize the engine
        try:
            self.engine = self._create_engine(engine_name)
            self.engine.initialize()
            self._engine_name = engine_name
            logger.info(f"OCR engine '{engine_name}' initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize {engine_name}: {e}")

            if self.fallback_enabled and engine_name != "tesseract":
                logger.info("Attempting fallback to Tesseract...")
                try:
                    self.engine = self._create_engine("tesseract")
                    self.engine.initialize()
                    self._engine_name = "tesseract"
                    logger.info("Fallback to Tesseract successful")
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise RuntimeError(
                        f"Failed to initialize any OCR engine. "
                        f"Primary error: {e}, Fallback error: {fallback_error}"
                    )
            else:
                raise

    def _create_engine(self, engine_name: str) -> OCREngine:
        """
        Factory method to create OCR engine.

        Args:
            engine_name: Name of engine to create

        Returns:
            OCREngine instance

        Raises:
            ValueError: If engine name is unknown
        """
        if engine_name == "paddleocr":
            return PaddleOCREngine(self.config)
        elif engine_name == "tesseract":
            return TesseractEngine(self.config)
        else:
            raise ValueError(f"Unknown OCR engine: {engine_name}")

    def process_image(self, image) -> OCRResult:
        """
        Process a single image.

        Args:
            image: Image as numpy array or file path

        Returns:
            OCRResult

        Raises:
            RuntimeError: If engine not initialized
        """
        if self.engine is None:
            raise RuntimeError("OCR engine not initialized. Call initialize() first.")

        return self.engine.process_image(image)

    def process_batch(self, images: List) -> List[OCRResult]:
        """
        Process multiple images in batch.

        Args:
            images: List of images as numpy arrays or file paths

        Returns:
            List of OCRResult objects

        Raises:
            RuntimeError: If engine not initialized
        """
        if self.engine is None:
            raise RuntimeError("OCR engine not initialized. Call initialize() first.")

        return self.engine.process_batch(images)

    def cleanup(self) -> None:
        """Clean up and release OCR engine resources"""
        if self.engine is not None:
            self.engine.cleanup()
            self.engine = None
            logger.info(f"OCR engine '{self._engine_name}' cleaned up")

    def get_engine_name(self) -> Optional[str]:
        """Get the name of the currently active engine"""
        return self._engine_name

    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        if self.engine is not None:
            return self.engine.get_memory_usage()
        return 0

    def supports_gpu(self) -> bool:
        """Check if current engine supports GPU"""
        if self.engine is not None:
            return self.engine.supports_gpu()
        return False

    def __enter__(self):
        """Context manager support"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.cleanup()
        return False


def create_ocr_manager(
    engine: Optional[str] = None,
    use_gpu: bool = True,
    fallback_enabled: bool = True
) -> OCRManager:
    """
    Convenience function to create and initialize OCR manager.

    Args:
        engine: Specific engine to use ("paddleocr", "tesseract", or "auto")
        use_gpu: Whether to use GPU if available
        fallback_enabled: Enable fallback to alternative engine

    Returns:
        Initialized OCRManager

    Example:
        >>> with create_ocr_manager(use_gpu=True) as ocr:
        ...     result = ocr.process_image(image)
        ...     print(result.text)
    """
    config = get_default_config(
        engine=engine,
        prefer_gpu=use_gpu,
        model_dir=get_model_directory()
    )

    manager = OCRManager(config=config, fallback_enabled=fallback_enabled)
    manager.initialize()

    return manager
