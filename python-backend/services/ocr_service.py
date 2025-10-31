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
        fallback_enabled: bool = True
    ):
        """
        Initialize OCR service.

        Args:
            gpu: Whether to use GPU if available (auto-detects)
            engine: Specific engine to use ("paddleocr", "tesseract", or "auto")
            fallback_enabled: Enable fallback to alternative engine if primary fails
        """
        self.gpu = gpu
        self.engine_name = engine
        self.fallback_enabled = fallback_enabled
        self.manager: Optional[OCRManager] = None

        # Initialize OCR manager
        try:
            config = get_default_config(
                engine=engine,
                prefer_gpu=gpu,
                model_dir=get_model_directory()
            )
            self.manager = OCRManager(config=config, fallback_enabled=fallback_enabled)
            self.manager.initialize()
            logger.info(f"OCR service initialized with {self.manager.get_engine_name()}")
        except Exception as e:
            logger.error(f"Failed to initialize OCR service: {e}", exc_info=True)
            self.manager = None

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

        try:
            result = self.manager.process_image(image_array)

            if result.error:
                logger.error(f"OCR error: {result.error}")
                return ""

            logger.debug(f"OCR confidence: {result.confidence:.2f}, time: {result.processing_time:.2f}s")
            return result.text

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

        try:
            results = self.manager.process_batch(images)
            return [r.text for r in results]

        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            return [""] * len(images)

    def is_available(self) -> bool:
        """Check if OCR is available and initialized"""
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
        """Release OCR resources"""
        if self.manager is not None:
            self.manager.cleanup()
            self.manager = None
            logger.info("OCR service cleaned up")

    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()
