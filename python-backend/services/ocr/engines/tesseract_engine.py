"""
Tesseract OCR Engine Implementation
Lightweight CPU-only OCR engine
"""

import logging
import time
import gc
from typing import List
from pathlib import Path
import numpy as np
from PIL import Image

from ..base import OCREngine, OCRResult, OCRConfig
from ...resource_path import setup_tesseract_environment, verify_tesseract_setup

logger = logging.getLogger(__name__)


class TesseractEngine(OCREngine):
    """Tesseract OCR implementation (CPU-only, lightweight)"""

    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self.tesseract_available = False

    def initialize(self) -> None:
        """Initialize Tesseract OCR engine with bundled or system executable"""
        if self._initialized:
            logger.warning("Tesseract already initialized")
            return

        try:
            import pytesseract

            # Configure Tesseract paths (bundled or system)
            tesseract_config = setup_tesseract_environment()

            # Set pytesseract to use bundled executable if available
            if tesseract_config['tesseract_cmd']:
                pytesseract.pytesseract.tesseract_cmd = tesseract_config['tesseract_cmd']
                logger.info(f"Using bundled Tesseract: {tesseract_config['tesseract_cmd']}")

                # Set tessdata prefix for language files
                if tesseract_config['tessdata_prefix']:
                    # This is already set in environment, but also set on pytesseract for safety
                    logger.info(f"Tessdata prefix: {tesseract_config['tessdata_prefix']}")
            else:
                logger.info("Using system Tesseract executable")

            # Verify setup
            success, message = verify_tesseract_setup()
            if not success:
                logger.error(f"Tesseract setup verification failed: {message}")
                raise RuntimeError(f"Tesseract configuration error: {message}")

            logger.info(f"Tesseract setup verified: {message}")

            # Test if Tesseract is actually working
            try:
                version = pytesseract.get_tesseract_version()
                logger.info(f"Tesseract version: {version}")
                self.tesseract_available = True
            except Exception as e:
                logger.error(f"Tesseract executable test failed: {e}")
                raise RuntimeError(
                    f"Tesseract executable not working: {e}\n"
                    "If using bundled Tesseract, ensure all files are in python-backend/bin/tesseract/\n"
                    "See python-backend/bin/README.md for setup instructions."
                )

            self._initialized = True
            logger.info(f"Tesseract engine initialized successfully (mode: {tesseract_config['mode']})")

        except ImportError as e:
            logger.error(f"pytesseract not installed: {e}")
            raise RuntimeError(
                "pytesseract is not installed. "
                "Install with: pip install pytesseract"
            )

    def _is_valid_text(self, text: str) -> bool:
        """
        Validate that text contains mostly valid characters and isn't garbage.

        Args:
            text: Text string to validate

        Returns:
            True if text appears valid, False if likely garbage
        """
        if not text or len(text) == 0:
            return False

        # Check alphanumeric ratio (at least 50% should be letters/numbers)
        alnum_count = sum(c.isalnum() for c in text)
        if alnum_count / len(text) < 0.5:
            return False

        # Reject common OCR garbage patterns
        garbage_patterns = ['|', '||', '|||', '#', '##', '###', '...', '---', ']', '[', '}{', '()']
        if text.strip() in garbage_patterns:
            return False

        # Reject text that's only punctuation
        punct_count = sum(c in '.,!?;:' for c in text)
        if punct_count == len(text):
            return False

        return True

    def process_image(self, image: np.ndarray) -> OCRResult:
        """
        Process a single image with Tesseract.

        Args:
            image: Image as numpy array (RGB)

        Returns:
            OCRResult with extracted text
        """
        if not self._initialized:
            raise RuntimeError("OCR engine not initialized")

        import pytesseract

        start_time = time.time()

        try:
            # Convert numpy array to PIL Image
            if len(image.shape) == 2:
                # Grayscale
                pil_image = Image.fromarray(image, mode='L')
            else:
                # RGB or BGR
                if image.shape[2] == 4:
                    # RGBA
                    pil_image = Image.fromarray(image, mode='RGBA')
                else:
                    # RGB
                    pil_image = Image.fromarray(image, mode='RGB')

            # Get language code
            lang = '+'.join(self.config.languages) if self.config.languages else 'eng'

            # Configure Tesseract
            custom_config = f'--oem 3 --psm 3'  # OEM 3 = Default, PSM 3 = Automatic

            # Get text with confidence data
            data = pytesseract.image_to_data(
                pil_image,
                lang=lang,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )

            # Extract text and calculate confidence
            text_parts = []
            confidences = []

            # Minimum confidence threshold (60%) to filter low-quality detections
            MIN_CONFIDENCE = 60

            for i, conf in enumerate(data['conf']):
                if int(conf) > MIN_CONFIDENCE:  # Require 60% confidence minimum
                    text = data['text'][i].strip()
                    if text and self._is_valid_text(text):  # Validate text quality
                        text_parts.append(text)
                        confidences.append(int(conf) / 100.0)  # Convert to 0-1 range

            # Combine text
            full_text = ' '.join(text_parts)

            # Calculate average confidence
            avg_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

            processing_time = time.time() - start_time

            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                raw_result=data,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Tesseract OCR processing failed: {e}", exc_info=True)
            processing_time = time.time() - start_time
            return OCRResult(
                text="",
                confidence=0.0,
                error=str(e),
                processing_time=processing_time
            )

    def process_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """
        Process multiple images.
        Tesseract doesn't have native batch processing,
        so we process sequentially.

        Args:
            images: List of images as numpy arrays

        Returns:
            List of OCRResult objects
        """
        if not self._initialized:
            raise RuntimeError("OCR engine not initialized")

        results = []

        for i, image in enumerate(images):
            result = self.process_image(image)
            results.append(result)

            # Periodic garbage collection
            if (i + 1) % 10 == 0:
                gc.collect()

        return results

    def cleanup(self) -> None:
        """Release Tesseract resources"""
        gc.collect()
        self._initialized = False
        logger.info("Tesseract engine cleaned up")

    def get_memory_usage(self) -> int:
        """Get estimated memory usage in bytes"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    def supports_gpu(self) -> bool:
        """Tesseract does not support GPU"""
        return False

    def is_available(self) -> bool:
        """Check if Tesseract engine is available and ready to process"""
        return self._initialized and self.tesseract_available
