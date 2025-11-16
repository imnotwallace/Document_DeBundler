"""
Image Preprocessing for Photo-Based PDFs

Enhances image quality before OCR to improve text detection:
- Contrast enhancement (CLAHE)
- Binarization (Otsu's thresholding)
- Noise reduction
- Sharpening

Target: Improve OCR accuracy on low-quality photo scans at 300 DPI
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocesses images to improve OCR accuracy on photo-based scans"""

    def __init__(
        self,
        enable_clahe: bool = True,
        enable_binarization: bool = True,
        enable_denoising: bool = True,
        enable_sharpening: bool = True
    ):
        """
        Initialize preprocessor with enhancement options

        Args:
            enable_clahe: Enable contrast enhancement (recommended for photos)
            enable_binarization: Enable adaptive thresholding (helps with shadows)
            enable_denoising: Enable noise reduction (removes artifacts)
            enable_sharpening: Enable sharpening (improves edge clarity)
        """
        self.enable_clahe = enable_clahe
        self.enable_binarization = enable_binarization
        self.enable_denoising = enable_denoising
        self.enable_sharpening = enable_sharpening

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to enhance image for OCR

        Args:
            image: Input image (RGB or BGR)

        Returns:
            Preprocessed image ready for OCR
        """
        logger.info(f"Preprocessing image: {image.shape}")

        # Convert to grayscale if color
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            logger.debug("Converted to grayscale")
        else:
            gray = image.copy()

        # Step 1: Denoise first to remove noise before enhancement
        if self.enable_denoising:
            gray = self._denoise(gray)

        # Step 2: Enhance contrast (critical for photos with poor lighting)
        if self.enable_clahe:
            gray = self._enhance_contrast(gray)

        # Step 3: Binarize to pure black/white (helps with detection)
        if self.enable_binarization:
            binary = self._binarize(gray)
        else:
            binary = gray

        # Step 4: Sharpen to improve edge clarity
        if self.enable_sharpening:
            binary = self._sharpen(binary)

        # Convert back to RGB for PaddleOCR (expects 3-channel)
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        logger.info(f"Preprocessing complete: {result.shape}")
        return result

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)

        This is critical for photos with uneven lighting, shadows, or low contrast.
        Much better than simple histogram equalization for photos.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        logger.debug("Applied CLAHE contrast enhancement")
        return enhanced

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding to convert to pure black/white

        Adaptive thresholding handles varying lighting across the image
        (shadows, uneven illumination) better than global thresholding.
        """
        # Use Gaussian adaptive thresholding (works well for documents)
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,  # Neighborhood size
            C=2  # Constant subtracted from mean
        )
        logger.debug("Applied adaptive thresholding")
        return binary

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise while preserving edges

        Uses Non-Local Means Denoising - effective for photo noise
        """
        denoised = cv2.fastNlMeansDenoising(image, h=10)
        logger.debug("Applied denoising")
        return denoised

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen image to improve edge clarity

        Uses unsharp masking - enhances edges without amplifying noise
        """
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), 3)

        # Unsharp mask: original + (original - blurred) * amount
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

        logger.debug("Applied sharpening")
        return sharpened


def preprocess_for_ocr(
    image: np.ndarray,
    quality: str = "high"
) -> np.ndarray:
    """
    Convenience function to preprocess image for OCR

    Args:
        image: Input image (RGB or BGR)
        quality: Preprocessing quality level
            - "high": Full preprocessing (recommended for photos)
            - "medium": Contrast + binarization only
            - "low": Contrast enhancement only
            - "none": No preprocessing

    Returns:
        Preprocessed image
    """
    if quality == "none":
        return image

    if quality == "low":
        processor = ImagePreprocessor(
            enable_clahe=True,
            enable_binarization=False,
            enable_denoising=False,
            enable_sharpening=False
        )
    elif quality == "medium":
        processor = ImagePreprocessor(
            enable_clahe=True,
            enable_binarization=True,
            enable_denoising=False,
            enable_sharpening=False
        )
    else:  # high
        processor = ImagePreprocessor(
            enable_clahe=True,
            enable_binarization=True,
            enable_denoising=True,
            enable_sharpening=True
        )

    return processor.preprocess(image)
