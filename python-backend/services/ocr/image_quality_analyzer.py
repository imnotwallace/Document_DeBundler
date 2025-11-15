"""
Image Quality Analysis for Intelligent OCR Preprocessing

Analyzes image quality metrics to determine optimal preprocessing strategies:
- Blur detection (Laplacian variance)
- Contrast measurement (histogram analysis)
- Noise level estimation (local variance)
- Edge strength (Sobel gradient magnitude)
- Document type classification

This enables intelligent, adaptive preprocessing that only applies techniques
when they will improve OCR quality.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Detected document type based on image characteristics"""
    SCAN_HIGH_QUALITY = "scan_high_quality"     # Professional scanner, minimal noise
    SCAN_LOW_QUALITY = "scan_low_quality"       # Old scanner or low DPI
    PHOTO_GOOD_LIGHTING = "photo_good_lighting" # Camera photo, good conditions
    PHOTO_POOR_LIGHTING = "photo_poor_lighting" # Camera photo, low light/glare
    DIGITAL_BORN = "digital_born"               # Computer-generated (e.g., PDF export)
    RECEIPT = "receipt"                         # Thermal receipt, faded text
    UNKNOWN = "unknown"                         # Cannot classify confidently


@dataclass
class ImageQualityMetrics:
    """
    Comprehensive quality metrics for an image.

    Attributes:
        blur_score: Laplacian variance (higher = sharper, <100 = blurry)
        contrast_score: Standard deviation of pixel intensities (higher = more contrast)
        contrast_ratio: Contrast as ratio of std/mean (normalized measure)
        noise_level: Local variance estimate (higher = noisier)
        edge_strength: Mean Sobel gradient magnitude (higher = stronger edges)
        brightness: Mean pixel intensity (0-255)
        dynamic_range: Difference between max and min intensities
        document_type: Classified document type
        is_blurry: Boolean flag if blur_score below threshold
        is_low_contrast: Boolean flag if contrast below threshold
        is_noisy: Boolean flag if noise above threshold
        needs_preprocessing: Overall assessment if preprocessing recommended
    """
    blur_score: float
    contrast_score: float
    contrast_ratio: float
    noise_level: float
    edge_strength: float
    brightness: float
    dynamic_range: float
    document_type: DocumentType

    # Quality flags
    is_blurry: bool
    is_low_contrast: bool
    is_noisy: bool
    needs_preprocessing: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/serialization"""
        return {
            "blur_score": float(self.blur_score),
            "contrast_score": float(self.contrast_score),
            "contrast_ratio": float(self.contrast_ratio),
            "noise_level": float(self.noise_level),
            "edge_strength": float(self.edge_strength),
            "brightness": float(self.brightness),
            "dynamic_range": float(self.dynamic_range),
            "document_type": self.document_type.value,
            "is_blurry": self.is_blurry,
            "is_low_contrast": self.is_low_contrast,
            "is_noisy": self.is_noisy,
            "needs_preprocessing": self.needs_preprocessing
        }


class ImageQualityAnalyzer:
    """
    Analyzes image quality to determine optimal preprocessing strategies.

    Uses multiple metrics to assess image characteristics:
    - Blur detection via Laplacian variance
    - Contrast via histogram statistics
    - Noise via local variance estimation
    - Edge strength via Sobel gradients
    - Document type classification

    Example:
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(image)

        if metrics.is_blurry:
            # Apply deblurring
        if metrics.is_low_contrast:
            # Apply contrast enhancement
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,  # Lower = more sensitive
        contrast_threshold: float = 40.0,  # Lower = more sensitive
        noise_threshold: float = 50.0  # Higher = more sensitive
    ):
        """
        Initialize quality analyzer with thresholds.

        Args:
            blur_threshold: Laplacian variance below which image is blurry (default: 100)
            contrast_threshold: Std dev below which image is low contrast (default: 40)
            noise_threshold: Local variance above which image is noisy (default: 50)
        """
        self.blur_threshold = blur_threshold
        self.contrast_threshold = contrast_threshold
        self.noise_threshold = noise_threshold

        logger.info(
            f"ImageQualityAnalyzer initialized with thresholds: "
            f"blur={blur_threshold}, contrast={contrast_threshold}, "
            f"noise={noise_threshold}"
        )

    def analyze(self, image: np.ndarray) -> ImageQualityMetrics:
        """
        Perform comprehensive quality analysis on image.

        Args:
            image: Input image (grayscale or RGB)

        Returns:
            ImageQualityMetrics with all quality measurements
        """
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV (cv2) not installed, cannot analyze quality")
            # Return default metrics indicating no analysis possible
            return self._default_metrics()

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 1:
                # Grayscale with channel dimension - squeeze it
                gray = image.squeeze()
            elif image.shape[2] in (3, 4):
                # RGB or RGBA - convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                logger.warning(f"Unexpected number of channels: {image.shape[2]}, returning default metrics")
                return self._default_metrics()
        elif len(image.shape) == 2:
            # Already grayscale
            gray = image.copy()
        else:
            logger.warning(f"Unexpected image shape: {image.shape}, returning default metrics")
            return self._default_metrics()

        logger.debug(f"Analyzing image quality: shape={gray.shape}, dtype={gray.dtype}")

        # Compute all metrics
        blur_score = self._compute_blur_score(gray)
        contrast_score = self._compute_contrast_score(gray)
        contrast_ratio = self._compute_contrast_ratio(gray)
        noise_level = self._compute_noise_level(gray)
        edge_strength = self._compute_edge_strength(gray)
        brightness = self._compute_brightness(gray)
        dynamic_range = self._compute_dynamic_range(gray)

        # Classify document type based on metrics
        document_type = self._classify_document_type(
            blur_score, contrast_score, noise_level,
            edge_strength, brightness, dynamic_range
        )

        # Determine quality flags
        is_blurry = blur_score < self.blur_threshold
        is_low_contrast = contrast_score < self.contrast_threshold
        is_noisy = noise_level > self.noise_threshold

        # Overall preprocessing recommendation
        needs_preprocessing = is_blurry or is_low_contrast or is_noisy

        metrics = ImageQualityMetrics(
            blur_score=blur_score,
            contrast_score=contrast_score,
            contrast_ratio=contrast_ratio,
            noise_level=noise_level,
            edge_strength=edge_strength,
            brightness=brightness,
            dynamic_range=dynamic_range,
            document_type=document_type,
            is_blurry=is_blurry,
            is_low_contrast=is_low_contrast,
            is_noisy=is_noisy,
            needs_preprocessing=needs_preprocessing
        )

        logger.debug(f"Quality analysis complete: {metrics.to_dict()}")
        return metrics

    def _compute_blur_score(self, gray: np.ndarray) -> float:
        """
        Compute blur score using Laplacian variance.

        Higher values indicate sharper images. Typically:
        - > 500: Very sharp
        - 100-500: Acceptable
        - < 100: Blurry

        Args:
            gray: Grayscale image

        Returns:
            Blur score (Laplacian variance)
        """
        try:
            import cv2

            # Compute Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)

            # Variance of Laplacian as blur measure
            variance = laplacian.var()

            logger.debug(f"Blur score (Laplacian variance): {variance:.2f}")
            return float(variance)

        except Exception as e:
            logger.warning(f"Blur score computation failed: {e}")
            return 0.0

    def _compute_contrast_score(self, gray: np.ndarray) -> float:
        """
        Compute contrast score using histogram standard deviation.

        Higher values indicate better contrast. Typically:
        - > 60: Good contrast
        - 40-60: Acceptable
        - < 40: Low contrast

        Args:
            gray: Grayscale image

        Returns:
            Contrast score (standard deviation of pixel intensities)
        """
        try:
            std_dev = np.std(gray)
            logger.debug(f"Contrast score (std dev): {std_dev:.2f}")
            return float(std_dev)
        except Exception as e:
            logger.warning(f"Contrast score computation failed: {e}")
            return 0.0

    def _compute_contrast_ratio(self, gray: np.ndarray) -> float:
        """
        Compute normalized contrast ratio (std/mean).

        Normalized measure independent of brightness. Typically:
        - > 0.5: Good contrast
        - 0.3-0.5: Acceptable
        - < 0.3: Low contrast

        Args:
            gray: Grayscale image

        Returns:
            Contrast ratio (std/mean)
        """
        try:
            mean_val = np.mean(gray)
            std_val = np.std(gray)

            if mean_val > 0:
                ratio = std_val / mean_val
            else:
                ratio = 0.0

            logger.debug(f"Contrast ratio (std/mean): {ratio:.3f}")
            return float(ratio)
        except Exception as e:
            logger.warning(f"Contrast ratio computation failed: {e}")
            return 0.0

    def _compute_noise_level(self, gray: np.ndarray) -> float:
        """
        Estimate noise level using local variance.

        Uses a median filtering approach to estimate noise. Typically:
        - < 30: Clean
        - 30-50: Acceptable
        - > 50: Noisy

        Args:
            gray: Grayscale image

        Returns:
            Noise level estimate
        """
        try:
            import cv2

            # Use median filter to estimate signal
            median = cv2.medianBlur(gray, 5)

            # Noise is difference between original and median-filtered
            noise = gray.astype(np.float32) - median.astype(np.float32)

            # Standard deviation of noise
            noise_level = np.std(noise)

            logger.debug(f"Noise level (local variance): {noise_level:.2f}")
            return float(noise_level)

        except Exception as e:
            logger.warning(f"Noise level computation failed: {e}")
            return 0.0

    def _compute_edge_strength(self, gray: np.ndarray) -> float:
        """
        Compute edge strength using Sobel gradients.

        Higher values indicate stronger edges. Typically:
        - > 30: Strong edges
        - 15-30: Moderate edges
        - < 15: Weak edges

        Args:
            gray: Grayscale image

        Returns:
            Mean edge strength (Sobel gradient magnitude)
        """
        try:
            import cv2

            # Compute Sobel gradients
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Gradient magnitude
            magnitude = np.sqrt(sobelx**2 + sobely**2)

            # Mean magnitude as edge strength
            edge_strength = np.mean(magnitude)

            logger.debug(f"Edge strength (Sobel magnitude): {edge_strength:.2f}")
            return float(edge_strength)

        except Exception as e:
            logger.warning(f"Edge strength computation failed: {e}")
            return 0.0

    def _compute_brightness(self, gray: np.ndarray) -> float:
        """
        Compute average brightness.

        Args:
            gray: Grayscale image

        Returns:
            Mean pixel intensity (0-255)
        """
        try:
            brightness = np.mean(gray)
            logger.debug(f"Brightness (mean intensity): {brightness:.2f}")
            return float(brightness)
        except Exception as e:
            logger.warning(f"Brightness computation failed: {e}")
            return 0.0

    def _compute_dynamic_range(self, gray: np.ndarray) -> float:
        """
        Compute dynamic range (max - min intensity).

        Args:
            gray: Grayscale image

        Returns:
            Dynamic range (0-255)
        """
        try:
            dynamic_range = float(np.max(gray)) - float(np.min(gray))
            logger.debug(f"Dynamic range: {dynamic_range:.2f}")
            return dynamic_range
        except Exception as e:
            logger.warning(f"Dynamic range computation failed: {e}")
            return 0.0

    def _classify_document_type(
        self,
        blur_score: float,
        contrast_score: float,
        noise_level: float,
        edge_strength: float,
        brightness: float,
        dynamic_range: float
    ) -> DocumentType:
        """
        Classify document type based on quality metrics.

        Uses heuristics to determine document origin:
        - High quality scan: sharp, good contrast, low noise
        - Low quality scan: moderate blur, acceptable contrast
        - Good photo: good sharpness, variable contrast
        - Poor photo: blurry, low contrast, noisy
        - Digital-born: very sharp, perfect contrast, no noise
        - Receipt: low contrast, faded, specific brightness range

        Args:
            blur_score: Laplacian variance
            contrast_score: Standard deviation
            noise_level: Local variance estimate
            edge_strength: Sobel magnitude
            brightness: Mean intensity
            dynamic_range: Max - min intensity

        Returns:
            Classified DocumentType
        """
        # Digital-born (PDF export, computer-generated)
        # Very sharp, perfect contrast, minimal noise, strong edges
        if (blur_score > 500 and contrast_score > 60 and
            noise_level < 20 and edge_strength > 40):
            return DocumentType.DIGITAL_BORN

        # High-quality scan
        # Sharp, good contrast, low noise
        if (blur_score > 200 and contrast_score > 50 and
            noise_level < 30 and edge_strength > 25):
            return DocumentType.SCAN_HIGH_QUALITY

        # Thermal receipt
        # Low contrast, faded, specific brightness characteristics
        if (contrast_score < 35 and dynamic_range < 150 and
            brightness > 180):
            return DocumentType.RECEIPT

        # Photo detection (broader criteria)
        # Photos typically have:
        # - Low to moderate blur scores (< 150)
        # - Good to moderate contrast (35-70 range)
        # - Low noise if well-lit, higher if poor lighting
        # - Continuous tone (not binary like scans)
        
        # Blurry photos (common for handheld shots)
        if (blur_score < 60 and contrast_score > 35 and noise_level < 40):
            # Blurry but decent contrast/low noise = photo
            return DocumentType.PHOTO_POOR_LIGHTING
        
        # Poor lighting photos
        if ((brightness < 80 or brightness > 200) and contrast_score < 45):
            return DocumentType.PHOTO_POOR_LIGHTING
        
        # Good lighting photos
        if (60 <= blur_score <= 150 and contrast_score > 40 and noise_level < 60):
            return DocumentType.PHOTO_GOOD_LIGHTING

        # Low-quality scan
        # Some blur, acceptable contrast
        if (blur_score > 50 and contrast_score > 35 and
            noise_level < 70):
            return DocumentType.SCAN_LOW_QUALITY

        # Cannot classify confidently
        return DocumentType.UNKNOWN

    def _default_metrics(self) -> ImageQualityMetrics:
        """
        Return default metrics when analysis fails.

        Assumes worst-case quality to enable preprocessing.
        """
        return ImageQualityMetrics(
            blur_score=0.0,
            contrast_score=0.0,
            contrast_ratio=0.0,
            noise_level=100.0,
            edge_strength=0.0,
            brightness=128.0,
            dynamic_range=0.0,
            document_type=DocumentType.UNKNOWN,
            is_blurry=True,
            is_low_contrast=True,
            is_noisy=True,
            needs_preprocessing=True
        )


def analyze_image_quality(
    image: np.ndarray,
    blur_threshold: float = 150.0,
    contrast_threshold: float = 50.0,
    noise_threshold: float = 40.0
) -> ImageQualityMetrics:
    """
    Convenience function to analyze image quality.

    Args:
        image: Input image (grayscale or RGB)
        blur_threshold: Threshold for blur detection
        contrast_threshold: Threshold for contrast detection
        noise_threshold: Threshold for noise detection

    Returns:
        ImageQualityMetrics with comprehensive analysis

    Example:
        metrics = analyze_image_quality(image)
        if metrics.needs_preprocessing:
            # Apply preprocessing
    """
    analyzer = ImageQualityAnalyzer(
        blur_threshold=blur_threshold,
        contrast_threshold=contrast_threshold,
        noise_threshold=noise_threshold
    )
    return analyzer.analyze(image)
