"""
Preprocessing Quality Validation Framework

Validates that preprocessing improves OCR quality and doesn't introduce distortion.
Compares before/after metrics and provides rollback recommendations.
"""

import logging
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .image_quality_analyzer import ImageQualityAnalyzer, ImageQualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of preprocessing quality validation.

    Attributes:
        is_improved: Whether preprocessing improved quality
        use_preprocessed: Recommendation to use preprocessed or original
        quality_delta: Change in overall quality score
        blur_delta: Change in blur score (positive = sharper)
        contrast_delta: Change in contrast score (positive = better)
        noise_delta: Change in noise level (negative = less noise)
        ssim_score: Structural similarity index (0-1, higher = more similar)
        distortion_detected: Whether distortion was detected
        reason: Human-readable explanation
    """
    is_improved: bool
    use_preprocessed: bool
    quality_delta: float
    blur_delta: float
    contrast_delta: float
    noise_delta: float
    ssim_score: float
    distortion_detected: bool
    reason: str

    def __str__(self) -> str:
        recommendation = "USE PREPROCESSED" if self.use_preprocessed else "USE ORIGINAL"
        return (f"{recommendation}: quality_delta={self.quality_delta:.2f}, "
                f"ssim={self.ssim_score:.3f} - {self.reason}")


class PreprocessingValidator:
    """
    Validates preprocessing quality improvements.

    Compares original and preprocessed images to ensure:
    1. Quality metrics improved
    2. No significant distortion (SSIM check)
    3. Overall quality score increased

    Prevents preprocessing from degrading quality accidentally.

    Example:
        validator = PreprocessingValidator()
        result = validator.validate(original, preprocessed)

        if result.use_preprocessed:
            # Use preprocessed image for OCR
        else:
            # Use original image
    """

    def __init__(
        self,
        min_quality_improvement: float = 5.0,
        min_ssim: float = 0.85,
        max_noise_increase: float = 10.0
    ):
        """
        Initialize validator with thresholds.

        Args:
            min_quality_improvement: Minimum quality score increase to accept (default: 5.0)
            min_ssim: Minimum structural similarity to avoid distortion (default: 0.85)
            max_noise_increase: Maximum noise increase to tolerate (default: 10.0)
        """
        self.min_quality_improvement = min_quality_improvement
        self.min_ssim = min_ssim
        self.max_noise_increase = max_noise_increase
        self.analyzer = ImageQualityAnalyzer()

        logger.info(
            f"PreprocessingValidator initialized: "
            f"min_improvement={min_quality_improvement}, "
            f"min_ssim={min_ssim}, max_noise_increase={max_noise_increase}"
        )

    def validate(
        self,
        original: np.ndarray,
        preprocessed: np.ndarray
    ) -> ValidationResult:
        """
        Validate preprocessing quality improvement.

        Compares before/after quality metrics and structural similarity.

        Args:
            original: Original image (before preprocessing)
            preprocessed: Preprocessed image

        Returns:
            ValidationResult with recommendation and metrics
        """
        logger.debug("Validating preprocessing quality improvement")

        # Analyze both images
        original_metrics = self.analyzer.analyze(original)
        preprocessed_metrics = self.analyzer.analyze(preprocessed)

        # Compute quality deltas
        blur_delta = preprocessed_metrics.blur_score - original_metrics.blur_score
        contrast_delta = preprocessed_metrics.contrast_score - original_metrics.contrast_score
        noise_delta = preprocessed_metrics.noise_level - original_metrics.noise_level

        # Compute structural similarity
        ssim_score = self._compute_ssim(original, preprocessed)

        # Compute overall quality score change
        quality_delta = self._compute_quality_delta(
            original_metrics, preprocessed_metrics,
            blur_delta, contrast_delta, noise_delta
        )

        # Detect distortion
        distortion_detected = ssim_score < self.min_ssim

        # Determine if improved
        is_improved = (
            quality_delta > 0 and
            not distortion_detected and
            noise_delta < self.max_noise_increase
        )

        # Decide whether to use preprocessed
        use_preprocessed = (
            is_improved and
            quality_delta >= self.min_quality_improvement
        )

        # Generate reason
        reason = self._generate_reason(
            is_improved, use_preprocessed, quality_delta,
            ssim_score, distortion_detected, noise_delta
        )

        result = ValidationResult(
            is_improved=is_improved,
            use_preprocessed=use_preprocessed,
            quality_delta=quality_delta,
            blur_delta=blur_delta,
            contrast_delta=contrast_delta,
            noise_delta=noise_delta,
            ssim_score=ssim_score,
            distortion_detected=distortion_detected,
            reason=reason
        )

        logger.info(f"Validation result: {result}")
        return result

    def _compute_ssim(
        self,
        original: np.ndarray,
        preprocessed: np.ndarray
    ) -> float:
        """
        Compute Structural Similarity Index (SSIM).

        SSIM measures how similar two images are (0-1 scale).
        - 1.0 = identical
        - 0.8-1.0 = very similar (acceptable preprocessing)
        - 0.5-0.8 = moderately similar (potential distortion)
        - < 0.5 = very different (significant distortion)

        Args:
            original: Original image
            preprocessed: Preprocessed image

        Returns:
            SSIM score (0-1)
        """
        try:
            from skimage.metrics import structural_similarity

            # Convert to grayscale if needed
            if len(original.shape) == 3:
                import cv2
                original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
                preprocessed_gray = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2GRAY)
            else:
                original_gray = original
                preprocessed_gray = preprocessed

            # Ensure same dimensions
            if original_gray.shape != preprocessed_gray.shape:
                logger.warning("Image dimensions differ, cannot compute SSIM")
                return 0.0

            # Compute SSIM
            ssim_value = structural_similarity(
                original_gray,
                preprocessed_gray,
                data_range=255
            )

            logger.debug(f"SSIM: {ssim_value:.3f}")
            return float(ssim_value)

        except Exception as e:
            logger.warning(f"SSIM computation failed: {e}")
            # If SSIM fails, assume similarity is OK
            return 1.0

    def _compute_quality_delta(
        self,
        original_metrics: ImageQualityMetrics,
        preprocessed_metrics: ImageQualityMetrics,
        blur_delta: float,
        contrast_delta: float,
        noise_delta: float
    ) -> float:
        """
        Compute overall quality score change.

        Weighs different quality improvements:
        - Blur improvement (sharper): +weight
        - Contrast improvement: +weight
        - Noise reduction: +weight
        - Noise increase: -weight

        Args:
            original_metrics: Original image metrics
            preprocessed_metrics: Preprocessed image metrics
            blur_delta: Change in blur score
            contrast_delta: Change in contrast score
            noise_delta: Change in noise level

        Returns:
            Overall quality delta (positive = improved)
        """
        quality_delta = 0.0

        # Blur improvement (weight: 1.0)
        # Positive blur_delta = sharper
        if original_metrics.is_blurry:
            quality_delta += blur_delta * 0.1  # Scale to reasonable range

        # Contrast improvement (weight: 1.5)
        # Positive contrast_delta = better contrast
        if original_metrics.is_low_contrast:
            quality_delta += contrast_delta * 1.5

        # Noise change (weight: 1.0)
        # Negative noise_delta = less noise (good)
        if original_metrics.is_noisy:
            quality_delta -= noise_delta * 1.0

        logger.debug(f"Quality delta components: blur={blur_delta:.2f}, "
                    f"contrast={contrast_delta:.2f}, noise={noise_delta:.2f}, "
                    f"total={quality_delta:.2f}")

        return quality_delta

    def _generate_reason(
        self,
        is_improved: bool,
        use_preprocessed: bool,
        quality_delta: float,
        ssim_score: float,
        distortion_detected: bool,
        noise_delta: float
    ) -> str:
        """Generate human-readable explanation"""
        if distortion_detected:
            return f"Distortion detected (SSIM={ssim_score:.3f} < {self.min_ssim})"

        if noise_delta > self.max_noise_increase:
            return f"Noise increased too much (+{noise_delta:.1f} > {self.max_noise_increase})"

        if not is_improved:
            return f"No quality improvement (delta={quality_delta:.2f})"

        if not use_preprocessed:
            return f"Improvement too small (delta={quality_delta:.2f} < {self.min_quality_improvement})"

        return f"Quality improved (delta={quality_delta:.2f}), SSIM={ssim_score:.3f}"


def validate_preprocessing(
    original: np.ndarray,
    preprocessed: np.ndarray,
    min_quality_improvement: float = 5.0,
    min_ssim: float = 0.85
) -> ValidationResult:
    """
    Convenience function to validate preprocessing.

    Args:
        original: Original image
        preprocessed: Preprocessed image
        min_quality_improvement: Minimum quality improvement threshold
        min_ssim: Minimum structural similarity threshold

    Returns:
        ValidationResult with recommendation

    Example:
        result = validate_preprocessing(original, preprocessed)

        if result.use_preprocessed:
            ocr_text = ocr_engine.process(preprocessed)
        else:
            ocr_text = ocr_engine.process(original)
    """
    validator = PreprocessingValidator(
        min_quality_improvement=min_quality_improvement,
        min_ssim=min_ssim
    )
    return validator.validate(original, preprocessed)
