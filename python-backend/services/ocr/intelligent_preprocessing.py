"""
Intelligent OCR Preprocessing Orchestrator

Integrates quality analysis, strategy selection, and validation
to automatically optimize images for OCR without distortion.

This module provides the main entry point for intelligent preprocessing.
"""

import logging
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .image_quality_analyzer import (
    ImageQualityAnalyzer,
    ImageQualityMetrics,
    analyze_image_quality
)
from .preprocessing_strategy import (
    PreprocessingStrategy,
    PreprocessingStrategySelector,
    PreprocessingTechnique,
    select_preprocessing_strategy
)
from .preprocessing_validator import (
    PreprocessingValidator,
    ValidationResult,
    validate_preprocessing
)
from .preprocessing import ImagePreprocessor, PreprocessingPreset
from . import advanced_preprocessing

logger = logging.getLogger(__name__)


@dataclass
class IntelligentPreprocessingResult:
    """
    Result of intelligent preprocessing.

    Attributes:
        image: Final image to use for OCR (either original or preprocessed)
        original_metrics: Quality metrics of original image
        preprocessed_metrics: Quality metrics of preprocessed image (if applied)
        strategy: Strategy that was selected
        validation: Validation result (if validation was performed)
        used_preprocessed: Whether preprocessed image was used
        techniques_applied: List of techniques that were applied
    """
    image: np.ndarray
    original_metrics: ImageQualityMetrics
    preprocessed_metrics: Optional[ImageQualityMetrics]
    strategy: PreprocessingStrategy
    validation: Optional[ValidationResult]
    used_preprocessed: bool
    techniques_applied: list


class IntelligentPreprocessor:
    """
    Orchestrates intelligent preprocessing pipeline.

    Automatically analyzes image quality, selects optimal preprocessing
    strategy, applies techniques, and validates improvements.

    Pipeline:
    1. Analyze original image quality
    2. Select preprocessing strategy based on metrics
    3. Apply preprocessing techniques in optimal order
    4. Validate that preprocessing improved quality
    5. Return best image (original or preprocessed)

    Example:
        preprocessor = IntelligentPreprocessor()
        result = preprocessor.process(image)

        # Use the best image for OCR
        ocr_text = ocr_engine.process(result.image)

        # Log what was done
        logger.info(f"Used {'preprocessed' if result.used_preprocessed else 'original'}")
        logger.info(f"Techniques applied: {result.techniques_applied}")
    """

    def __init__(
        self,
        allow_destructive: bool = True,
        enable_validation: bool = True,
        min_quality_improvement: float = 5.0,
        min_ssim: float = 0.85
    ):
        """
        Initialize intelligent preprocessor.

        Args:
            allow_destructive: Allow destructive techniques like binarization
            enable_validation: Validate quality improvements
            min_quality_improvement: Minimum quality score increase to use preprocessed
            min_ssim: Minimum structural similarity to avoid distortion
        """
        self.allow_destructive = allow_destructive
        self.enable_validation = enable_validation

        self.analyzer = ImageQualityAnalyzer()
        self.strategy_selector = PreprocessingStrategySelector()
        self.validator = PreprocessingValidator(
            min_quality_improvement=min_quality_improvement,
            min_ssim=min_ssim
        )

        logger.info(
            f"IntelligentPreprocessor initialized: "
            f"destructive={allow_destructive}, validation={enable_validation}"
        )

    def process(self, image: np.ndarray) -> IntelligentPreprocessingResult:
        """
        Process image with intelligent preprocessing.

        Args:
            image: Input image (RGB or grayscale)

        Returns:
            IntelligentPreprocessingResult with best image and metadata
        """
        logger.debug("Starting intelligent preprocessing")

        # Step 1: Analyze original image quality
        original_metrics = self.analyzer.analyze(image)
        logger.info(
            f"Original quality: blur={original_metrics.blur_score:.1f}, "
            f"contrast={original_metrics.contrast_score:.1f}, "
            f"noise={original_metrics.noise_level:.1f}, "
            f"type={original_metrics.document_type.value}"
        )

        # Step 2: Select preprocessing strategy
        strategy = self.strategy_selector.select_strategy(
            original_metrics,
            allow_destructive=self.allow_destructive
        )
        logger.info(f"Strategy: {strategy}")

        # If no preprocessing needed, return original
        if not strategy.techniques:
            logger.info("No preprocessing needed, using original image")
            return IntelligentPreprocessingResult(
                image=image,
                original_metrics=original_metrics,
                preprocessed_metrics=None,
                strategy=strategy,
                validation=None,
                used_preprocessed=False,
                techniques_applied=[]
            )

        # Step 3: Apply preprocessing techniques
        try:
            preprocessed = self._apply_techniques(image, strategy.techniques)
            techniques_applied = [t.value for t in strategy.techniques]
            logger.info(f"Applied techniques: {techniques_applied}")

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}", exc_info=True)
            # Return original if preprocessing fails
            return IntelligentPreprocessingResult(
                image=image,
                original_metrics=original_metrics,
                preprocessed_metrics=None,
                strategy=strategy,
                validation=None,
                used_preprocessed=False,
                techniques_applied=[]
            )

        # Step 4: Analyze preprocessed quality
        preprocessed_metrics = self.analyzer.analyze(preprocessed)
        logger.info(
            f"Preprocessed quality: blur={preprocessed_metrics.blur_score:.1f}, "
            f"contrast={preprocessed_metrics.contrast_score:.1f}, "
            f"noise={preprocessed_metrics.noise_level:.1f}"
        )

        # Step 5: Validate improvement (if enabled)
        if self.enable_validation:
            validation = self.validator.validate(image, preprocessed)
            logger.info(f"Validation: {validation}")

            # Use validator's recommendation
            if validation.use_preprocessed:
                logger.info("Using preprocessed image (validated improvement)")
                return IntelligentPreprocessingResult(
                    image=preprocessed,
                    original_metrics=original_metrics,
                    preprocessed_metrics=preprocessed_metrics,
                    strategy=strategy,
                    validation=validation,
                    used_preprocessed=True,
                    techniques_applied=techniques_applied
                )
            else:
                logger.info(f"Using original image (validation rejected: {validation.reason})")
                return IntelligentPreprocessingResult(
                    image=image,
                    original_metrics=original_metrics,
                    preprocessed_metrics=preprocessed_metrics,
                    strategy=strategy,
                    validation=validation,
                    used_preprocessed=False,
                    techniques_applied=[]
                )

        else:
            # No validation, use preprocessed
            logger.info("Using preprocessed image (validation disabled)")
            return IntelligentPreprocessingResult(
                image=preprocessed,
                original_metrics=original_metrics,
                preprocessed_metrics=preprocessed_metrics,
                strategy=strategy,
                validation=None,
                used_preprocessed=True,
                techniques_applied=techniques_applied
            )

    def _apply_techniques(
        self,
        image: np.ndarray,
        techniques: list[PreprocessingTechnique]
    ) -> np.ndarray:
        """
        Apply preprocessing techniques in order.

        Args:
            image: Input image
            techniques: Ordered list of techniques to apply

        Returns:
            Preprocessed image
        """
        import cv2

        # Convert to grayscale for processing (no unnecessary copy)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()  # Only copy if already grayscale

        # Apply each technique in order
        for technique in techniques:
            logger.debug(f"Applying technique: {technique.value}")
            gray = self._apply_single_technique(gray, technique)

        # IMPORTANT: Return grayscale image directly (no RGB conversion needed)
        # Preprocessed images are ONLY used for OCR text extraction, then discarded
        # OCR engines (PaddleOCR/Tesseract) work perfectly with grayscale
        # Benefits: 3x less memory, faster processing, no color artifacts
        return gray

    def _apply_single_technique(
        self,
        image: np.ndarray,
        technique: PreprocessingTechnique
    ) -> np.ndarray:
        """
        Apply a single preprocessing technique.

        Args:
            image: Input image (grayscale)
            technique: Technique to apply

        Returns:
            Processed image
        """
        # Create standard preprocessor for basic techniques
        preprocessor = ImagePreprocessor(preset=PreprocessingPreset.NONE)

        if technique == PreprocessingTechnique.DENOISE:
            return preprocessor._reduce_noise(image)

        elif technique == PreprocessingTechnique.CONTRAST:
            return preprocessor._enhance_contrast(image)

        elif technique == PreprocessingTechnique.SHARPEN:
            return preprocessor._sharpen(image)

        elif technique == PreprocessingTechnique.BINARIZE:
            return preprocessor._binarize(image)

        elif technique == PreprocessingTechnique.DESKEW:
            return preprocessor._deskew(image)

        # Advanced techniques
        elif technique == PreprocessingTechnique.ADVANCED_DEBLUR:
            return advanced_preprocessing.richardson_lucy_deblur(image, iterations=10)

        elif technique == PreprocessingTechnique.ADAPTIVE_BINARIZE:
            # Try Sauvola first, fall back to Wolf if needed
            return advanced_preprocessing.adaptive_binarize_sauvola(image, k=0.2)

        elif technique == PreprocessingTechnique.MORPH_OPEN:
            return advanced_preprocessing.morph_open(image, kernel_size=3)

        elif technique == PreprocessingTechnique.MORPH_CLOSE:
            return advanced_preprocessing.morph_close(image, kernel_size=3)

        else:
            logger.warning(f"Unknown technique: {technique}")
            return image


def process_with_intelligent_preprocessing(
    image: np.ndarray,
    allow_destructive: bool = True,
    enable_validation: bool = True
) -> IntelligentPreprocessingResult:
    """
    Convenience function for intelligent preprocessing.

    Args:
        image: Input image
        allow_destructive: Allow destructive techniques like binarization
        enable_validation: Validate quality improvements

    Returns:
        IntelligentPreprocessingResult with best image

    Example:
        result = process_with_intelligent_preprocessing(image)
        ocr_text = ocr_engine.process(result.image)

        if result.used_preprocessed:
            logger.info(f"Improved quality with: {result.techniques_applied}")
    """
    preprocessor = IntelligentPreprocessor(
        allow_destructive=allow_destructive,
        enable_validation=enable_validation
    )
    return preprocessor.process(image)


def process_batch_with_intelligent_preprocessing(
    images: list[np.ndarray],
    allow_destructive: bool = True,
    enable_validation: bool = True
) -> list[IntelligentPreprocessingResult]:
    """
    Process batch of images with intelligent preprocessing.

    Args:
        images: List of input images
        allow_destructive: Allow destructive techniques
        enable_validation: Validate improvements

    Returns:
        List of IntelligentPreprocessingResult

    Example:
        results = process_batch_with_intelligent_preprocessing(images)
        for i, result in enumerate(results):
            logger.info(f"Page {i}: {result.strategy.rationale}")
    """
    preprocessor = IntelligentPreprocessor(
        allow_destructive=allow_destructive,
        enable_validation=enable_validation
    )

    results = []
    for i, image in enumerate(images):
        logger.debug(f"Processing image {i+1}/{len(images)}")
        result = preprocessor.process(image)
        results.append(result)

    return results
