"""
OCR-Based Preprocessing Validator

Validates preprocessing by running actual OCR and comparing results.
This is more reliable than image quality metrics for OCR optimization.
"""

import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRValidationResult:
    """Results from OCR-based validation"""
    original_text_length: int
    preprocessed_text_length: int
    original_confidence: float  # Average confidence score (if available)
    preprocessed_confidence: float
    improvement_pct: float
    use_preprocessed: bool
    reason: str


class OCRBasedValidator:
    """
    Validates preprocessing effectiveness using actual OCR results.

    This is the gold standard for preprocessing validation - we directly
    measure whether OCR improves, rather than relying on quality metrics.
    """

    def __init__(
        self,
        min_improvement_pct: float = 5.0,
        confidence_weight: float = 0.3
    ):
        """
        Initialize OCR-based validator.

        Args:
            min_improvement_pct: Minimum % improvement in text length to accept
            confidence_weight: How much to weight confidence vs text length (0-1)
        """
        self.min_improvement_pct = min_improvement_pct
        self.confidence_weight = confidence_weight

    def validate(
        self,
        original_image: np.ndarray,
        preprocessed_image: np.ndarray,
        ocr_service  # OCRService instance
    ) -> OCRValidationResult:
        """
        Validate preprocessing by running OCR on both images.

        Args:
            original_image: Original image
            preprocessed_image: Preprocessed image
            ocr_service: OCRService instance for text extraction

        Returns:
            Validation result with decision
        """
        try:
            # Run OCR on original
            original_text = ocr_service.extract_text_from_array(original_image)
            original_len = len(original_text.strip())

            # Run OCR on preprocessed
            preprocessed_text = ocr_service.extract_text_from_array(preprocessed_image)
            preprocessed_len = len(preprocessed_text.strip())

            # Calculate improvement
            if original_len > 0:
                improvement_pct = ((preprocessed_len - original_len) / original_len) * 100
            else:
                improvement_pct = 100.0 if preprocessed_len > 0 else 0.0

            # TODO: Extract confidence scores if OCR engine provides them
            original_conf = 0.0
            preprocessed_conf = 0.0

            # Decide whether to use preprocessed
            use_preprocessed = improvement_pct >= self.min_improvement_pct

            if use_preprocessed:
                reason = f"OCR improved by {improvement_pct:.1f}% ({original_len} -> {preprocessed_len} chars)"
            elif improvement_pct < 0:
                reason = f"OCR degraded by {abs(improvement_pct):.1f}% - using original"
            else:
                reason = f"Improvement too small ({improvement_pct:.1f}% < {self.min_improvement_pct}%) - using original"

            logger.info(f"OCR validation: {reason}")

            return OCRValidationResult(
                original_text_length=original_len,
                preprocessed_text_length=preprocessed_len,
                original_confidence=original_conf,
                preprocessed_confidence=preprocessed_conf,
                improvement_pct=improvement_pct,
                use_preprocessed=use_preprocessed,
                reason=reason
            )

        except Exception as e:
            logger.error(f"OCR validation failed: {e}")
            # On error, default to original
            return OCRValidationResult(
                original_text_length=0,
                preprocessed_text_length=0,
                original_confidence=0.0,
                preprocessed_confidence=0.0,
                improvement_pct=0.0,
                use_preprocessed=False,
                reason=f"Validation failed: {e}"
            )

    def find_best_technique(
        self,
        image: np.ndarray,
        techniques: List[Tuple[str, np.ndarray]],
        ocr_service
    ) -> Tuple[str, np.ndarray, OCRValidationResult]:
        """
        Test multiple preprocessing techniques and return the best one.

        Args:
            image: Original image
            techniques: List of (name, preprocessed_image) tuples to test
            ocr_service: OCRService instance

        Returns:
            (best_name, best_image, best_result)
        """
        best_name = "original"
        best_image = image
        best_result = None
        best_score = 0

        # Test each technique
        results = []
        for name, preprocessed in techniques:
            result = self.validate(image, preprocessed, ocr_service)
            results.append((name, preprocessed, result))

            # Calculate score (text length + confidence)
            score = (result.preprocessed_text_length * (1 - self.confidence_weight) +
                    result.preprocessed_confidence * self.confidence_weight)

            if score > best_score:
                best_score = score
                best_name = name
                best_image = preprocessed
                best_result = result

        logger.info(f"Best technique: {best_name} (score: {best_score:.1f})")

        return best_name, best_image, best_result


def create_ocr_validator(
    min_improvement_pct: float = 5.0
) -> OCRBasedValidator:
    """
    Factory function to create OCR-based validator.

    Args:
        min_improvement_pct: Minimum improvement to accept preprocessing

    Returns:
        Configured validator
    """
    return OCRBasedValidator(min_improvement_pct=min_improvement_pct)
