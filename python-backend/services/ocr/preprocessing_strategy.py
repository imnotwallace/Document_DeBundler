"""
Preprocessing Strategy Selection for Intelligent OCR

Determines optimal preprocessing techniques based on image quality metrics.
Implements safe combination rules to prevent distortion and ensure quality.
"""

import logging
from typing import List, Set, Optional
from dataclasses import dataclass
from enum import Enum

from .image_quality_analyzer import ImageQualityMetrics, DocumentType

logger = logging.getLogger(__name__)


class PreprocessingTechnique(Enum):
    """Available preprocessing techniques"""
    DENOISE = "denoise"
    CONTRAST = "contrast"
    SHARPEN = "sharpen"
    ADVANCED_DEBLUR = "advanced_deblur"  # Richardson-Lucy
    BINARIZE = "binarize"
    ADAPTIVE_BINARIZE = "adaptive_binarize"  # Sauvola/Wolf
    MORPH_OPEN = "morph_open"     # Remove noise (binary only)
    MORPH_CLOSE = "morph_close"   # Fill gaps (binary only)
    DESKEW = "deskew"


@dataclass
class PreprocessingStrategy:
    """
    A preprocessing strategy with ordered techniques.

    Attributes:
        techniques: Ordered list of techniques to apply
        rationale: Human-readable explanation of why this strategy was chosen
        expected_improvement: Estimated quality improvement (low/medium/high)
        apply_validation: Whether to validate quality improvement
    """
    techniques: List[PreprocessingTechnique]
    rationale: str
    expected_improvement: str  # "low", "medium", "high"
    apply_validation: bool = True

    def __str__(self) -> str:
        tech_names = [t.value for t in self.techniques]
        return f"Strategy({', '.join(tech_names)}) - {self.rationale}"


class PreprocessingStrategySelector:
    """
    Selects optimal preprocessing strategy based on image quality metrics.

    Uses quality analysis to determine which techniques will improve OCR:
    - Analyzes quality metrics (blur, contrast, noise)
    - Considers document type
    - Applies safe combination rules
    - Returns ordered technique sequence

    Example:
        selector = PreprocessingStrategySelector()
        strategy = selector.select_strategy(quality_metrics)

        for technique in strategy.techniques:
            image = apply_technique(image, technique)
    """

    # Safe combination rules: techniques that conflict and should not be combined
    MUTUALLY_EXCLUSIVE = {
        frozenset([PreprocessingTechnique.SHARPEN, PreprocessingTechnique.ADVANCED_DEBLUR]),
        frozenset([PreprocessingTechnique.BINARIZE, PreprocessingTechnique.ADAPTIVE_BINARIZE]),
    }

    # Techniques that require binary images
    BINARY_ONLY = {
        PreprocessingTechnique.MORPH_OPEN,
        PreprocessingTechnique.MORPH_CLOSE,
    }

    # Optimal technique ordering (earlier = applied first)
    TECHNIQUE_ORDER = [
        PreprocessingTechnique.DENOISE,          # 1. Remove noise first
        PreprocessingTechnique.CONTRAST,         # 2. Improve dynamic range
        PreprocessingTechnique.ADVANCED_DEBLUR,  # 3. Advanced deblurring
        PreprocessingTechnique.SHARPEN,          # 3. Or simple sharpening
        PreprocessingTechnique.DESKEW,           # 4. Correct rotation
        PreprocessingTechnique.ADAPTIVE_BINARIZE,# 5. Adaptive binarization
        PreprocessingTechnique.BINARIZE,         # 5. Or simple binarization
        PreprocessingTechnique.MORPH_OPEN,       # 6. Morphological cleanup
        PreprocessingTechnique.MORPH_CLOSE,      # 7. Fill gaps
    ]

    def __init__(self):
        """Initialize strategy selector"""
        logger.info("PreprocessingStrategySelector initialized")

    def select_strategy(
        self,
        metrics: ImageQualityMetrics,
        allow_destructive: bool = True
    ) -> PreprocessingStrategy:
        """
        Select optimal preprocessing strategy based on quality metrics.

        Args:
            metrics: Image quality metrics from ImageQualityAnalyzer
            allow_destructive: Allow techniques like binarization (default: True)

        Returns:
            PreprocessingStrategy with ordered technique list
        """
        logger.debug(f"Selecting strategy for: {metrics.document_type.value}")

        # If image is already high quality, skip preprocessing
        if not metrics.needs_preprocessing:
            logger.info("Image is high quality, no preprocessing needed")
            return PreprocessingStrategy(
                techniques=[],
                rationale="High quality image, no preprocessing required",
                expected_improvement="none",
                apply_validation=False
            )

        # Select techniques based on quality issues
        selected = self._select_techniques_for_metrics(metrics, allow_destructive)

        # Ensure safe combinations
        selected = self._enforce_safe_combinations(selected, metrics)

        # Order techniques optimally
        ordered = self._order_techniques(selected)

        # Generate rationale
        rationale = self._generate_rationale(metrics, ordered)

        # Estimate improvement
        improvement = self._estimate_improvement(metrics, ordered)

        strategy = PreprocessingStrategy(
            techniques=ordered,
            rationale=rationale,
            expected_improvement=improvement,
            apply_validation=True
        )

        logger.info(f"Selected strategy: {strategy}")
        return strategy

    def _select_techniques_for_metrics(
        self,
        metrics: ImageQualityMetrics,
        allow_destructive: bool
    ) -> Set[PreprocessingTechnique]:
        """
        Select techniques based on specific quality issues.

        Args:
            metrics: Quality metrics
            allow_destructive: Allow destructive techniques

        Returns:
            Set of selected preprocessing techniques
        """
        techniques = set()

        # Handle noise
        if metrics.is_noisy:
            techniques.add(PreprocessingTechnique.DENOISE)
            logger.debug("Added DENOISE (noisy image)")

        # Handle low contrast
        if metrics.is_low_contrast:
            techniques.add(PreprocessingTechnique.CONTRAST)
            logger.debug("Added CONTRAST (low contrast)")

        # Handle blur
        if metrics.is_blurry:
            # Use advanced deblur for scans/documents, sharpen for photos
            if metrics.document_type in [DocumentType.SCAN_LOW_QUALITY, 
                                        DocumentType.SCAN_HIGH_QUALITY,
                                        DocumentType.DIGITAL_BORN]:
                techniques.add(PreprocessingTechnique.ADVANCED_DEBLUR)
                logger.debug("Added ADVANCED_DEBLUR (blurry scan/document)")
            else:
                techniques.add(PreprocessingTechnique.SHARPEN)
                logger.debug("Added SHARPEN (blurry photo)")

        # Document type-specific techniques
        if metrics.document_type == DocumentType.RECEIPT:
            if allow_destructive:
                techniques.add(PreprocessingTechnique.CONTRAST)
                techniques.add(PreprocessingTechnique.ADAPTIVE_BINARIZE)
                techniques.add(PreprocessingTechnique.MORPH_CLOSE)  # Fill faded text gaps
                logger.debug("Added RECEIPT-specific techniques")

        elif metrics.document_type == DocumentType.PHOTO_POOR_LIGHTING:
            techniques.add(PreprocessingTechnique.CONTRAST)
            techniques.add(PreprocessingTechnique.DENOISE)
            if metrics.is_blurry:
                techniques.add(PreprocessingTechnique.SHARPEN)
            logger.debug("Added PHOTO_POOR_LIGHTING-specific techniques")

        elif metrics.document_type == DocumentType.SCAN_LOW_QUALITY:
            techniques.add(PreprocessingTechnique.SHARPEN)
            if metrics.is_noisy:
                techniques.add(PreprocessingTechnique.DENOISE)
            logger.debug("Added SCAN_LOW_QUALITY-specific techniques")

        # Add binarization for very low quality if allowed
        if allow_destructive and (
            (metrics.contrast_score < 30 and metrics.document_type != DocumentType.RECEIPT) or
            (metrics.is_blurry and metrics.is_low_contrast and metrics.is_noisy)
        ):
            # Use adaptive binarization for better results
            techniques.add(PreprocessingTechnique.ADAPTIVE_BINARIZE)
            logger.debug("Added ADAPTIVE_BINARIZE (very low quality)")

        return techniques

    def _enforce_safe_combinations(
        self,
        techniques: Set[PreprocessingTechnique],
        metrics: ImageQualityMetrics
    ) -> Set[PreprocessingTechnique]:
        """
        Ensure selected techniques can be safely combined.

        Removes conflicting techniques based on priority rules.

        Args:
            techniques: Initially selected techniques
            metrics: Quality metrics for priority decisions

        Returns:
            Safe set of techniques
        """
        safe_techniques = techniques.copy()

        # Check for mutually exclusive techniques
        for exclusive_set in self.MUTUALLY_EXCLUSIVE:
            present = exclusive_set & techniques
            if len(present) > 1:
                # Multiple exclusive techniques present, choose best
                chosen = self._choose_best_technique(present, metrics)
                # Remove all but chosen
                for tech in present:
                    if tech != chosen:
                        safe_techniques.remove(tech)
                        logger.debug(f"Removed {tech.value} (conflicts with {chosen.value})")

        # If binarization is used, ensure morphological ops are included
        has_binarize = (PreprocessingTechnique.BINARIZE in safe_techniques or
                       PreprocessingTechnique.ADAPTIVE_BINARIZE in safe_techniques)

        if has_binarize:
            # Remove non-binary techniques that conflict with binarization
            if PreprocessingTechnique.SHARPEN in safe_techniques:
                # Sharpen before binarize is OK
                pass

        # If morphological ops present without binarization, remove them
        if not has_binarize:
            for binary_tech in self.BINARY_ONLY:
                if binary_tech in safe_techniques:
                    safe_techniques.remove(binary_tech)
                    logger.debug(f"Removed {binary_tech.value} (no binarization)")

        return safe_techniques

    def _choose_best_technique(
        self,
        conflicting: Set[PreprocessingTechnique],
        metrics: ImageQualityMetrics
    ) -> PreprocessingTechnique:
        """
        Choose the best technique from a conflicting set.

        Args:
            conflicting: Set of conflicting techniques
            metrics: Quality metrics for decision

        Returns:
            Best technique from the set
        """
        # Advanced deblur vs sharpen: choose based on blur severity
        if {PreprocessingTechnique.SHARPEN, PreprocessingTechnique.ADVANCED_DEBLUR}.issubset(conflicting):
            if metrics.blur_score < 50:
                return PreprocessingTechnique.ADVANCED_DEBLUR
            else:
                return PreprocessingTechnique.SHARPEN

        # Adaptive binarize vs simple binarize: prefer adaptive
        if {PreprocessingTechnique.BINARIZE, PreprocessingTechnique.ADAPTIVE_BINARIZE}.issubset(conflicting):
            return PreprocessingTechnique.ADAPTIVE_BINARIZE

        # Default: return first in technique order
        for tech in self.TECHNIQUE_ORDER:
            if tech in conflicting:
                return tech

        return list(conflicting)[0]

    def _order_techniques(
        self,
        techniques: Set[PreprocessingTechnique]
    ) -> List[PreprocessingTechnique]:
        """
        Order techniques for optimal application.

        Applies techniques in safe order:
        1. Denoise (remove artifacts first)
        2. Contrast (improve dynamic range)
        3. Deblur/Sharpen (enhance edges)
        4. Deskew (correct rotation)
        5. Binarize (most destructive, last)
        6. Morphological (cleanup after binarization)

        Args:
            techniques: Unordered set of techniques

        Returns:
            Ordered list of techniques
        """
        ordered = []
        for tech in self.TECHNIQUE_ORDER:
            if tech in techniques:
                ordered.append(tech)

        logger.debug(f"Technique order: {[t.value for t in ordered]}")
        return ordered

    def _generate_rationale(
        self,
        metrics: ImageQualityMetrics,
        techniques: List[PreprocessingTechnique]
    ) -> str:
        """Generate human-readable rationale for strategy"""
        issues = []
        if metrics.is_blurry:
            issues.append("blurry")
        if metrics.is_low_contrast:
            issues.append("low contrast")
        if metrics.is_noisy:
            issues.append("noisy")

        if not issues:
            return "High quality image"

        issue_str = ", ".join(issues)
        doc_type = metrics.document_type.value.replace("_", " ")

        return f"{doc_type.title()} with {issue_str} - applying {len(techniques)} techniques"

    def _estimate_improvement(
        self,
        metrics: ImageQualityMetrics,
        techniques: List[PreprocessingTechnique]
    ) -> str:
        """
        Estimate expected quality improvement.

        Args:
            metrics: Quality metrics
            techniques: Selected techniques

        Returns:
            "low", "medium", or "high"
        """
        # More severe issues = higher improvement potential
        issue_count = sum([
            metrics.is_blurry,
            metrics.is_low_contrast,
            metrics.is_noisy
        ])

        # More techniques = potentially higher improvement
        technique_count = len(techniques)

        if issue_count == 0 or technique_count == 0:
            return "none"
        elif issue_count == 1 and technique_count <= 2:
            return "low"
        elif issue_count == 2 or technique_count <= 4:
            return "medium"
        else:
            return "high"


def select_preprocessing_strategy(
    metrics: ImageQualityMetrics,
    allow_destructive: bool = True
) -> PreprocessingStrategy:
    """
    Convenience function to select preprocessing strategy.

    Args:
        metrics: Image quality metrics
        allow_destructive: Allow destructive techniques like binarization

    Returns:
        PreprocessingStrategy with ordered techniques

    Example:
        metrics = analyze_image_quality(image)
        strategy = select_preprocessing_strategy(metrics)

        for technique in strategy.techniques:
            image = apply_technique(image, technique)
    """
    selector = PreprocessingStrategySelector()
    return selector.select_strategy(metrics, allow_destructive)
