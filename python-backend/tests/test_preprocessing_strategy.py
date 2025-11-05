"""
Unit Tests for PreprocessingStrategySelector

Tests strategy selection logic and safe combination rules.
"""

import pytest
import numpy as np

from services.ocr.preprocessing_strategy import (
    PreprocessingStrategySelector,
    PreprocessingStrategy,
    PreprocessingTechnique,
    select_preprocessing_strategy
)
from services.ocr.image_quality_analyzer import (
    ImageQualityAnalyzer,
    ImageQualityMetrics,
    DocumentType
)
from .test_fixtures_preprocessing import (
    blurry_image,
    low_contrast_image,
    noisy_image,
    perfect_image,
    low_quality_image,
    receipt_image
)


class TestPreprocessingStrategySelector:
    """Test PreprocessingStrategySelector class"""

    def test_selector_initialization(self):
        """Test selector initializes correctly"""
        selector = PreprocessingStrategySelector()
        assert selector is not None

    def test_select_strategy_for_perfect_image(self, perfect_image):
        """Test no preprocessing selected for perfect image"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(perfect_image)

        selector = PreprocessingStrategySelector()
        strategy = selector.select_strategy(metrics)

        # Perfect image should not need preprocessing
        assert len(strategy.techniques) == 0, "Perfect image should have no techniques"
        assert strategy.expected_improvement == "none"

    def test_select_strategy_for_blurry_image(self, blurry_image):
        """Test deblurring/sharpening selected for blurry image"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(blurry_image)

        selector = PreprocessingStrategySelector()
        strategy = selector.select_strategy(metrics)

        # Should include sharpening or advanced deblurring
        assert (
            PreprocessingTechnique.SHARPEN in strategy.techniques or
            PreprocessingTechnique.ADVANCED_DEBLUR in strategy.techniques
        ), "Blurry image should get sharpen or deblur"

    def test_select_strategy_for_low_contrast_image(self, low_contrast_image):
        """Test contrast enhancement selected for low contrast image"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(low_contrast_image)

        selector = PreprocessingStrategySelector()
        strategy = selector.select_strategy(metrics)

        # Should include contrast enhancement
        assert PreprocessingTechnique.CONTRAST in strategy.techniques, \
            "Low contrast image should get contrast enhancement"

    def test_select_strategy_for_noisy_image(self, noisy_image):
        """Test denoising selected for noisy image"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(noisy_image)

        selector = PreprocessingStrategySelector()
        strategy = selector.select_strategy(metrics)

        # Should include denoising
        assert PreprocessingTechnique.DENOISE in strategy.techniques, \
            "Noisy image should get denoising"

    def test_select_strategy_for_receipt(self, receipt_image):
        """Test receipt-specific strategy"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(receipt_image)

        selector = PreprocessingStrategySelector()
        strategy = selector.select_strategy(metrics, allow_destructive=True)

        # Should include adaptive binarization for receipt
        assert (
            PreprocessingTechnique.ADAPTIVE_BINARIZE in strategy.techniques or
            PreprocessingTechnique.BINARIZE in strategy.techniques
        ), "Receipt should get binarization"

    def test_select_strategy_respects_allow_destructive(self, low_quality_image):
        """Test allow_destructive flag prevents binarization"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(low_quality_image)

        selector = PreprocessingStrategySelector()

        # With destructive techniques allowed
        strategy_allow = selector.select_strategy(metrics, allow_destructive=True)

        # Without destructive techniques
        strategy_deny = selector.select_strategy(metrics, allow_destructive=False)

        # Denied strategy should not have binarization or morphological ops
        destructive_techniques = {
            PreprocessingTechnique.BINARIZE,
            PreprocessingTechnique.ADAPTIVE_BINARIZE,
            PreprocessingTechnique.MORPH_OPEN,
            PreprocessingTechnique.MORPH_CLOSE
        }

        has_destructive = any(t in destructive_techniques for t in strategy_deny.techniques)
        assert not has_destructive, "Destructive techniques should be excluded when allow_destructive=False"

    def test_safe_combination_rules_no_double_sharpen(self):
        """Test safe combination rules prevent multiple sharpening"""
        selector = PreprocessingStrategySelector()

        # Manually create a conflicting set
        conflicting = {
            PreprocessingTechnique.SHARPEN,
            PreprocessingTechnique.ADVANCED_DEBLUR
        }

        # Create mock metrics
        from services.ocr.image_quality_analyzer import ImageQualityMetrics, DocumentType
        mock_metrics = ImageQualityMetrics(
            blur_score=50.0,
            contrast_score=30.0,
            contrast_ratio=0.2,
            noise_level=60.0,
            edge_strength=15.0,
            brightness=128.0,
            dynamic_range=150.0,
            document_type=DocumentType.SCAN_LOW_QUALITY,
            is_blurry=True,
            is_low_contrast=True,
            is_noisy=True,
            needs_preprocessing=True
        )

        # Enforce safe combinations
        safe_techniques = selector._enforce_safe_combinations(conflicting, mock_metrics)

        # Should only have one sharpening technique
        sharpen_count = sum([
            PreprocessingTechnique.SHARPEN in safe_techniques,
            PreprocessingTechnique.ADVANCED_DEBLUR in safe_techniques
        ])

        assert sharpen_count == 1, "Should only have one sharpening technique"

    def test_technique_ordering(self):
        """Test techniques are ordered correctly"""
        selector = PreprocessingStrategySelector()

        # Create unordered set
        techniques = {
            PreprocessingTechnique.BINARIZE,
            PreprocessingTechnique.DENOISE,
            PreprocessingTechnique.SHARPEN,
            PreprocessingTechnique.CONTRAST
        }

        ordered = selector._order_techniques(techniques)

        # Verify order: denoise -> contrast -> sharpen -> binarize
        denoise_idx = ordered.index(PreprocessingTechnique.DENOISE)
        contrast_idx = ordered.index(PreprocessingTechnique.CONTRAST)
        sharpen_idx = ordered.index(PreprocessingTechnique.SHARPEN)
        binarize_idx = ordered.index(PreprocessingTechnique.BINARIZE)

        assert denoise_idx < contrast_idx, "Denoise should come before contrast"
        assert contrast_idx < sharpen_idx, "Contrast should come before sharpen"
        assert sharpen_idx < binarize_idx, "Sharpen should come before binarize"

    def test_strategy_generation_has_rationale(self, low_contrast_image):
        """Test generated strategy has rationale"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(low_contrast_image)

        selector = PreprocessingStrategySelector()
        strategy = selector.select_strategy(metrics)

        # Should have non-empty rationale
        assert len(strategy.rationale) > 0, "Strategy should have rationale"
        assert isinstance(strategy.rationale, str)

    def test_improvement_estimation(self):
        """Test improvement estimation logic"""
        selector = PreprocessingStrategySelector()

        # Create metrics with different issue counts
        # No issues
        perfect_metrics = ImageQualityMetrics(
            blur_score=500.0,
            contrast_score=80.0,
            contrast_ratio=0.6,
            noise_level=10.0,
            edge_strength=40.0,
            brightness=180.0,
            dynamic_range=200.0,
            document_type=DocumentType.DIGITAL_BORN,
            is_blurry=False,
            is_low_contrast=False,
            is_noisy=False,
            needs_preprocessing=False
        )

        # Multiple issues
        poor_metrics = ImageQualityMetrics(
            blur_score=30.0,
            contrast_score=25.0,
            contrast_ratio=0.2,
            noise_level=70.0,
            edge_strength=10.0,
            brightness=100.0,
            dynamic_range=80.0,
            document_type=DocumentType.PHOTO_POOR_LIGHTING,
            is_blurry=True,
            is_low_contrast=True,
            is_noisy=True,
            needs_preprocessing=True
        )

        perfect_improvement = selector._estimate_improvement(
            perfect_metrics,
            []
        )

        poor_improvement = selector._estimate_improvement(
            poor_metrics,
            [PreprocessingTechnique.DENOISE, PreprocessingTechnique.CONTRAST,
             PreprocessingTechnique.SHARPEN, PreprocessingTechnique.ADAPTIVE_BINARIZE]
        )

        # Poor image with many techniques should have higher estimated improvement
        improvement_levels = ["none", "low", "medium", "high"]
        assert improvement_levels.index(perfect_improvement) < improvement_levels.index(poor_improvement), \
            f"Poor image should have higher improvement estimate than perfect: {perfect_improvement} vs {poor_improvement}"

    def test_morphological_ops_only_with_binarization(self):
        """Test morphological ops only selected when binarization present"""
        selector = PreprocessingStrategySelector()

        # Mock metrics
        metrics = ImageQualityMetrics(
            blur_score=50.0,
            contrast_score=30.0,
            contrast_ratio=0.2,
            noise_level=60.0,
            edge_strength=15.0,
            brightness=128.0,
            dynamic_range=150.0,
            document_type=DocumentType.RECEIPT,
            is_blurry=True,
            is_low_contrast=True,
            is_noisy=True,
            needs_preprocessing=True
        )

        # Create set with morph ops but no binarization
        techniques_without_binary = {
            PreprocessingTechnique.DENOISE,
            PreprocessingTechnique.CONTRAST,
            PreprocessingTechnique.MORPH_CLOSE  # Should be removed
        }

        safe_techniques = selector._enforce_safe_combinations(techniques_without_binary, metrics)

        # Morphological ops should be removed
        assert PreprocessingTechnique.MORPH_CLOSE not in safe_techniques, \
            "Morphological ops should be removed without binarization"


class TestConvenienceFunction:
    """Test convenience function select_preprocessing_strategy"""

    def test_convenience_function(self, blurry_image):
        """Test convenience function works correctly"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(blurry_image)

        strategy = select_preprocessing_strategy(metrics)

        assert isinstance(strategy, PreprocessingStrategy)
        assert len(strategy.techniques) > 0

    def test_convenience_function_with_destructive_flag(self, receipt_image):
        """Test convenience function respects allow_destructive"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(receipt_image)

        strategy_allow = select_preprocessing_strategy(metrics, allow_destructive=True)
        strategy_deny = select_preprocessing_strategy(metrics, allow_destructive=False)

        # Allow should potentially have more techniques (binarization)
        assert len(strategy_allow.techniques) >= len(strategy_deny.techniques)


class TestDocumentTypeSpecificStrategies:
    """Test document type-specific strategy selection"""

    def test_digital_born_no_preprocessing(self):
        """Test digital-born documents get no preprocessing"""
        # Mock digital-born metrics (perfect quality)
        metrics = ImageQualityMetrics(
            blur_score=600.0,
            contrast_score=80.0,
            contrast_ratio=0.7,
            noise_level=5.0,
            edge_strength=50.0,
            brightness=200.0,
            dynamic_range=220.0,
            document_type=DocumentType.DIGITAL_BORN,
            is_blurry=False,
            is_low_contrast=False,
            is_noisy=False,
            needs_preprocessing=False
        )

        selector = PreprocessingStrategySelector()
        strategy = selector.select_strategy(metrics)

        # Should have no techniques
        assert len(strategy.techniques) == 0, "Digital-born should not need preprocessing"

    def test_scan_high_quality_minimal_preprocessing(self):
        """Test high quality scans get minimal preprocessing"""
        # Mock high-quality scan
        metrics = ImageQualityMetrics(
            blur_score=250.0,
            contrast_score=55.0,
            contrast_ratio=0.5,
            noise_level=20.0,
            edge_strength=30.0,
            brightness=180.0,
            dynamic_range=180.0,
            document_type=DocumentType.SCAN_HIGH_QUALITY,
            is_blurry=False,
            is_low_contrast=False,
            is_noisy=False,
            needs_preprocessing=False
        )

        selector = PreprocessingStrategySelector()
        strategy = selector.select_strategy(metrics)

        # Should have few or no techniques
        assert len(strategy.techniques) <= 1, "High quality scan should need minimal preprocessing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
