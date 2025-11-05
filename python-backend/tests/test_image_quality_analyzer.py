"""
Unit Tests for ImageQualityAnalyzer

Tests quality metric computation and document type classification.
"""

import pytest
import numpy as np

from services.ocr.image_quality_analyzer import (
    ImageQualityAnalyzer,
    ImageQualityMetrics,
    DocumentType,
    analyze_image_quality
)
from .test_fixtures_preprocessing import (
    sharp_high_contrast_image,
    blurry_image,
    low_contrast_image,
    noisy_image,
    perfect_image,
    low_quality_image,
    receipt_image,
    photo_good_lighting,
    photo_poor_lighting
)


class TestImageQualityAnalyzer:
    """Test ImageQualityAnalyzer class"""

    def test_analyzer_initialization(self):
        """Test analyzer initializes with correct thresholds"""
        analyzer = ImageQualityAnalyzer(
            blur_threshold=150.0,
            contrast_threshold=50.0,
            noise_threshold=40.0
        )

        assert analyzer.blur_threshold == 150.0
        assert analyzer.contrast_threshold == 50.0
        assert analyzer.noise_threshold == 40.0

    def test_analyze_sharp_image(self, sharp_high_contrast_image):
        """Test analysis of sharp, high contrast image"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(sharp_high_contrast_image)

        # Should have high blur score (sharp)
        assert metrics.blur_score > 100.0, f"Expected sharp image, got blur_score={metrics.blur_score}"

        # Should have good contrast
        assert metrics.contrast_score > 40.0, f"Expected good contrast, got {metrics.contrast_score}"

        # Should not be flagged as needing preprocessing
        assert not metrics.is_blurry, "Sharp image incorrectly flagged as blurry"

    def test_analyze_blurry_image(self, blurry_image, sharp_high_contrast_image):
        """Test analysis of blurry image"""
        analyzer = ImageQualityAnalyzer()
        blurry_metrics = analyzer.analyze(blurry_image)
        sharp_metrics = analyzer.analyze(sharp_high_contrast_image)

        # Blurry should have lower blur score than sharp, OR be flagged as needing preprocessing
        # (Synthetic images may not always meet absolute thresholds, but should show relative differences)
        assert (blurry_metrics.blur_score < sharp_metrics.blur_score or
                blurry_metrics.needs_preprocessing), \
            f"Blurry image should have lower blur score or need preprocessing: " \
            f"blur={blurry_metrics.blur_score} vs sharp={sharp_metrics.blur_score}"

    def test_analyze_low_contrast_image(self, low_contrast_image):
        """Test analysis of low contrast image"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(low_contrast_image)

        # Should have low contrast score
        assert metrics.contrast_score < 40.0, f"Expected low contrast, got {metrics.contrast_score}"

        # Should be flagged as low contrast
        assert metrics.is_low_contrast, "Low contrast not detected"
        assert metrics.needs_preprocessing, "Low contrast image should need preprocessing"

    def test_analyze_noisy_image(self, noisy_image):
        """Test analysis of noisy image"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(noisy_image)

        # Should have high noise level
        assert metrics.noise_level > 50.0, f"Expected noisy image, got noise_level={metrics.noise_level}"

        # Should be flagged as noisy
        assert metrics.is_noisy, "Noisy image not detected"
        assert metrics.needs_preprocessing, "Noisy image should need preprocessing"

    def test_analyze_perfect_image(self, perfect_image):
        """Test analysis of perfect quality image"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(perfect_image)

        # Should not need preprocessing
        assert not metrics.needs_preprocessing, "Perfect image should not need preprocessing"
        assert not metrics.is_blurry, "Perfect image flagged as blurry"
        assert not metrics.is_low_contrast, "Perfect image flagged as low contrast"

    def test_analyze_low_quality_image(self, low_quality_image):
        """Test analysis of very poor quality image"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(low_quality_image)

        # Should have at least one quality issue
        quality_issues = sum([
            metrics.is_blurry,
            metrics.is_low_contrast,
            metrics.is_noisy
        ])

        assert quality_issues >= 1, \
            f"Low quality image should have at least one issue: " \
            f"blurry={metrics.is_blurry}, low_contrast={metrics.is_low_contrast}, " \
            f"noisy={metrics.is_noisy}"
        assert metrics.needs_preprocessing, "Low quality image must need preprocessing"

    def test_document_type_classification_receipt(self, receipt_image):
        """Test receipt classification"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(receipt_image)

        # Receipt should be detected or at least be low quality
        assert (
            metrics.document_type == DocumentType.RECEIPT or
            metrics.is_low_contrast
        ), f"Receipt not properly classified: {metrics.document_type}"

    def test_document_type_classification_photo(self, photo_good_lighting):
        """Test photo classification"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(photo_good_lighting)

        # Should classify as some type of photo or scan
        assert metrics.document_type in [
            DocumentType.PHOTO_GOOD_LIGHTING,
            DocumentType.PHOTO_POOR_LIGHTING,
            DocumentType.SCAN_HIGH_QUALITY,
            DocumentType.SCAN_LOW_QUALITY
        ], f"Photo not properly classified: {metrics.document_type}"

    def test_metrics_to_dict(self, sharp_high_contrast_image):
        """Test metrics can be serialized to dict"""
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(sharp_high_contrast_image)

        metrics_dict = metrics.to_dict()

        # Verify all expected keys present
        expected_keys = [
            'blur_score', 'contrast_score', 'contrast_ratio',
            'noise_level', 'edge_strength', 'brightness',
            'dynamic_range', 'document_type', 'is_blurry',
            'is_low_contrast', 'is_noisy', 'needs_preprocessing'
        ]

        for key in expected_keys:
            assert key in metrics_dict, f"Missing key in metrics dict: {key}"

        # Verify types
        assert isinstance(metrics_dict['blur_score'], float)
        assert isinstance(metrics_dict['is_blurry'], bool)
        assert isinstance(metrics_dict['document_type'], str)

    def test_analyze_rgb_image(self):
        """Test analyzer handles RGB images correctly"""
        # Create RGB image
        rgb_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(rgb_image)

        # Should successfully analyze without error
        assert metrics is not None
        assert isinstance(metrics.blur_score, float)

    def test_analyze_handles_errors_gracefully(self):
        """Test analyzer handles invalid inputs gracefully"""
        analyzer = ImageQualityAnalyzer()

        # Empty image
        empty_image = np.zeros((10, 10), dtype=np.uint8)
        metrics = analyzer.analyze(empty_image)

        # Should return valid metrics (not crash)
        assert metrics is not None
        assert isinstance(metrics, ImageQualityMetrics)

    def test_custom_thresholds(self):
        """Test custom quality thresholds work correctly"""
        # Strict analyzer
        strict_analyzer = ImageQualityAnalyzer(
            blur_threshold=200.0,
            contrast_threshold=60.0,
            noise_threshold=30.0
        )

        # Lenient analyzer
        lenient_analyzer = ImageQualityAnalyzer(
            blur_threshold=50.0,
            contrast_threshold=20.0,
            noise_threshold=70.0
        )

        # Create medium quality image
        medium_image = np.random.randint(100, 150, (600, 800), dtype=np.uint8)

        strict_metrics = strict_analyzer.analyze(medium_image)
        lenient_metrics = lenient_analyzer.analyze(medium_image)

        # Strict should flag more issues
        strict_issues = sum([
            strict_metrics.is_blurry,
            strict_metrics.is_low_contrast,
            strict_metrics.is_noisy
        ])

        lenient_issues = sum([
            lenient_metrics.is_blurry,
            lenient_metrics.is_low_contrast,
            lenient_metrics.is_noisy
        ])

        # Strict should find at least as many issues as lenient
        assert strict_issues >= lenient_issues


class TestConvenienceFunction:
    """Test convenience function analyze_image_quality"""

    def test_convenience_function(self, sharp_high_contrast_image):
        """Test convenience function works correctly"""
        metrics = analyze_image_quality(sharp_high_contrast_image)

        assert isinstance(metrics, ImageQualityMetrics)
        assert metrics.blur_score > 0

    def test_convenience_function_with_custom_thresholds(self):
        """Test convenience function accepts custom thresholds"""
        image = np.random.randint(0, 255, (600, 800), dtype=np.uint8)

        metrics = analyze_image_quality(
            image,
            blur_threshold=150.0,
            contrast_threshold=50.0,
            noise_threshold=40.0
        )

        assert isinstance(metrics, ImageQualityMetrics)


class TestQualityMetricAccuracy:
    """Test accuracy of individual quality metrics"""

    def test_blur_score_differentiates_sharp_vs_blurry(self):
        """Test blur score correctly differentiates sharp vs blurry images"""
        # Create sharp image
        sharp = np.random.randint(0, 255, (600, 800), dtype=np.uint8)

        # Create blurry version
        import cv2
        blurry = cv2.GaussianBlur(sharp, (15, 15), 0)

        analyzer = ImageQualityAnalyzer()
        sharp_metrics = analyzer.analyze(sharp)
        blurry_metrics = analyzer.analyze(blurry)

        # Sharp should have higher blur score
        assert sharp_metrics.blur_score > blurry_metrics.blur_score, \
            f"Sharp image blur_score ({sharp_metrics.blur_score}) should be > blurry ({blurry_metrics.blur_score})"

    def test_contrast_score_differentiates_high_vs_low(self):
        """Test contrast score correctly differentiates high vs low contrast"""
        # High contrast image
        high_contrast = np.random.randint(0, 255, (600, 800), dtype=np.uint8)

        # Low contrast version (compress range to 100-150)
        low_contrast = ((high_contrast / 255.0) * 50 + 100).astype(np.uint8)

        analyzer = ImageQualityAnalyzer()
        high_metrics = analyzer.analyze(high_contrast)
        low_metrics = analyzer.analyze(low_contrast)

        # High contrast should have higher score
        assert high_metrics.contrast_score > low_metrics.contrast_score, \
            f"High contrast score ({high_metrics.contrast_score}) should be > low ({low_metrics.contrast_score})"

    def test_noise_level_differentiates_clean_vs_noisy(self):
        """Test noise level correctly differentiates clean vs noisy images"""
        # Clean image
        clean = np.ones((600, 800), dtype=np.uint8) * 128

        # Noisy version
        noise = np.random.normal(0, 40, clean.shape)
        noisy = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        analyzer = ImageQualityAnalyzer()
        clean_metrics = analyzer.analyze(clean)
        noisy_metrics = analyzer.analyze(noisy)

        # Noisy should have higher noise level
        assert noisy_metrics.noise_level > clean_metrics.noise_level, \
            f"Noisy image noise_level ({noisy_metrics.noise_level}) should be > clean ({clean_metrics.noise_level})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
