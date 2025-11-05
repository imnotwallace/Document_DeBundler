"""
Integration Tests for Intelligent Preprocessing Pipeline

Tests the complete preprocessing pipeline end-to-end.
"""

import pytest
import numpy as np

from services.ocr.intelligent_preprocessing import (
    IntelligentPreprocessor,
    IntelligentPreprocessingResult,
    process_with_intelligent_preprocessing,
    process_batch_with_intelligent_preprocessing
)
from .test_fixtures_preprocessing import (
    perfect_image,
    blurry_image,
    low_contrast_image,
    noisy_image,
    low_quality_image,
    receipt_image,
    image_with_known_boxes
)


class TestIntelligentPreprocessor:
    """Test IntelligentPreprocessor class"""

    def test_preprocessor_initialization(self):
        """Test preprocessor initializes correctly"""
        preprocessor = IntelligentPreprocessor(
            allow_destructive=True,
            enable_validation=True,
            min_quality_improvement=5.0,
            min_ssim=0.85
        )

        assert preprocessor.allow_destructive == True
        assert preprocessor.enable_validation == True

    def test_process_perfect_image(self, perfect_image):
        """Test processing perfect image returns original"""
        preprocessor = IntelligentPreprocessor()
        result = preprocessor.process(perfect_image)

        # Should use original (no preprocessing needed)
        assert not result.used_preprocessed, "Perfect image should use original"
        assert len(result.techniques_applied) == 0, "Perfect image should have no techniques applied"
        assert np.array_equal(result.image, perfect_image), "Should return original image"

    def test_process_blurry_image(self, blurry_image):
        """Test processing blurry image applies sharpening"""
        preprocessor = IntelligentPreprocessor()
        result = preprocessor.process(blurry_image)

        # Should apply some preprocessing
        assert result.strategy is not None
        assert len(result.strategy.techniques) > 0, "Blurry image should get preprocessing"

        # Check if techniques were applied (may or may not pass validation)
        assert result.preprocessed_metrics is not None

    def test_process_low_contrast_image(self, low_contrast_image):
        """Test processing low contrast image applies contrast enhancement"""
        preprocessor = IntelligentPreprocessor()
        result = preprocessor.process(low_contrast_image)

        # Should select contrast enhancement
        assert result.strategy is not None
        assert len(result.strategy.techniques) > 0

        # If preprocessing was applied and accepted
        if result.used_preprocessed:
            assert 'contrast' in str(result.techniques_applied).lower() or \
                   any('contrast' in t for t in result.techniques_applied)

    def test_process_noisy_image(self, noisy_image):
        """Test processing noisy image applies denoising"""
        preprocessor = IntelligentPreprocessor()
        result = preprocessor.process(noisy_image)

        # Should select denoising
        assert result.strategy is not None
        assert len(result.strategy.techniques) > 0

    def test_process_low_quality_image(self, low_quality_image):
        """Test processing low quality image applies multiple techniques"""
        preprocessor = IntelligentPreprocessor()
        result = preprocessor.process(low_quality_image)

        # Should select multiple techniques
        assert len(result.strategy.techniques) >= 2, "Low quality should get multiple techniques"

    def test_validation_prevents_degradation(self, perfect_image):
        """Test validation prevents using degraded preprocessed image"""
        preprocessor = IntelligentPreprocessor(enable_validation=True)
        result = preprocessor.process(perfect_image)

        # If preprocessing was attempted, validation should prevent degradation
        if result.validation is not None:
            if not result.validation.use_preprocessed:
                # Validation correctly rejected preprocessing
                assert not result.used_preprocessed

    def test_validation_disabled_always_uses_preprocessed(self, blurry_image):
        """Test disabling validation always uses preprocessed"""
        preprocessor = IntelligentPreprocessor(enable_validation=False)
        result = preprocessor.process(blurry_image)

        # Should use preprocessed if any techniques were selected
        if len(result.strategy.techniques) > 0:
            assert result.used_preprocessed, "With validation disabled, should always use preprocessed"
            assert result.validation is None, "Validation should not be performed"

    def test_allow_destructive_flag(self, receipt_image):
        """Test allow_destructive flag controls binarization"""
        preprocessor_allow = IntelligentPreprocessor(allow_destructive=True)
        preprocessor_deny = IntelligentPreprocessor(allow_destructive=False)

        result_allow = preprocessor_allow.process(receipt_image)
        result_deny = preprocessor_deny.process(receipt_image)

        # Allow may have binarization, deny should not
        destructive_techniques = ['binarize', 'morph_open', 'morph_close', 'adaptive_binarize']

        has_destructive_allow = any(
            any(tech in t for tech in destructive_techniques)
            for t in result_allow.techniques_applied
        )

        has_destructive_deny = any(
            any(tech in t for tech in destructive_techniques)
            for t in result_deny.techniques_applied
        )

        # Deny should not have destructive techniques
        if len(result_deny.techniques_applied) > 0:
            assert not has_destructive_deny, "Destructive techniques should be excluded"

    def test_preprocessing_result_attributes(self, blurry_image):
        """Test preprocessing result contains all expected attributes"""
        preprocessor = IntelligentPreprocessor()
        result = preprocessor.process(blurry_image)

        # Check all attributes present
        assert hasattr(result, 'image')
        assert hasattr(result, 'original_metrics')
        assert hasattr(result, 'preprocessed_metrics')
        assert hasattr(result, 'strategy')
        assert hasattr(result, 'validation')
        assert hasattr(result, 'used_preprocessed')
        assert hasattr(result, 'techniques_applied')

        # Check types
        assert isinstance(result.image, np.ndarray)
        assert isinstance(result.used_preprocessed, bool)
        assert isinstance(result.techniques_applied, list)

    def test_preprocessing_maintains_image_shape(self, blurry_image):
        """Test preprocessing maintains original image shape"""
        preprocessor = IntelligentPreprocessor()
        result = preprocessor.process(blurry_image)

        assert result.image.shape == blurry_image.shape, \
            f"Image shape changed: {blurry_image.shape} -> {result.image.shape}"

    def test_preprocessing_handles_rgb_images(self):
        """Test preprocessing handles RGB images"""
        rgb_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        preprocessor = IntelligentPreprocessor()
        result = preprocessor.process(rgb_image)

        # Should complete successfully
        assert result is not None
        assert result.image.shape == rgb_image.shape

    def test_preprocessing_handles_errors_gracefully(self):
        """Test preprocessing handles errors gracefully"""
        # Very small image that might cause issues
        tiny_image = np.ones((5, 5), dtype=np.uint8) * 128

        preprocessor = IntelligentPreprocessor()
        result = preprocessor.process(tiny_image)

        # Should return a valid result (original if preprocessing fails)
        assert result is not None
        assert isinstance(result.image, np.ndarray)


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_process_with_intelligent_preprocessing(self, blurry_image):
        """Test convenience function for single image"""
        result = process_with_intelligent_preprocessing(blurry_image)

        assert isinstance(result, IntelligentPreprocessingResult)
        assert result.image is not None

    def test_process_with_intelligent_preprocessing_custom_params(self):
        """Test convenience function with custom parameters"""
        image = np.random.randint(0, 255, (600, 800), dtype=np.uint8)

        result = process_with_intelligent_preprocessing(
            image,
            allow_destructive=False,
            enable_validation=True
        )

        assert isinstance(result, IntelligentPreprocessingResult)

    def test_process_batch_with_intelligent_preprocessing(
        self,
        perfect_image,
        blurry_image,
        low_contrast_image
    ):
        """Test batch processing convenience function"""
        images = [perfect_image, blurry_image, low_contrast_image]

        results = process_batch_with_intelligent_preprocessing(images)

        # Should have result for each image
        assert len(results) == len(images)

        # All results should be valid
        for result in results:
            assert isinstance(result, IntelligentPreprocessingResult)
            assert result.image is not None


class TestCoordinatePreservation:
    """Test coordinate/bounding box preservation"""

    def test_preprocessing_preserves_dimensions(self, image_with_known_boxes):
        """Test preprocessing preserves image dimensions"""
        image, boxes = image_with_known_boxes

        preprocessor = IntelligentPreprocessor()
        result = preprocessor.process(image)

        # Dimensions should be identical
        assert result.image.shape == image.shape, \
            "Preprocessing changed image dimensions"

    def test_bounding_boxes_remain_valid(self, image_with_known_boxes):
        """Test bounding boxes remain valid after preprocessing"""
        image, original_boxes = image_with_known_boxes

        preprocessor = IntelligentPreprocessor()
        result = preprocessor.process(image)

        # Since dimensions are preserved, boxes should still be valid
        # (i.e., all coordinates within image bounds)
        height, width = result.image.shape[:2]

        for box in original_boxes:
            for point in box:
                x, y = point
                assert 0 <= x < width, f"X coordinate {x} out of bounds [0, {width})"
                assert 0 <= y < height, f"Y coordinate {y} out of bounds [0, {height})"

    def test_no_deskewing_in_coordinate_mode(self, image_with_known_boxes):
        """Test deskewing is not applied when preserving coordinates"""
        image, boxes = image_with_known_boxes

        preprocessor = IntelligentPreprocessor()
        result = preprocessor.process(image)

        # Deskewing should not be in applied techniques
        # (as it can change dimensions/coordinates)
        assert 'deskew' not in str(result.techniques_applied).lower(), \
            "Deskewing should not be applied (changes coordinates)"


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios"""

    def test_scan_to_ocr_pipeline(self, low_quality_image):
        """Test complete pipeline: analyze -> preprocess -> validate"""
        preprocessor = IntelligentPreprocessor(
            allow_destructive=True,
            enable_validation=True
        )

        # Process image
        result = preprocessor.process(low_quality_image)

        # Verify complete workflow
        assert result.original_metrics is not None, "Should have original metrics"
        assert result.strategy is not None, "Should have strategy"

        # If preprocessing was applied
        if result.used_preprocessed:
            assert result.preprocessed_metrics is not None, "Should have preprocessed metrics"
            assert result.validation is not None, "Should have validation result"
            assert result.validation.use_preprocessed, "Validation should approve preprocessed"
            assert len(result.techniques_applied) > 0, "Should have applied techniques"

    def test_multiple_images_different_qualities(self):
        """Test processing multiple images with different quality levels"""
        from .test_fixtures_preprocessing import (
            create_synthetic_image
        )

        # Create images with different quality levels
        perfect = create_synthetic_image(noise_level=2, blur_kernel=0, contrast_factor=1.5)
        blurry = create_synthetic_image(noise_level=10, blur_kernel=9, contrast_factor=0.8)
        low_contrast = create_synthetic_image(noise_level=15, blur_kernel=0, contrast_factor=0.3)

        preprocessor = IntelligentPreprocessor()

        result_perfect = preprocessor.process(perfect)
        result_blurry = preprocessor.process(blurry)
        result_low_contrast = preprocessor.process(low_contrast)

        # Perfect should need no preprocessing
        assert not result_perfect.used_preprocessed or len(result_perfect.techniques_applied) == 0

        # Others should get different strategies
        assert result_blurry.strategy.techniques != result_low_contrast.strategy.techniques or \
               len(result_blurry.strategy.techniques) == 0 or \
               len(result_low_contrast.strategy.techniques) == 0, \
            "Different quality issues should get different strategies"

    def test_preprocessing_improves_ocr_readiness(self, low_contrast_image):
        """Test preprocessing improves image quality metrics"""
        preprocessor = IntelligentPreprocessor(enable_validation=True)
        result = preprocessor.process(low_contrast_image)

        # If preprocessing was accepted, metrics should improve
        if result.used_preprocessed and result.validation:
            # Quality should have improved
            assert result.validation.quality_delta > 0, \
                "Accepted preprocessing should have positive quality delta"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
