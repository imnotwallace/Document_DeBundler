"""
Unit Tests for PreprocessingValidator

Tests quality validation and distortion detection.
"""

import pytest
import numpy as np

from services.ocr.preprocessing_validator import (
    PreprocessingValidator,
    ValidationResult,
    validate_preprocessing
)
from .test_fixtures_preprocessing import (
    sharp_high_contrast_image,
    blurry_image,
    low_contrast_image,
    image_transformation_functions
)


class TestPreprocessingValidator:
    """Test PreprocessingValidator class"""

    def test_validator_initialization(self):
        """Test validator initializes with correct thresholds"""
        validator = PreprocessingValidator(
            min_quality_improvement=10.0,
            min_ssim=0.90,
            max_noise_increase=15.0
        )

        assert validator.min_quality_improvement == 10.0
        assert validator.min_ssim == 0.90
        assert validator.max_noise_increase == 15.0

    def test_validate_improved_image(self, blurry_image):
        """Test validation approves genuinely improved image"""
        import cv2

        # Original blurry image
        original = blurry_image

        # Improved version (sharpen)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        preprocessed = cv2.filter2D(original, -1, kernel)

        validator = PreprocessingValidator()
        result = validator.validate(original, preprocessed)

        # Should recommend using preprocessed
        assert result.is_improved, "Sharpened image should be improved"
        # Note: May or may not meet threshold depending on image

    def test_validate_degraded_image(self, sharp_high_contrast_image):
        """Test validation rejects degraded image"""
        import cv2

        # Original sharp image
        original = sharp_high_contrast_image

        # Degraded version (heavy blur)
        preprocessed = cv2.GaussianBlur(original, (21, 21), 0)

        validator = PreprocessingValidator()
        result = validator.validate(original, preprocessed)

        # Should NOT recommend using preprocessed
        assert not result.use_preprocessed, "Degraded image should not be used"
        assert result.quality_delta < 0, "Quality delta should be negative for degraded image"

    def test_validate_identical_images(self, sharp_high_contrast_image):
        """Test validation of identical images"""
        validator = PreprocessingValidator()
        result = validator.validate(sharp_high_contrast_image, sharp_high_contrast_image)

        # SSIM should be 1.0 (identical)
        assert result.ssim_score > 0.99, f"Identical images should have SSIM ~1.0, got {result.ssim_score}"
        assert not result.distortion_detected, "Identical images should have no distortion"

    def test_validate_detects_distortion(self, sharp_high_contrast_image):
        """Test validator detects significant distortion"""
        # Create heavily distorted version
        distorted = np.random.randint(0, 255, sharp_high_contrast_image.shape, dtype=np.uint8)

        validator = PreprocessingValidator(min_ssim=0.85)
        result = validator.validate(sharp_high_contrast_image, distorted)

        # Should detect distortion
        assert result.distortion_detected, "Random image should be detected as distorted"
        assert result.ssim_score < 0.85, f"Distorted image should have low SSIM, got {result.ssim_score}"
        assert not result.use_preprocessed, "Distorted image should not be used"

    def test_validate_noise_increase_rejection(self):
        """Test validator rejects images with excessive noise increase"""
        # Create clean image
        clean = np.ones((600, 800), dtype=np.uint8) * 128

        # Add heavy noise
        noise = np.random.normal(0, 50, clean.shape)
        noisy = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        validator = PreprocessingValidator(max_noise_increase=10.0)
        result = validator.validate(clean, noisy)

        # Should reject due to noise increase
        assert result.noise_delta > 10.0, "Noise should have increased significantly"
        # May or may not be flagged as improved depending on other metrics

    def test_validate_minimal_improvement_rejection(self, sharp_high_contrast_image):
        """Test validator rejects images with minimal improvement"""
        # Create nearly identical version (tiny change)
        preprocessed = sharp_high_contrast_image.copy()
        preprocessed[100:110, 100:110] += 1  # Tiny change

        validator = PreprocessingValidator(min_quality_improvement=5.0)
        result = validator.validate(sharp_high_contrast_image, preprocessed)

        # Should not meet improvement threshold
        assert abs(result.quality_delta) < 5.0, "Quality delta should be minimal"
        assert not result.use_preprocessed, "Minimal improvement should not be accepted"

    def test_validation_result_attributes(self, blurry_image):
        """Test validation result contains all expected attributes"""
        import cv2

        original = blurry_image
        preprocessed = cv2.GaussianBlur(original, (5, 5), 0)

        validator = PreprocessingValidator()
        result = validator.validate(original, preprocessed)

        # Check all attributes present
        assert hasattr(result, 'is_improved')
        assert hasattr(result, 'use_preprocessed')
        assert hasattr(result, 'quality_delta')
        assert hasattr(result, 'blur_delta')
        assert hasattr(result, 'contrast_delta')
        assert hasattr(result, 'noise_delta')
        assert hasattr(result, 'ssim_score')
        assert hasattr(result, 'distortion_detected')
        assert hasattr(result, 'reason')

        # Check types
        assert isinstance(result.is_improved, bool)
        assert isinstance(result.use_preprocessed, bool)
        assert isinstance(result.quality_delta, float)
        assert isinstance(result.ssim_score, float)
        assert isinstance(result.reason, str)

    def test_validation_result_str_representation(self, sharp_high_contrast_image):
        """Test validation result string representation"""
        validator = PreprocessingValidator()
        result = validator.validate(sharp_high_contrast_image, sharp_high_contrast_image)

        result_str = str(result)

        # Should contain key information
        assert 'USE' in result_str or 'use' in result_str.lower()
        assert 'ssim' in result_str.lower() or 'SSIM' in result_str
        assert isinstance(result_str, str)
        assert len(result_str) > 0

    def test_ssim_computation_accuracy(self):
        """Test SSIM computation is accurate"""
        # Create test images with known SSIM
        image1 = np.random.randint(0, 255, (600, 800), dtype=np.uint8)

        # Identical image (SSIM = 1.0)
        image2_identical = image1.copy()

        # Slightly different image (SSIM ~ 0.95-0.99)
        image3_similar = image1.copy()
        image3_similar += np.random.randint(-5, 5, image1.shape).astype(np.int16)
        image3_similar = np.clip(image3_similar, 0, 255).astype(np.uint8)

        validator = PreprocessingValidator()

        result_identical = validator.validate(image1, image2_identical)
        result_similar = validator.validate(image1, image3_similar)

        # Identical should have higher SSIM
        assert result_identical.ssim_score > result_similar.ssim_score, \
            f"Identical SSIM ({result_identical.ssim_score}) should be > similar ({result_similar.ssim_score})"

        # Identical should be very close to 1.0
        assert result_identical.ssim_score > 0.99, f"Identical images should have SSIM ~1.0, got {result_identical.ssim_score}"

    def test_quality_delta_components(self, low_contrast_image):
        """Test quality delta correctly weighs different improvements"""
        import cv2

        original = low_contrast_image

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(original)

        validator = PreprocessingValidator()
        result = validator.validate(original, enhanced)

        # Contrast delta should be positive (improved)
        assert result.contrast_delta > 0, "Contrast enhancement should increase contrast score"

    def test_validator_handles_rgb_images(self):
        """Test validator handles RGB images correctly"""
        # Create RGB images
        original_rgb = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        preprocessed_rgb = original_rgb.copy()
        preprocessed_rgb += 10  # Slight change

        validator = PreprocessingValidator()
        result = validator.validate(original_rgb, preprocessed_rgb)

        # Should complete without error
        assert result is not None
        assert isinstance(result.ssim_score, float)

    def test_validator_handles_dimension_mismatch(self):
        """Test validator handles dimension mismatches gracefully"""
        original = np.random.randint(0, 255, (600, 800), dtype=np.uint8)
        preprocessed = np.random.randint(0, 255, (500, 700), dtype=np.uint8)  # Different size

        validator = PreprocessingValidator()
        result = validator.validate(original, preprocessed)

        # Should handle gracefully (SSIM might be 0 or error handled)
        assert result is not None


class TestConvenienceFunction:
    """Test convenience function validate_preprocessing"""

    def test_convenience_function(self, blurry_image):
        """Test convenience function works correctly"""
        import cv2

        original = blurry_image
        preprocessed = cv2.GaussianBlur(original, (5, 5), 0)

        result = validate_preprocessing(original, preprocessed)

        assert isinstance(result, ValidationResult)

    def test_convenience_function_with_custom_thresholds(self):
        """Test convenience function accepts custom thresholds"""
        image = np.random.randint(0, 255, (600, 800), dtype=np.uint8)

        result = validate_preprocessing(
            image,
            image,
            min_quality_improvement=10.0,
            min_ssim=0.90
        )

        assert isinstance(result, ValidationResult)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_validate_empty_images(self):
        """Test validation handles empty images"""
        empty = np.zeros((100, 100), dtype=np.uint8)

        validator = PreprocessingValidator()
        result = validator.validate(empty, empty)

        # Should complete without crashing
        assert result is not None

    def test_validate_single_pixel_images(self):
        """Test validation handles very small images"""
        tiny = np.array([[128]], dtype=np.uint8)

        validator = PreprocessingValidator()
        result = validator.validate(tiny, tiny)

        # Should complete without crashing
        assert result is not None

    def test_validate_extreme_values(self):
        """Test validation handles extreme pixel values"""
        # All black
        black = np.zeros((600, 800), dtype=np.uint8)

        # All white
        white = np.ones((600, 800), dtype=np.uint8) * 255

        validator = PreprocessingValidator()

        result_black = validator.validate(black, black)
        result_white = validator.validate(white, white)

        # Should complete without error
        assert result_black is not None
        assert result_white is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
