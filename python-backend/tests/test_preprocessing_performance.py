"""
Performance Benchmark Tests for Intelligent Preprocessing

Measures processing time and validates performance requirements.
"""

import pytest
import numpy as np
import time

from services.ocr.intelligent_preprocessing import IntelligentPreprocessor
from services.ocr import advanced_preprocessing
from services.ocr.image_quality_analyzer import ImageQualityAnalyzer
from services.ocr.preprocessing_strategy import PreprocessingStrategySelector
from services.ocr.preprocessing_validator import PreprocessingValidator
from .test_fixtures_preprocessing import create_synthetic_image


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    def test_quality_analysis_performance(self):
        """Test quality analysis completes within time limit"""
        # Create test image
        image = create_synthetic_image()

        analyzer = ImageQualityAnalyzer()

        # Measure analysis time
        start = time.time()
        metrics = analyzer.analyze(image)
        elapsed = (time.time() - start) * 1000  # Convert to ms

        # Should complete within 50ms
        assert elapsed < 50.0, f"Quality analysis took {elapsed:.1f}ms (expected <50ms)"

        print(f"\nQuality analysis: {elapsed:.2f}ms")

    def test_strategy_selection_performance(self):
        """Test strategy selection completes within time limit"""
        # Analyze image first
        image = create_synthetic_image(noise_level=30, blur_kernel=7, contrast_factor=0.4)
        analyzer = ImageQualityAnalyzer()
        metrics = analyzer.analyze(image)

        selector = PreprocessingStrategySelector()

        # Measure selection time
        start = time.time()
        strategy = selector.select_strategy(metrics)
        elapsed = (time.time() - start) * 1000

        # Should complete within 5ms
        assert elapsed < 5.0, f"Strategy selection took {elapsed:.1f}ms (expected <5ms)"

        print(f"Strategy selection: {elapsed:.2f}ms")

    def test_validation_performance(self):
        """Test validation completes within time limit"""
        image = create_synthetic_image()
        preprocessed = image.copy()

        validator = PreprocessingValidator()

        # Measure validation time
        start = time.time()
        result = validator.validate(image, preprocessed)
        elapsed = (time.time() - start) * 1000

        # Should complete within 100ms
        assert elapsed < 100.0, f"Validation took {elapsed:.1f}ms (expected <100ms)"

        print(f"Validation (with SSIM): {elapsed:.2f}ms")

    def test_richardson_lucy_performance(self):
        """Test Richardson-Lucy deblurring performance"""
        image = create_synthetic_image(blur_kernel=7)

        # Measure deblurring time
        start = time.time()
        deblurred = advanced_preprocessing.richardson_lucy_deblur(image, iterations=10)
        elapsed = (time.time() - start) * 1000

        # Should complete within 500ms for 10 iterations
        assert elapsed < 500.0, f"Richardson-Lucy took {elapsed:.1f}ms (expected <500ms)"

        print(f"Richardson-Lucy deblur (10 iter): {elapsed:.2f}ms")

    def test_adaptive_binarization_performance(self):
        """Test adaptive binarization performance"""
        image = create_synthetic_image(contrast_factor=0.3)

        # Measure Sauvola binarization time
        start = time.time()
        binary = advanced_preprocessing.adaptive_binarize_sauvola(image)
        elapsed = (time.time() - start) * 1000

        # Should complete within 200ms
        assert elapsed < 200.0, f"Sauvola binarization took {elapsed:.1f}ms (expected <200ms)"

        print(f"Sauvola binarization: {elapsed:.2f}ms")

    def test_full_pipeline_performance(self):
        """Test complete preprocessing pipeline performance"""
        # Create low quality image requiring multiple techniques
        image = create_synthetic_image(
            noise_level=30,
            blur_kernel=5,
            contrast_factor=0.4
        )

        preprocessor = IntelligentPreprocessor()

        # Measure full pipeline time
        start = time.time()
        result = preprocessor.process(image)
        elapsed = (time.time() - start) * 1000

        # Should complete within 300ms total
        assert elapsed < 300.0, f"Full pipeline took {elapsed:.1f}ms (expected <300ms)"

        print(f"Full pipeline: {elapsed:.2f}ms")
        print(f"  - Techniques applied: {len(result.techniques_applied)}")
        print(f"  - Used preprocessed: {result.used_preprocessed}")

    def test_batch_processing_performance(self):
        """Test batch processing performance"""
        # Create batch of 10 images
        images = [
            create_synthetic_image(
                noise_level=np.random.uniform(10, 40),
                blur_kernel=int(np.random.choice([0, 1, 3, 5, 7])),
                contrast_factor=np.random.uniform(0.3, 1.2)
            )
            for _ in range(10)
        ]

        preprocessor = IntelligentPreprocessor()

        # Measure batch processing time
        start = time.time()
        results = [preprocessor.process(img) for img in images]
        elapsed = (time.time() - start) * 1000

        # Average time per image
        avg_time = elapsed / len(images)

        print(f"Batch processing (10 images): {elapsed:.2f}ms total, {avg_time:.2f}ms avg")

        # Should average <300ms per image
        assert avg_time < 300.0, f"Batch avg time {avg_time:.1f}ms (expected <300ms)"


class TestMemoryUsage:
    """Test memory usage and efficiency"""

    def test_preprocessing_memory_efficiency(self):
        """Test preprocessing doesn't create excessive memory overhead"""
        import sys

        # Create large image
        large_image = create_synthetic_image(width=2400, height=1800)  # ~4MP

        # Get baseline memory
        image_size = sys.getsizeof(large_image)

        preprocessor = IntelligentPreprocessor()

        # Process image
        result = preprocessor.process(large_image)

        # Result should not be excessively larger than original
        result_size = sys.getsizeof(result.image)

        # Should be roughly same size (within 2x)
        assert result_size <= image_size * 2, \
            f"Result size {result_size} bytes exceeds 2x original {image_size} bytes"

    def test_batch_memory_cleanup(self):
        """Test batch processing cleans up memory"""
        # Process multiple images sequentially
        for i in range(5):
            image = create_synthetic_image()
            preprocessor = IntelligentPreprocessor()
            result = preprocessor.process(image)

            # Explicitly delete to free memory
            del image
            del result

        # If we get here without OOM, memory cleanup is working


class TestScalability:
    """Test scalability with different image sizes"""

    @pytest.mark.parametrize("size", [
        (400, 300),   # Small
        (800, 600),   # Medium
        (1600, 1200), # Large
        (2400, 1800)  # Very large
    ])
    def test_performance_scales_linearly(self, size):
        """Test performance scales reasonably with image size"""
        width, height = size
        image = create_synthetic_image(width=width, height=height)

        preprocessor = IntelligentPreprocessor()

        start = time.time()
        result = preprocessor.process(image)
        elapsed = (time.time() - start) * 1000

        # Calculate pixels
        pixels = width * height
        megapixels = pixels / 1_000_000

        # Time per megapixel
        time_per_mp = elapsed / megapixels if megapixels > 0 else elapsed

        print(f"\n{width}x{height} ({megapixels:.1f}MP): {elapsed:.2f}ms ({time_per_mp:.2f}ms/MP)")

        # Should not exceed 200ms per megapixel
        assert time_per_mp < 200.0, \
            f"Time per MP {time_per_mp:.1f}ms exceeds 200ms/MP for {width}x{height}"


class TestThroughput:
    """Test processing throughput"""

    def test_pages_per_second_throughput(self):
        """Test pages per second processing rate"""
        # Simulate typical document pages
        num_pages = 20
        images = [
            create_synthetic_image(
                width=800,
                height=1100,  # Letter size aspect ratio
                noise_level=np.random.uniform(15, 30),
                blur_kernel=int(np.random.choice([0, 1, 3, 5])),
                contrast_factor=np.random.uniform(0.5, 1.0)
            )
            for _ in range(num_pages)
        ]

        preprocessor = IntelligentPreprocessor()

        start = time.time()
        results = [preprocessor.process(img) for img in images]
        elapsed = time.time() - start

        # Calculate throughput
        pages_per_second = num_pages / elapsed

        print(f"\nProcessed {num_pages} pages in {elapsed:.2f}s")
        print(f"Throughput: {pages_per_second:.2f} pages/second")

        # Should process at least 3 pages per second
        assert pages_per_second >= 3.0, \
            f"Throughput {pages_per_second:.2f} pages/s below minimum 3.0 pages/s"


class TestWorstCaseScenarios:
    """Test worst-case performance scenarios"""

    def test_worst_case_all_techniques(self):
        """Test worst case: image needing all techniques"""
        # Create very poor quality image
        image = create_synthetic_image(
            noise_level=40,
            blur_kernel=9,
            contrast_factor=0.2,
            brightness=110
        )

        preprocessor = IntelligentPreprocessor(allow_destructive=True)

        start = time.time()
        result = preprocessor.process(image)
        elapsed = (time.time() - start) * 1000

        print(f"\nWorst case (all techniques): {elapsed:.2f}ms")
        print(f"  - Techniques: {result.techniques_applied}")

        # Even worst case should complete within 500ms
        assert elapsed < 500.0, f"Worst case took {elapsed:.1f}ms (expected <500ms)"

    def test_worst_case_validation_rejection(self):
        """Test worst case: preprocessing rejected by validation"""
        # Create perfect image (preprocessing will be rejected)
        image = create_synthetic_image(
            noise_level=2,
            blur_kernel=0,
            contrast_factor=1.5,
            brightness=200
        )

        preprocessor = IntelligentPreprocessor(enable_validation=True)

        start = time.time()
        result = preprocessor.process(image)
        elapsed = (time.time() - start) * 1000

        # Should still be fast even with validation
        assert elapsed < 100.0, f"Validation overhead {elapsed:.1f}ms too high"


class TestComparisonBenchmarks:
    """Compare preprocessing vs no preprocessing"""

    def test_preprocessing_overhead_acceptable(self):
        """Test preprocessing adds acceptable overhead"""
        image = create_synthetic_image(blur_kernel=5, noise_level=25)

        # Time without preprocessing (baseline)
        start = time.time()
        _ = image.copy()  # Simulate minimal processing
        baseline = (time.time() - start) * 1000

        # Time with preprocessing
        preprocessor = IntelligentPreprocessor()
        start = time.time()
        result = preprocessor.process(image)
        with_preprocessing = (time.time() - start) * 1000

        overhead = with_preprocessing - baseline

        print(f"\nBaseline: {baseline:.2f}ms")
        print(f"With preprocessing: {with_preprocessing:.2f}ms")
        print(f"Overhead: {overhead:.2f}ms")

        # Overhead should be reasonable (< 300ms)
        assert overhead < 300.0, f"Preprocessing overhead {overhead:.1f}ms too high"


if __name__ == "__main__":
    # Run with verbose output to see timing details
    pytest.main([__file__, "-v", "-s"])
