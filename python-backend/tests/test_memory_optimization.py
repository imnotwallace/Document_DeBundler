"""
Test script for memory optimization features
Tests VRAM monitoring, adaptive batch sizing, and configuration generation
"""

import sys
import logging
from pathlib import Path
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ocr.config import (
    detect_hardware_capabilities,
    get_optimal_batch_size,
    get_adaptive_dpi,
    get_default_config
)
from services.ocr.vram_monitor import VRAMMonitor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def capabilities():
    """Fixture that provides hardware capabilities for all tests"""
    return detect_hardware_capabilities()


def test_hardware_detection(capabilities):
    """Test hardware capability detection"""
    logger.info("=" * 60)
    logger.info("Testing Hardware Detection")
    logger.info("=" * 60)

    logger.info(f"GPU Available: {capabilities['gpu_available']}")
    logger.info(f"CUDA Available: {capabilities['cuda_available']}")
    logger.info(f"DirectML Available: {capabilities['directml_available']}")
    logger.info(f"GPU Memory: {capabilities['gpu_memory_gb']:.2f} GB")
    logger.info(f"System Memory: {capabilities['system_memory_gb']:.2f} GB")
    logger.info(f"CPU Count: {capabilities['cpu_count']}")
    logger.info(f"Platform: {capabilities['platform']}")

    # Add assertions to validate the fixture data
    assert 'gpu_available' in capabilities
    assert 'system_memory_gb' in capabilities
    assert capabilities['system_memory_gb'] > 0
    assert capabilities['cpu_count'] > 0


def test_batch_size_calculation(capabilities):
    """Test batch size calculation for different scenarios"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Batch Size Calculation")
    logger.info("=" * 60)

    # Test GPU scenarios
    gpu_configs = [
        (1.5, "1.5GB VRAM (ultra low-end)"),
        (2.0, "2GB VRAM (low-end)"),
        (3.0, "3GB VRAM (entry-level)"),
        (4.0, "4GB VRAM (TARGET HARDWARE)"),
        (6.0, "6GB VRAM (mid-range)"),
        (8.0, "8GB VRAM (high-end)"),
        (12.0, "12GB VRAM (enthusiast)"),
    ]

    logger.info("\nGPU Batch Sizes:")
    for vram_gb, description in gpu_configs:
        batch_size = get_optimal_batch_size(
            use_gpu=True,
            gpu_memory_gb=vram_gb,
            system_memory_gb=capabilities['system_memory_gb']
        )
        logger.info(f"  {description}: batch_size = {batch_size}")

        # Add assertions
        assert batch_size > 0, f"Batch size must be positive, got {batch_size} for {description}"
        assert batch_size <= 100, f"Batch size too large: {batch_size} for {description}"

    # Verify target hardware gets batch_size = 25
    target_batch = get_optimal_batch_size(use_gpu=True, gpu_memory_gb=4.0, system_memory_gb=16.0)
    assert target_batch == 25, f"Expected batch size 25 for 4GB VRAM, got {target_batch}"

    # Test CPU scenarios
    cpu_configs = [
        (4, "4GB RAM (minimal)"),
        (8, "8GB RAM (basic)"),
        (16, "16GB RAM (recommended)"),
        (32, "32GB RAM (optimal)"),
    ]

    logger.info("\nCPU Batch Sizes:")
    for ram_gb, description in cpu_configs:
        batch_size = get_optimal_batch_size(
            use_gpu=False,
            gpu_memory_gb=0,
            system_memory_gb=ram_gb
        )
        logger.info(f"  {description}: batch_size = {batch_size}")

        # Add assertions
        assert batch_size > 0, f"Batch size must be positive, got {batch_size} for {description}"
        assert batch_size <= 100, f"Batch size too large: {batch_size} for {description}"


def test_adaptive_dpi(capabilities):
    """Test adaptive DPI calculation"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Adaptive DPI Calculation")
    logger.info("=" * 60)

    quality_levels = ["low", "balanced", "high", "max"]

    # Test with current hardware
    logger.info(f"\nCurrent Hardware ({capabilities['gpu_memory_gb']:.1f}GB VRAM, "
                f"{capabilities['system_memory_gb']:.1f}GB RAM):")

    for quality in quality_levels:
        dpi = get_adaptive_dpi(
            use_gpu=capabilities['gpu_available'],
            gpu_memory_gb=capabilities['gpu_memory_gb'],
            system_memory_gb=capabilities['system_memory_gb'],
            target_quality=quality
        )
        logger.info(f"  {quality}: {dpi} DPI")

        # Add assertions
        assert 100 <= dpi <= 1200, f"DPI out of reasonable range: {dpi} for {quality}"

    # Test 4GB VRAM scenario
    logger.info("\nTarget Hardware (4GB VRAM, 16GB RAM):")
    for quality in quality_levels:
        dpi = get_adaptive_dpi(
            use_gpu=True,
            gpu_memory_gb=4.0,
            system_memory_gb=16.0,
            target_quality=quality
        )
        logger.info(f"  {quality}: {dpi} DPI")

        # Add assertions
        assert 100 <= dpi <= 1200, f"DPI out of reasonable range: {dpi} for {quality}"

    # Verify DPI increases with quality level
    low_dpi = get_adaptive_dpi(True, 4.0, 16.0, "low")
    high_dpi = get_adaptive_dpi(True, 4.0, 16.0, "high")
    assert low_dpi <= high_dpi, f"Low quality DPI ({low_dpi}) should be <= high quality DPI ({high_dpi})"


def test_vram_monitor():
    """Test VRAM monitoring functionality"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing VRAM Monitor")
    logger.info("=" * 60)

    monitor = VRAMMonitor(check_interval=0.1)

    # Get current stats
    info = monitor.get_info()

    if info["available"]:
        logger.info(f"VRAM Available: Yes")
        logger.info(f"Total VRAM: {info['total_gb']} GB")
        logger.info(f"Used VRAM: {info['used_gb']} GB")
        logger.info(f"Free VRAM: {info['free_gb']} GB")
        logger.info(f"Utilization: {info['utilization_percent']}%")
        logger.info(f"Memory Pressure: {info['pressure_level']}")
        logger.info(f"Should Reduce Batch: {info['should_reduce_batch']}")

        # Test batch size suggestions
        current_batch = 25
        suggested_batch = monitor.suggest_batch_size_adjustment(current_batch)
        logger.info(f"\nBatch Size Adjustment:")
        logger.info(f"  Current: {current_batch}")
        logger.info(f"  Suggested: {suggested_batch}")
    else:
        logger.info(f"VRAM monitoring not available: {info['reason']}")
        logger.info("This is expected on CPU-only systems")


def test_default_config(capabilities):
    """Test default configuration generation"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Default Configuration")
    logger.info("=" * 60)

    config = get_default_config(prefer_gpu=True)

    logger.info(f"Engine: {config.engine}")
    logger.info(f"Use GPU: {config.use_gpu}")
    logger.info(f"GPU Batch Size: {config.batch_size}")
    logger.info(f"CPU Batch Size: {config.cpu_batch_size}")
    logger.info(f"Max Memory: {config.max_memory_mb} MB")
    logger.info(f"Threads: {config.num_threads}")
    logger.info(f"Hybrid Mode Enabled: {config.enable_hybrid_mode}")
    logger.info(f"VRAM Monitoring: {config.enable_vram_monitoring}")
    logger.info(f"Adaptive Batch Sizing: {config.enable_adaptive_batch_sizing}")

    # Add assertions
    assert config.engine in ["paddleocr", "tesseract"], f"Unknown engine: {config.engine}"
    assert config.batch_size > 0, f"Batch size must be positive: {config.batch_size}"
    assert config.cpu_batch_size > 0, f"CPU batch size must be positive: {config.cpu_batch_size}"
    assert config.max_memory_mb > 0, f"Max memory must be positive: {config.max_memory_mb}"
    assert config.num_threads > 0, f"Thread count must be positive: {config.num_threads}"

    # Explain hybrid mode decision
    if config.enable_hybrid_mode:
        logger.info("\n✓ Hybrid mode ENABLED:")
        logger.info(f"  - VRAM: {capabilities['gpu_memory_gb']:.1f}GB ≤ 4.5GB")
        logger.info(f"  - RAM: {capabilities['system_memory_gb']:.1f}GB ≥ 15GB")
        logger.info("  - System will automatically offload to CPU under memory pressure")
    else:
        logger.info("\n✗ Hybrid mode DISABLED:")
        if capabilities['gpu_memory_gb'] > 4.5:
            logger.info(f"  - VRAM ({capabilities['gpu_memory_gb']:.1f}GB) is sufficient")
        elif capabilities['system_memory_gb'] < 15:
            logger.info(f"  - RAM ({capabilities['system_memory_gb']:.1f}GB) is insufficient")
        logger.info("  - System will use GPU or CPU exclusively")


def test_memory_estimates():
    """Display memory usage estimates"""
    logger.info("\n" + "=" * 60)
    logger.info("Memory Usage Estimates")
    logger.info("=" * 60)

    # Per-page estimates at different DPIs
    dpis = [150, 200, 300, 400, 600]

    logger.info("\nPer-Page Memory (A4 page):")
    for dpi in dpis:
        # Calculate image size
        width = int(8.3 * dpi)
        height = int(11.7 * dpi)
        raw_mb = (width * height * 3) / (1024 * 1024)
        processing_mb = raw_mb * 3  # Estimate 3x for processing overhead

        logger.info(f"  {dpi} DPI: ~{raw_mb:.1f}MB raw, ~{processing_mb:.1f}MB with overhead")

    # Batch estimates
    logger.info("\nBatch Memory Estimates (300 DPI):")
    raw_per_page = 26  # MB
    processing_per_page = 80  # MB with overhead

    batch_sizes = [5, 10, 15, 20, 25, 30, 50]
    for batch_size in batch_sizes:
        total_mb = batch_size * processing_per_page
        total_gb = total_mb / 1024

        safe_vram = "✓" if total_gb <= 3.5 else "✗"
        logger.info(f"  Batch {batch_size:2d}: ~{total_gb:.2f}GB VRAM {safe_vram}")

    logger.info("\nRecommendation: batch_size=25 uses ~2.5GB, safe for 4GB VRAM")


def main():
    """Run all tests"""
    logger.info("Memory Optimization Test Suite")
    logger.info("Testing optimizations for 4GB VRAM systems\n")

    try:
        # Detect hardware
        capabilities = test_hardware_detection()

        # Run tests
        test_batch_size_calculation(capabilities)
        test_adaptive_dpi(capabilities)
        test_vram_monitor()
        test_default_config(capabilities)
        test_memory_estimates()

        logger.info("\n" + "=" * 60)
        logger.info("All Tests Completed Successfully!")
        logger.info("=" * 60)

        # Summary for 4GB VRAM systems
        if 3.5 < capabilities.get('gpu_memory_gb', 0) < 5.0:
            logger.info("\n🎯 Your system has ~4GB VRAM!")
            logger.info("Optimizations active:")
            logger.info("  ✓ Increased batch size to 25 (2.5x improvement)")
            logger.info("  ✓ Real-time VRAM monitoring enabled")
            logger.info("  ✓ Adaptive batch sizing on memory pressure")

            if capabilities.get('system_memory_gb', 0) >= 15:
                logger.info("  ✓ Hybrid GPU/CPU mode enabled")
                logger.info("  → Automatic CPU offload when needed")
            else:
                logger.info("  ℹ Hybrid mode disabled (need 16GB+ RAM)")
                logger.info("  → Consider upgrading RAM for best performance")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
