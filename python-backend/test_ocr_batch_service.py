"""
Test script for OCR Batch Service
Demonstrates usage and validates the implementation
"""

import sys
import logging
from pathlib import Path
import threading
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from services.ocr_batch_service import OCRBatchService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def progress_callback(current, total, message, percent, eta):
    """Progress callback for testing"""
    eta_str = f"{int(eta)}s" if eta > 0 else "calculating..."
    logger.info(
        f"Progress: {percent:.1f}% ({current}/{total}) - {message} - ETA: {eta_str}"
    )


def test_basic_usage():
    """Test basic service initialization and configuration"""
    logger.info("=" * 60)
    logger.info("Test 1: Basic Service Initialization")
    logger.info("=" * 60)

    service = OCRBatchService(
        progress_callback=progress_callback,
        use_gpu=True
    )

    logger.info(f"Batch size: {service.batch_size}")
    logger.info(f"GPU available: {service.capabilities['gpu_available']}")
    logger.info(f"VRAM: {service.capabilities['gpu_memory_gb']:.1f}GB")
    logger.info(f"System RAM: {service.capabilities['system_memory_gb']:.1f}GB")

    service.cleanup()
    logger.info("Test 1: PASSED\n")


def test_cancellation():
    """Test cancellation handling"""
    logger.info("=" * 60)
    logger.info("Test 2: Cancellation Handling")
    logger.info("=" * 60)

    cancellation_flag = threading.Event()

    service = OCRBatchService(
        progress_callback=progress_callback,
        cancellation_flag=cancellation_flag,
        use_gpu=False
    )

    # Test flag behavior
    assert service._is_cancelled() == False
    cancellation_flag.set()
    assert service._is_cancelled() == True

    service.cleanup()
    logger.info("Test 2: PASSED\n")


def test_retry_logic():
    """Test retry with backoff"""
    logger.info("=" * 60)
    logger.info("Test 3: Retry Logic")
    logger.info("=" * 60)

    service = OCRBatchService(use_gpu=False)

    # Test successful operation
    call_count = [0]
    def success_operation():
        call_count[0] += 1
        return "success"

    result = service._retry_with_backoff(success_operation, "Test operation")
    assert result == "success"
    assert call_count[0] == 1
    logger.info("Successful operation: OK")

    service.cleanup()
    logger.info("Test 3: PASSED\n")


def main():
    """Run all tests"""
    logger.info("\nOCR Batch Service Test Suite\n")

    try:
        test_basic_usage()
        test_cancellation()
        test_retry_logic()
        logger.info("ALL TESTS PASSED")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
