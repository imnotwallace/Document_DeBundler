"""
PaddleOCR Engine Implementation
High-performance OCR with GPU support
"""

import logging
import time
import gc
from typing import List, Optional
from pathlib import Path
import numpy as np

from ..base import OCREngine, OCRResult, OCRConfig
from ..vram_monitor import VRAMMonitor

logger = logging.getLogger(__name__)


class PaddleOCREngine(OCREngine):
    """PaddleOCR implementation with GPU auto-detection and adaptive batch sizing"""

    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self.ocr = None
        self._gpu_available = False
        self._vram_monitor: Optional[VRAMMonitor] = None
        self._current_batch_size = config.batch_size
        self._adaptive_enabled = config.enable_adaptive_batch_sizing

    def initialize(self) -> None:
        """Initialize PaddleOCR engine"""
        if self._initialized:
            logger.warning("PaddleOCR already initialized")
            return

        try:
            from paddleocr import PaddleOCR

            # Determine GPU usage
            use_gpu = self.config.use_gpu and self._check_gpu_available()

            if use_gpu:
                logger.info("Initializing PaddleOCR with GPU acceleration")
            else:
                if self.config.use_gpu:
                    logger.warning("GPU requested but not available, using CPU")
                else:
                    logger.info("Initializing PaddleOCR with CPU")

            # Get language(s)
            lang = self.config.languages[0] if self.config.languages else 'en'

            # Initialize PaddleOCR
            ocr_kwargs = {
                'use_angle_cls': self.config.enable_angle_classification,
                'lang': lang,
                'use_gpu': use_gpu,
            }

            # If models are bundled, point to model directory
            if self.config.model_dir and self.config.model_dir.exists():
                logger.info(f"Using bundled models from: {self.config.model_dir}")
                # PaddleOCR will look for models in specified directory
                # Structure: model_dir/{det,rec,cls}_model/
                ocr_kwargs['det_model_dir'] = str(self.config.model_dir / 'det')
                ocr_kwargs['rec_model_dir'] = str(self.config.model_dir / 'rec')
                if self.config.enable_angle_classification:
                    ocr_kwargs['cls_model_dir'] = str(self.config.model_dir / 'cls')

            # Apply engine-specific settings
            ocr_kwargs.update(self.config.engine_settings)

            # Create OCR instance
            self.ocr = PaddleOCR(**ocr_kwargs)
            self._gpu_available = use_gpu
            self._initialized = True

            # Initialize VRAM monitor if GPU is being used and monitoring is enabled
            if use_gpu and self.config.enable_vram_monitoring:
                try:
                    self._vram_monitor = VRAMMonitor(check_interval=0.5)
                    logger.info("VRAM monitoring enabled")
                except Exception as e:
                    logger.warning(f"Failed to initialize VRAM monitor: {e}")
                    self._vram_monitor = None

            logger.info(f"PaddleOCR initialized successfully (GPU: {use_gpu})")

        except ImportError as e:
            logger.error(f"PaddleOCR not installed: {e}")
            raise RuntimeError(
                "PaddleOCR is not installed. "
                "Install with: pip install paddleocr paddlepaddle"
            )
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}", exc_info=True)
            raise

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for PaddlePaddle"""
        try:
            import paddle
            return paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        except:
            return False

    def process_image(self, image: np.ndarray) -> OCRResult:
        """
        Process a single image with PaddleOCR.

        Args:
            image: Image as numpy array (RGB or BGR)

        Returns:
            OCRResult with extracted text
        """
        if not self._initialized:
            raise RuntimeError("OCR engine not initialized")

        start_time = time.time()

        try:
            # Run OCR
            result = self.ocr.ocr(
                image,
                cls=self.config.enable_angle_classification
            )

            # Extract text and confidence
            text_lines = []
            confidences = []
            bboxes = []

            if result and result[0]:
                for line in result[0]:
                    if line:
                        # PaddleOCR result format: [bbox, (text, confidence)]
                        bbox, (text, conf) = line
                        text_lines.append(text)
                        confidences.append(conf)
                        bboxes.append(bbox)

            # Combine text
            full_text = '\n'.join(text_lines)

            # Calculate average confidence
            avg_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

            processing_time = time.time() - start_time

            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                bbox=bboxes if bboxes else None,
                raw_result=result,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"OCR processing failed: {e}", exc_info=True)
            processing_time = time.time() - start_time
            return OCRResult(
                text="",
                confidence=0.0,
                error=str(e),
                processing_time=processing_time
            )

    def process_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """
        Process multiple images in batch with adaptive batch sizing.

        Monitors VRAM usage and adjusts processing strategy to avoid OOM errors.
        Splits large batches if memory pressure is detected.

        Args:
            images: List of images as numpy arrays

        Returns:
            List of OCRResult objects
        """
        if not self._initialized:
            raise RuntimeError("OCR engine not initialized")

        # Check VRAM pressure before processing if monitoring enabled
        if self._vram_monitor and self._adaptive_enabled:
            pressure = self._vram_monitor.get_memory_pressure_level()

            if pressure in ("high", "critical"):
                # Adjust current batch size for future batches
                suggested_size = self._vram_monitor.suggest_batch_size_adjustment(
                    self._current_batch_size
                )

                if suggested_size != self._current_batch_size:
                    logger.warning(
                        f"Memory pressure detected ({pressure}): "
                        f"reducing batch size from {self._current_batch_size} to {suggested_size}"
                    )
                    self._current_batch_size = suggested_size

                # Split current batch if it's too large
                if len(images) > self._current_batch_size:
                    logger.info(
                        f"Splitting batch of {len(images)} into sub-batches "
                        f"of {self._current_batch_size} due to memory pressure"
                    )
                    return self._process_in_sub_batches(images, self._current_batch_size)

            # Log VRAM stats periodically
            if hasattr(self, '_images_processed'):
                self._images_processed += len(images)
                if self._images_processed % 50 == 0:
                    self._vram_monitor.log_stats()
            else:
                self._images_processed = len(images)

        results = []

        for i, image in enumerate(images):
            result = self.process_image(image)
            results.append(result)

            # Periodic garbage collection to manage memory
            if (i + 1) % 10 == 0:
                gc.collect()

                # Check memory pressure during processing
                if self._vram_monitor and self._adaptive_enabled:
                    if self._vram_monitor.should_reduce_batch_size():
                        logger.warning(
                            f"Memory pressure detected during processing "
                            f"(image {i+1}/{len(images)})"
                        )

        return results

    def _process_in_sub_batches(
        self,
        images: List[np.ndarray],
        sub_batch_size: int
    ) -> List[OCRResult]:
        """
        Process images in smaller sub-batches to manage memory.

        Args:
            images: List of all images
            sub_batch_size: Size of each sub-batch

        Returns:
            List of all OCRResult objects
        """
        all_results = []

        for i in range(0, len(images), sub_batch_size):
            sub_batch = images[i:i + sub_batch_size]
            logger.debug(
                f"Processing sub-batch {i//sub_batch_size + 1}: "
                f"images {i+1}-{min(i+sub_batch_size, len(images))}"
            )

            # Process sub-batch
            sub_results = []
            for image in sub_batch:
                result = self.process_image(image)
                sub_results.append(result)

            all_results.extend(sub_results)

            # Aggressive garbage collection between sub-batches
            gc.collect()

            # Check if we can increase batch size (memory freed up)
            if self._vram_monitor and self._adaptive_enabled:
                pressure = self._vram_monitor.get_memory_pressure_level()
                if pressure == "none" and sub_batch_size < self.config.batch_size:
                    # Can potentially increase batch size
                    self._current_batch_size = min(
                        self._current_batch_size + 5,
                        self.config.batch_size
                    )
                    logger.info(
                        f"Memory pressure relieved, increasing batch size to "
                        f"{self._current_batch_size}"
                    )

        return all_results

    def cleanup(self) -> None:
        """Release PaddleOCR resources"""
        if self.ocr is not None:
            del self.ocr
            self.ocr = None

        if self._vram_monitor is not None:
            self._vram_monitor = None

        gc.collect()
        self._initialized = False
        logger.info("PaddleOCR engine cleaned up")

    def get_memory_usage(self) -> int:
        """Get estimated memory usage in bytes"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    def supports_gpu(self) -> bool:
        """Check if GPU is supported and available"""
        return self._gpu_available
