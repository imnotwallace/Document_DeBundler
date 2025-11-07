"""
PaddleOCR Engine Implementation
High-performance OCR with GPU support
"""

# IMPORTANT: Import CUDA path fix FIRST, before any PyTorch/PaddleOCR imports
# This ensures cuDNN DLLs can be found without modifying system PATH
from ..cuda_path_fix import add_cuda_dlls_to_path
add_cuda_dlls_to_path()

# IMPORTANT: Apply max_side_limit patch BEFORE PaddleOCR imports
# This overrides the default 4000px limit to support high-DPI scans
from ..max_side_limit_patch import apply_max_side_limit_patch
apply_max_side_limit_patch()  # Auto-detects GPU/CPU memory and sets appropriate limit

import logging
import time
import gc
from typing import List, Optional
from pathlib import Path
import numpy as np

from ..base import OCREngine, OCRResult, OCRConfig
from ..vram_monitor import VRAMMonitor
from ..config import detect_model_type

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
        self._model_version: str = "mobile"  # Will be set during initialization
        self._rec_model_name: str = "PP-OCRv5_mobile_rec"  # Default fallback  # Will be detected during initialization

    def initialize(self) -> None:
        """Initialize PaddleOCR engine"""
        if self._initialized:
            logger.warning("PaddleOCR already initialized")
            return

        try:
            from paddleocr import PaddleOCR
            import paddle

            # Determine GPU usage
            use_gpu = self.config.use_gpu and self._check_gpu_available()

            if use_gpu:
                logger.info("Initializing PaddleOCR with GPU acceleration")
                
                # CRITICAL FIX: Pre-initialize CUDA context to prevent first-inference hang
                try:
                    logger.info("Pre-initializing CUDA context...")
                    paddle.device.set_device('gpu:0')
                    # Force CUDA initialization with a simple operation
                    dummy = paddle.to_tensor([1.0], place=paddle.CUDAPlace(0))
                    paddle.device.cuda.synchronize()
                    del dummy
                    logger.info("CUDA context initialized successfully")
                except Exception as e:
                    logger.warning(f"CUDA pre-initialization failed: {e}, will retry during model load")
            else:
                if self.config.use_gpu:
                    logger.warning("GPU requested but not available, using CPU")
                else:
                    logger.info("Initializing PaddleOCR with CPU")

            # Get language(s) and determine model version
            lang = self.config.languages[0] if self.config.languages else 'en'
            model_version = getattr(self.config, 'model_version', 'mobile')  # Default to mobile

            # Import language pack metadata to get correct model name
            try:
                import sys
                from pathlib import Path
                # Add services directory to path if not already there
                services_dir = Path(__file__).parent.parent.parent
                if str(services_dir) not in sys.path:
                    sys.path.insert(0, str(services_dir))

                from services.language_pack_metadata import get_language_pack_with_version

                # Get language pack info with the requested version
                lang_pack = get_language_pack_with_version(lang, model_version)

                if lang_pack:
                    rec_model_name = lang_pack.get_recognition_model_name()
                    det_model_name = lang_pack.detection_model_name
                    logger.info(f"Selected models - Detection: {det_model_name}, Recognition: {rec_model_name} (version: {model_version})")
                else:
                    # Fall back to English mobile version if language not found
                    logger.warning(f"Language '{lang}' not found in metadata, falling back to English mobile")
                    rec_model_name = "en_PP-OCRv5_mobile_rec"
                    det_model_name = "en_PP-OCRv5_mobile_det"
                    lang = "en"
            except Exception as e:
                # Fall back to English mobile version on any error
                logger.warning(f"Failed to determine model version from metadata: {e}, falling back to English mobile")
                rec_model_name = "en_PP-OCRv5_mobile_rec"
                det_model_name = "en_PP-OCRv5_mobile_det"
                lang = "en"
                model_version = "mobile"

            logger.info(f"Initializing PaddleOCR - Language: {lang}, Detection: {det_model_name}, Recognition: {rec_model_name}, Version: {model_version}")

            # Initialize PaddleOCR with 3.x API using correct parameter names
            # Use text_detection_model_name and text_recognition_model_name (NOT det_model_name/rec_model_name)
            ocr_kwargs = {
                'lang': lang,
                'text_detection_model_name': det_model_name,
                'text_recognition_model_name': rec_model_name,
                'use_textline_orientation': self.config.enable_angle_classification,
                'text_det_limit_side_len': 18000,      # Support ultra-high-DPI scans
                'text_det_limit_type': 'max',          # Specify max dimension behavior
                'device': 'gpu' if use_gpu else 'cpu',
            }

            # If models are bundled, point to model directory
            # Only specify model directories if they actually exist
            logger.info(f"Checking for bundled models in: {self.config.model_dir}")
            if self.config.model_dir and self.config.model_dir.exists():
                det_dir = self.config.model_dir / 'det'
                rec_dir = self.config.model_dir / 'rec'
                cls_dir = self.config.model_dir / 'cls'

                logger.info(f"Model directories - det exists: {det_dir.exists()}, rec exists: {rec_dir.exists()}, cls exists: {cls_dir.exists()}")

                has_models = False
                if det_dir.exists() and rec_dir.exists():
                    # Check if models have required inference.yml (PaddleOCR 3.x requirement)
                    det_config = det_dir / 'inference.yml'
                    rec_config = rec_dir / 'inference.yml'
                    
                    if det_config.exists() and rec_config.exists():
                        logger.info(f"Using bundled models from: {self.config.model_dir}")
                        ocr_kwargs['det_model_dir'] = str(det_dir)
                        ocr_kwargs['rec_model_dir'] = str(rec_dir)
                        has_models = True
                    else:
                        logger.warning(
                            "Bundled models are missing inference.yml (PaddleOCR 2.x format), "
                            "will auto-download PaddleOCR 3.x compatible models"
                        )

                    if self.config.enable_angle_classification and cls_dir.exists():
                        # Check if cls model has required inference.yml (PaddleOCR 3.x requirement)
                        cls_config = cls_dir / 'inference.yml'
                        if cls_config.exists():
                            ocr_kwargs['cls_model_dir'] = str(cls_dir)
                        else:
                            logger.warning(f"Classification model missing inference.yml, will auto-download")

                if not has_models:
                    logger.info("Bundled models not found, will auto-download from PaddleOCR")
            else:
                logger.info(f"Model directory does not exist or not configured: {self.config.model_dir}")

            # Add improved default parameters for better OCR quality (PaddleOCR 3.x API)
            # These prevent text fragmentation and improve character recognition
            default_params = {
                'text_det_box_thresh': 0.4,      # More sensitive text detection (default: 0.6)
                'text_det_unclip_ratio': 2.0,    # Prevent character fragmentation (default: 1.5)
                'text_rec_score_thresh': 0.4,    # Keep more low-confidence results (default: 0.5)
            }
            ocr_kwargs.update(default_params)
            
            # Apply engine-specific settings (overrides defaults)
            ocr_kwargs.update(self.config.engine_settings)

            # Create OCR instance
            logger.info(f"Creating PaddleOCR instance with kwargs: {ocr_kwargs}")
            logger.info("This may take 30-60 seconds on first run...")

            self.ocr = PaddleOCR(**ocr_kwargs)
            logger.info("PaddleOCR instance created successfully")

            self._gpu_available = use_gpu
            
            # Store model version from config
            self._model_version = model_version
            self._rec_model_name = rec_model_name
            logger.info(f"Model version: {self._model_version}, Recognition model: {self._rec_model_name}")
            
            # CRITICAL FIX: Warmup with dummy inference to complete CUDA initialization
            if use_gpu:
                try:
                    logger.info("Performing GPU warmup inference...")
                    # Create small dummy image (100x100 white image)
                    dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
                    # Run prediction to trigger full CUDA setup
                    _ = self.ocr.predict(dummy_image)
                    # Force synchronization
                    paddle.device.cuda.synchronize()
                    logger.info("GPU warmup completed successfully")
                except Exception as e:
                    logger.error(f"GPU warmup failed: {e}", exc_info=True)
                    raise RuntimeError(f"GPU initialization failed during warmup: {e}")

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

    def warmup(self, dummy_image: Optional[np.ndarray] = None) -> float:
        """
        Warm up the OCR engine by running a dummy inference.

        This pre-compiles CUDA kernels and caches the inference engine,
        making subsequent real operations much faster (typically 5-10s → <1s).

        Args:
            dummy_image: Optional dummy image for warmup. If None, creates a small test image.

        Returns:
            Warmup time in seconds

        Example:
            engine = PaddleOCREngine(config)
            engine.initialize()
            warmup_time = engine.warmup()
            # Now real operations are much faster
        """
        if not self._initialized:
            raise RuntimeError("Cannot warm up: OCR engine not initialized")

        logger.info("Warming up PaddleOCR engine (pre-compiling CUDA kernels)...")
        start_time = time.time()

        try:
            # Create small dummy image if not provided
            if dummy_image is None:
                # 100x100 white image with black text-like features
                dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
                # Add some text-like features (black rectangles simulating text)
                dummy_image[40:45, 20:80] = 0  # Horizontal line (text-like)
                dummy_image[55:60, 20:80] = 0  # Another line

            # Run dummy inference to trigger compilation
            _ = self.process_image(dummy_image)

            warmup_time = time.time() - start_time
            logger.info(
                f"PaddleOCR engine warmed up in {warmup_time:.2f}s "
                f"(CUDA kernels compiled, subsequent operations will be faster)"
            )

            return warmup_time

        except Exception as e:
            warmup_time = time.time() - start_time
            logger.warning(
                f"Warmup failed after {warmup_time:.2f}s: {e}. "
                f"Engine will still work but first operation may be slower."
            )
            # Don't raise - warmup failure shouldn't block actual operations
            return warmup_time

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

        # Log input dimensions for debugging
        logger.debug(f"[PaddleOCR] Input image shape: {image.shape} (H={image.shape[0]}, W={image.shape[1]})")

        start_time = time.time()

        try:
            # CRITICAL FIX: Ensure CUDA synchronization for GPU operations
            if self._gpu_available:
                import paddle
                paddle.device.cuda.synchronize()
            
            # Run OCR using Paddle 3.x predict() API
            result = self.ocr.predict(image)
            
            # Force CUDA synchronization after inference
            if self._gpu_available:
                import paddle
                paddle.device.cuda.synchronize()
            
            # CRITICAL DEBUG: Log what PaddleOCR actually returns
            logger.info(f"PaddleOCR predict() returned: type={type(result)}, value={result}")
            
            # Extract text and confidence from Paddle 3.x format
            text_lines = []
            confidences = []
            bboxes = []
            
            # Handle unexpected result types
            if result == 0 or result is None:
                logger.warning(f"PaddleOCR returned ZERO or None - no text detected in image")
                # Return empty result WITHOUT raw_result to avoid downstream parsing errors
                return OCRResult(
                    text="",
                    confidence=0.0,
                    error="PaddleOCR returned 0 (no text detected)",
                    raw_result=None,  # Explicitly set to None, NOT 0
                    processing_time=time.time() - start_time
                )
            
            if result and len(result) > 0:
                # Paddle 3.x returns list with dict containing 'rec_texts', 'rec_scores', 'rec_polys'
                ocr_data = result[0]
                texts = ocr_data.get('rec_texts', [])
                scores = ocr_data.get('rec_scores', [])
                polys = ocr_data.get('rec_polys', [])
                
                for text, score, poly in zip(texts, scores, polys):
                    if text and text.strip():  # Only add non-empty text
                        text_lines.append(text)
                        confidences.append(score)
                        # Convert numpy array to list of lists for bbox
                        bbox = poly.tolist() if hasattr(poly, 'tolist') else poly
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

    def is_available(self) -> bool:
        """Check if PaddleOCR engine is available and ready to process"""
        return self._initialized and self.ocr is not None

    
    @property
    def model_version(self) -> str:
        """
        Get the model version (mobile or server).
        
        Returns:
            "mobile" or "server"
        """
        return getattr(self, '_model_version', 'mobile')
    
    @property
    def rec_model_name(self) -> str:
        """
        Get the recognition model name being used.
        
        Returns:
            Recognition model name (e.g., "PP-OCRv5_server_rec")
        """
        return getattr(self, '_rec_model_name', 'PP-OCRv5_mobile_rec')
