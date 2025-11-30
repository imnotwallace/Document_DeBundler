"""
Model Lifecycle Manager for Multi-Pass Pipeline.

Ensures only ONE GPU model is loaded at any time.
Provides VRAM-aware loading and aggressive cleanup between passes.
"""

import gc
import time
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class ModelLifecycleManager:
    """
    Manages GPU model lifecycle for guaranteed 4GB VRAM operation.

    Key responsibilities:
    - Ensure only one model loaded at a time
    - Aggressive VRAM cleanup between passes
    - VRAM verification before loading
    - Graceful fallback to CPU if needed
    """

    def __init__(self, vram_monitor: Optional[Any] = None):
        """
        Initialize model lifecycle manager.

        Args:
            vram_monitor: Optional VRAMMonitor instance. If None, creates one.
        """
        if vram_monitor is None:
            from ..vram_monitor import VRAMMonitor
            self.vram_monitor = VRAMMonitor()
        else:
            self.vram_monitor = vram_monitor

        self._current_model: Optional[Any] = None
        self._current_model_name: Optional[str] = None
        self._torch_available = self._check_torch()

    def _check_torch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False

    def ensure_vram_clear(self, required_mb: int = 2000, timeout_sec: float = 10.0) -> bool:
        """
        Ensure sufficient VRAM is available.

        Args:
            required_mb: Required free VRAM in MB
            timeout_sec: Maximum time to wait for memory to free

        Returns:
            True if sufficient VRAM available, False otherwise
        """
        logger.info(f"Ensuring {required_mb}MB VRAM is free...")

        # First, do aggressive cleanup
        self.aggressive_cleanup()

        # Check VRAM
        stats = self.vram_monitor.get_stats(force_refresh=True)

        if stats is None:
            logger.warning("VRAM monitoring not available, proceeding anyway")
            return True

        free_mb = stats.free_gb * 1024
        logger.info(f"Current free VRAM: {free_mb:.0f}MB")

        if free_mb >= required_mb:
            return True

        # Wait for memory to free
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            self.aggressive_cleanup()
            time.sleep(0.5)

            stats = self.vram_monitor.get_stats(force_refresh=True)
            if stats:
                free_mb = stats.free_gb * 1024
                logger.debug(f"Free VRAM: {free_mb:.0f}MB")

                if free_mb >= required_mb:
                    logger.info(f"VRAM cleared: {free_mb:.0f}MB available")
                    return True

        logger.warning(
            f"Insufficient VRAM after {timeout_sec}s: {free_mb:.0f}MB < {required_mb}MB"
        )
        return False

    def aggressive_cleanup(self) -> None:
        """
        Aggressively free GPU memory.

        Performs:
        1. Python garbage collection
        2. CUDA cache clearing
        3. CUDA synchronization
        """
        gc.collect()

        if self._torch_available:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                logger.debug(f"CUDA cleanup warning: {e}")

        # Also try Paddle cleanup
        try:
            import paddle
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
        except Exception:
            pass

        # Small delay to allow GPU to reclaim
        time.sleep(0.2)

    def unload_model(self, model: Any, model_name: str = "unknown") -> None:
        """
        Unload a model and free its memory.

        Args:
            model: Model instance to unload
            model_name: Name for logging
        """
        if model is None:
            return

        logger.info(f"Unloading model: {model_name}")

        # Delete model reference
        del model

        # Clear tracked model if it matches
        if self._current_model is not None:
            self._current_model = None
            self._current_model_name = None

        # Aggressive cleanup
        self.aggressive_cleanup()

        logger.info(f"Model {model_name} unloaded")
        self.log_vram_status()

    def load_paddleocr(
        self,
        use_gpu: bool = True,
        lang: str = "en",
        **kwargs
    ) -> Any:
        """
        Load PaddleOCR engine with VRAM management.

        Args:
            use_gpu: Whether to use GPU
            lang: Language for OCR
            **kwargs: Additional PaddleOCR arguments

        Returns:
            PaddleOCREngine instance
        """
        logger.info("Loading PaddleOCR...")

        # Ensure previous model is unloaded
        if self._current_model is not None:
            logger.warning(f"Unloading previous model: {self._current_model_name}")
            self.unload_model(self._current_model, self._current_model_name)

        # Check VRAM (PaddleOCR needs ~500MB)
        if use_gpu:
            if not self.ensure_vram_clear(required_mb=500):
                logger.warning("Insufficient VRAM for PaddleOCR GPU, falling back to CPU")
                use_gpu = False

        # Import and create engine
        from ..engines.paddleocr_engine import PaddleOCREngine
        from ..base import OCRConfig

        config = OCRConfig(
            engine="paddleocr",
            use_gpu=use_gpu,
            languages=[lang],
            **kwargs
        )

        engine = PaddleOCREngine(config)
        engine.initialize()

        self._current_model = engine
        self._current_model_name = "PaddleOCR"

        logger.info(f"PaddleOCR loaded (GPU: {use_gpu})")
        self.log_vram_status()

        return engine

    def load_doclayout(
        self,
        model_name: str = "PP-DocLayout-M",
        device: str = "gpu",
        confidence_threshold: float = 0.5,
    ) -> Any:
        """
        Load PP-DocLayout model with VRAM management.

        PP-DocLayout is much smaller than Florence-2 (22MB vs 2.5GB),
        making it ideal for 4GB VRAM systems.

        Args:
            model_name: Model variant (PP-DocLayout-L, M, or S)
            device: Device to load on ("gpu" or "cpu")
            confidence_threshold: Minimum confidence for detections

        Returns:
            DocLayoutEngine instance
        """
        logger.info(f"Loading PP-DocLayout: {model_name}")

        # Ensure previous model is unloaded
        if self._current_model is not None:
            logger.warning(f"Unloading previous model: {self._current_model_name}")
            self.unload_model(self._current_model, self._current_model_name)

        # Check VRAM (PP-DocLayout needs only ~200MB)
        if device == "gpu":
            if not self.ensure_vram_clear(required_mb=200):
                logger.warning("Insufficient VRAM for PP-DocLayout GPU, falling back to CPU")
                device = "cpu"

        # Import and create engine
        from ..layout_analysis.doclayout_engine import DocLayoutEngine

        engine = DocLayoutEngine(
            model_name=model_name,
            device=device,
            confidence_threshold=confidence_threshold,
        )
        engine.initialize()

        self._current_model = engine
        self._current_model_name = "PP-DocLayout"

        logger.info(f"PP-DocLayout loaded on {device}")
        self.log_vram_status()

        return engine

    def log_vram_status(self) -> None:
        """Log current VRAM usage."""
        self.vram_monitor.log_stats()

    def get_vram_info(self) -> dict:
        """Get current VRAM information."""
        return self.vram_monitor.get_info()

    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        stats = self.vram_monitor.get_stats()
        return stats is not None

    def get_free_vram_mb(self) -> float:
        """Get free VRAM in MB."""
        stats = self.vram_monitor.get_stats(force_refresh=True)
        if stats:
            return stats.free_gb * 1024
        return 0.0

    def unload_current(self) -> None:
        """Unload currently loaded model."""
        if self._current_model is not None:
            self.unload_model(self._current_model, self._current_model_name)


def wait_for_memory_free(vram_monitor: Any, min_free_mb: int, timeout_sec: float = 30.0) -> bool:
    """
    Wait for GPU memory to become available.

    Args:
        vram_monitor: VRAMMonitor instance
        min_free_mb: Minimum required free memory in MB
        timeout_sec: Maximum time to wait

    Returns:
        True if memory is available, False if timeout
    """
    logger.info(f"Waiting for {min_free_mb}MB free VRAM...")

    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        # Force garbage collection
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        stats = vram_monitor.get_stats(force_refresh=True)
        if stats:
            free_mb = stats.free_gb * 1024
            if free_mb >= min_free_mb:
                logger.info(f"VRAM available: {free_mb:.0f}MB")
                return True

        time.sleep(0.5)

    logger.warning(f"Timeout waiting for VRAM after {timeout_sec}s")
    return False
