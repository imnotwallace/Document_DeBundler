"""
VRAM Monitor for GPU Memory Management
Provides real-time GPU memory tracking and adaptive batch sizing
"""

import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VRAMStats:
    """VRAM usage statistics"""
    total_gb: float
    used_gb: float
    free_gb: float
    utilization_percent: float
    timestamp: float


class VRAMMonitor:
    """
    Monitors GPU memory usage and provides adaptive batch sizing.

    Helps optimize processing for systems with limited VRAM (e.g., 4GB)
    by detecting memory pressure and recommending batch size adjustments.
    """

    # Memory pressure thresholds
    THRESHOLD_LOW = 0.6      # 60% - comfortable
    THRESHOLD_MEDIUM = 0.75  # 75% - moderate pressure
    THRESHOLD_HIGH = 0.85    # 85% - high pressure
    THRESHOLD_CRITICAL = 0.95  # 95% - critical, risk of OOM

    def __init__(self, check_interval: float = 1.0):
        """
        Initialize VRAM monitor.

        Args:
            check_interval: Minimum seconds between checks (rate limiting)
        """
        self.check_interval = check_interval
        self._last_check_time = 0.0
        self._cached_stats: Optional[VRAMStats] = None
        self._gpu_available = self._check_gpu_support()

        if self._gpu_available:
            logger.info("VRAM monitor initialized with GPU support")
        else:
            logger.info("VRAM monitor initialized without GPU (CPU mode)")

    def _check_gpu_support(self) -> bool:
        """Check if GPU monitoring is supported"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import paddle
                return paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
            except ImportError:
                return False

    def get_stats(self, force_refresh: bool = False) -> Optional[VRAMStats]:
        """
        Get current VRAM statistics.

        Args:
            force_refresh: Force a new check, ignoring cache

        Returns:
            VRAMStats if GPU available, None otherwise
        """
        if not self._gpu_available:
            return None

        current_time = time.time()

        # Use cached stats if within check interval
        if not force_refresh and self._cached_stats:
            if current_time - self._last_check_time < self.check_interval:
                return self._cached_stats

        # Get fresh stats
        try:
            stats = self._read_vram_stats()
            self._cached_stats = stats
            self._last_check_time = current_time
            return stats
        except Exception as e:
            logger.error(f"Failed to read VRAM stats: {e}")
            return None

    def _read_vram_stats(self) -> VRAMStats:
        """Read VRAM statistics from GPU"""
        # Try PyTorch first (most common)
        try:
            import torch
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory
                reserved = torch.cuda.memory_reserved(0)
                allocated = torch.cuda.memory_allocated(0)
                free = total - reserved

                total_gb = total / (1024 ** 3)
                used_gb = reserved / (1024 ** 3)
                free_gb = free / (1024 ** 3)
                utilization = reserved / total if total > 0 else 0.0

                return VRAMStats(
                    total_gb=total_gb,
                    used_gb=used_gb,
                    free_gb=free_gb,
                    utilization_percent=utilization * 100,
                    timestamp=time.time()
                )
        except (ImportError, RuntimeError):
            pass

        # Note: PaddlePaddle doesn't expose detailed VRAM stats easily.
        # We skip it here and fall through to nvidia-smi which works for all CUDA GPUs.
        # This ensures VRAM monitoring works correctly for the primary OCR engine.

        # Try nvidia-smi as last resort
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
                 '--format=csv,noheader,nounits', '--id=0'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                total_mb, used_mb, free_mb = map(float, result.stdout.strip().split(','))
                total_gb = total_mb / 1024
                used_gb = used_mb / 1024
                free_gb = free_mb / 1024
                utilization = (used_mb / total_mb * 100) if total_mb > 0 else 0.0

                return VRAMStats(
                    total_gb=total_gb,
                    used_gb=used_gb,
                    free_gb=free_gb,
                    utilization_percent=utilization,
                    timestamp=time.time()
                )
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        raise RuntimeError("Unable to read VRAM stats from any available method")

    def get_memory_pressure_level(self) -> str:
        """
        Get current memory pressure level.

        Returns:
            One of: "none", "low", "medium", "high", "critical"
        """
        stats = self.get_stats()

        if not stats:
            return "none"  # No GPU

        utilization = stats.utilization_percent / 100.0

        if utilization >= self.THRESHOLD_CRITICAL:
            return "critical"
        elif utilization >= self.THRESHOLD_HIGH:
            return "high"
        elif utilization >= self.THRESHOLD_MEDIUM:
            return "medium"
        elif utilization >= self.THRESHOLD_LOW:
            return "low"
        else:
            return "none"

    def should_reduce_batch_size(self) -> bool:
        """
        Check if batch size should be reduced due to memory pressure.

        Returns:
            True if memory pressure is high or critical
        """
        pressure = self.get_memory_pressure_level()
        return pressure in ("high", "critical")

    def suggest_batch_size_adjustment(self, current_batch_size: int) -> int:
        """
        Suggest adjusted batch size based on memory pressure.

        Args:
            current_batch_size: Current batch size

        Returns:
            Suggested batch size (may be same, lower, or higher)
        """
        pressure = self.get_memory_pressure_level()

        if pressure == "critical":
            # Emergency reduction
            return max(1, current_batch_size // 4)
        elif pressure == "high":
            # Significant reduction
            return max(1, current_batch_size // 2)
        elif pressure == "medium":
            # Moderate reduction
            return max(1, int(current_batch_size * 0.75))
        elif pressure == "low":
            # Keep as is
            return current_batch_size
        else:
            # No pressure - can potentially increase
            return min(current_batch_size + 5, current_batch_size * 2)

    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive VRAM information.

        Returns:
            Dictionary with VRAM status and recommendations
        """
        stats = self.get_stats()

        if not stats:
            return {
                "available": False,
                "reason": "No GPU detected or monitoring not supported"
            }

        pressure = self.get_memory_pressure_level()

        return {
            "available": True,
            "total_gb": round(stats.total_gb, 2),
            "used_gb": round(stats.used_gb, 2),
            "free_gb": round(stats.free_gb, 2),
            "utilization_percent": round(stats.utilization_percent, 1),
            "pressure_level": pressure,
            "should_reduce_batch": self.should_reduce_batch_size(),
            "timestamp": stats.timestamp
        }

    def log_stats(self) -> None:
        """Log current VRAM statistics"""
        info = self.get_info()

        if info["available"]:
            logger.info(
                f"VRAM: {info['used_gb']:.2f}GB / {info['total_gb']:.2f}GB "
                f"({info['utilization_percent']:.1f}%) - "
                f"Pressure: {info['pressure_level']}"
            )
        else:
            logger.debug("VRAM monitoring not available (CPU mode)")
