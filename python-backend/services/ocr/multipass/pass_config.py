"""
Configuration for the multi-pass OCR pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class MultiPassConfig:
    """Configuration for the multi-pass pipeline."""

    # Pass control
    enable_layout: bool = True  # Enable PP-DocLayout layout analysis
    layout_batch_size: int = 4  # PP-DocLayout is much smaller, can batch more
    optimistic_batch_start: int = 4  # Start with batch=4, fallback to 1 on OOM

    # Model settings
    doclayout_model: str = "PP-DocLayout-M"  # Options: PP-DocLayout-L, M, S
    layout_confidence_threshold: float = 0.5
    device: str = "gpu"  # PP-DocLayout uses "gpu" not "cuda"

    # PaddleOCR settings (reuse existing)
    paddle_use_gpu: bool = True
    paddle_lang: str = "en"

    # Checkpoint settings
    enable_checkpoints: bool = True
    checkpoint_dir: Optional[Path] = None
    cleanup_on_success: bool = False  # Keep temp files by default for debugging
    cleanup_on_failure: bool = False  # Never cleanup on failure

    # Memory management
    min_free_vram_mb: int = 500
    vram_check_interval_sec: float = 1.0
    aggressive_cleanup: bool = True

    # Fusion settings
    fusion_strategy: str = "simple"  # "simple" or "advanced" (future)
    min_region_overlap: float = 0.5

    # Parallel processing (CPU passes)
    reading_order_workers: int = 4
    pdf_creation_workers: int = 4

    # Fallback
    fallback_on_layout_failure: bool = True

    # PDF settings
    pdf_compression: bool = True
    jpeg_quality: int = 85

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "enable_layout": self.enable_layout,
            "layout_batch_size": self.layout_batch_size,
            "optimistic_batch_start": self.optimistic_batch_start,
            "doclayout_model": self.doclayout_model,
            "layout_confidence_threshold": self.layout_confidence_threshold,
            "device": self.device,
            "paddle_use_gpu": self.paddle_use_gpu,
            "paddle_lang": self.paddle_lang,
            "enable_checkpoints": self.enable_checkpoints,
            "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
            "cleanup_on_success": self.cleanup_on_success,
            "cleanup_on_failure": self.cleanup_on_failure,
            "min_free_vram_mb": self.min_free_vram_mb,
            "vram_check_interval_sec": self.vram_check_interval_sec,
            "aggressive_cleanup": self.aggressive_cleanup,
            "fusion_strategy": self.fusion_strategy,
            "min_region_overlap": self.min_region_overlap,
            "reading_order_workers": self.reading_order_workers,
            "pdf_creation_workers": self.pdf_creation_workers,
            "fallback_on_layout_failure": self.fallback_on_layout_failure,
            "pdf_compression": self.pdf_compression,
            "jpeg_quality": self.jpeg_quality,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiPassConfig":
        """Deserialize configuration from dictionary."""
        checkpoint_dir = data.get("checkpoint_dir")
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)

        return cls(
            enable_layout=data.get("enable_layout", True),
            layout_batch_size=data.get("layout_batch_size", 4),
            optimistic_batch_start=data.get("optimistic_batch_start", 4),
            doclayout_model=data.get("doclayout_model", "PP-DocLayout-M"),
            layout_confidence_threshold=data.get("layout_confidence_threshold", 0.5),
            device=data.get("device", "gpu"),
            paddle_use_gpu=data.get("paddle_use_gpu", True),
            paddle_lang=data.get("paddle_lang", "en"),
            enable_checkpoints=data.get("enable_checkpoints", True),
            checkpoint_dir=checkpoint_dir,
            cleanup_on_success=data.get("cleanup_on_success", False),
            cleanup_on_failure=data.get("cleanup_on_failure", False),
            min_free_vram_mb=data.get("min_free_vram_mb", 500),
            vram_check_interval_sec=data.get("vram_check_interval_sec", 1.0),
            aggressive_cleanup=data.get("aggressive_cleanup", True),
            fusion_strategy=data.get("fusion_strategy", "simple"),
            min_region_overlap=data.get("min_region_overlap", 0.5),
            reading_order_workers=data.get("reading_order_workers", 4),
            pdf_creation_workers=data.get("pdf_creation_workers", 4),
            fallback_on_layout_failure=data.get("fallback_on_layout_failure", True),
            pdf_compression=data.get("pdf_compression", True),
            jpeg_quality=data.get("jpeg_quality", 85),
        )
