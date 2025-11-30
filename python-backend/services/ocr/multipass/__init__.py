"""
Multi-pass OCR Pipeline with PP-DocLayout Layout Analysis.

This module implements a sequential multi-pass architecture that:
1. Runs PaddleOCR for text extraction (Pass 1)
2. Runs PP-DocLayout for layout analysis (Pass 2) - 22MB model, 13ms/page
3. Fuses results for reading order detection (Pass 3)
4. Creates searchable PDF pages (Pass 4)
5. Merges into final output (Pass 5)

Guaranteed to operate within 4GB VRAM by loading only ONE model at a time.

Note: FlorenceRegion/FlorenceLayoutResult class names are legacy and work
with PP-DocLayout. Will be renamed to LayoutRegion/LayoutResult in future.
"""

from .pipeline_orchestrator import MultiPassPipelineOrchestrator
from .checkpoint_manager import CheckpointManager
from .model_lifecycle import ModelLifecycleManager
from .pass_config import MultiPassConfig
from .data_structures import (
    FlorenceRegion,
    FlorenceLayoutResult,
    PageCheckpoint,
    PassStatus,
    PipelineState,
)

__all__ = [
    "MultiPassPipelineOrchestrator",
    "CheckpointManager",
    "ModelLifecycleManager",
    "MultiPassConfig",
    "FlorenceRegion",
    "FlorenceLayoutResult",
    "PageCheckpoint",
    "PassStatus",
    "PipelineState",
]
