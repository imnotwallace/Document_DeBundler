"""
Layout Analysis Module.

Provides semantic layout understanding for documents using
PP-DocLayout for semantic region detection and projection histogram
for column subdivision.

Features:
- Region detection with semantic labels (23 PP-DocLayout categories)
- Layout type classification (single/multi column, table, etc.)
- Region subdivision for complex multi-column layouts
- OCR + Layout fusion strategies
"""

from .florence_engine import FlorenceEngine
from .doclayout_engine import DocLayoutEngine
from .region_extractor import RegionExtractor
from .region_subdivider import RegionSubdivider, SubdivisionConfig
from .gap_filler import GapFiller, GapFillerConfig
from .fusion_strategies import SimpleFusion, ReadingOrderFusion, RegionFirstFusion

__all__ = [
    "FlorenceEngine",
    "DocLayoutEngine",
    "RegionExtractor",
    "RegionSubdivider",
    "SubdivisionConfig",
    "GapFiller",
    "GapFillerConfig",
    "SimpleFusion",
    "ReadingOrderFusion",
    "RegionFirstFusion",
]
