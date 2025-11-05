"""
OCR Service Package
Provides abstraction layer for multiple OCR engines with GPU support
"""

from .base import OCREngine, OCRResult, OCRConfig
from .config import get_default_config, detect_hardware_capabilities, get_adaptive_dpi, get_model_directory
from .vram_monitor import VRAMMonitor, VRAMStats

__all__ = [
    'OCREngine',
    'OCRResult',
    'OCRConfig',
    'OCRManager',
    'create_ocr_manager',
    'get_default_config',
    'detect_hardware_capabilities',
    'get_adaptive_dpi',
    'get_model_directory',
    'VRAMMonitor',
    'VRAMStats',
]


# Lazy imports for OCRManager to avoid eager loading of heavy OCR engines
# This allows `from services.ocr import OCRManager` to work while preventing
# transitive imports of PaddleOCREngine when only config/base classes are needed
def __getattr__(name):
    if name == "OCRManager":
        from .manager import OCRManager
        return OCRManager
    if name == "create_ocr_manager":
        from .manager import create_ocr_manager
        return create_ocr_manager
    raise AttributeError(f"module 'services.ocr' has no attribute '{name}'")
