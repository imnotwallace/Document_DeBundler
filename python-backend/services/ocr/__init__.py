"""
OCR Service Package
Provides abstraction layer for multiple OCR engines with GPU support
"""

from .base import OCREngine, OCRResult, OCRConfig
from .manager import OCRManager, create_ocr_manager
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
