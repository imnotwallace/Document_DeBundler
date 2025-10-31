"""OCR Engine Implementations"""

from .paddleocr_engine import PaddleOCREngine
from .tesseract_engine import TesseractEngine

__all__ = ['PaddleOCREngine', 'TesseractEngine']
