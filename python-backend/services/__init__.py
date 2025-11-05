"""Services package for Document De-Bundler"""

from .pdf_processor import PDFProcessor
from .naming_service import NamingService
from .bundler import Bundler

__all__ = ['PDFProcessor', 'OCRService', 'NamingService', 'Bundler']


# Lazy import for OCRService to avoid eager loading of heavy OCR engines
# This allows `from services import OCRService` to work while preventing
# transitive imports when only PDFProcessor is needed
def __getattr__(name):
    if name == "OCRService":
        from .ocr_service import OCRService
        return OCRService
    raise AttributeError(f"module 'services' has no attribute '{name}'")
