"""
Base OCR Engine Abstract Interface
Defines the contract that all OCR engines must implement
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
from pathlib import Path
import numpy as np


@dataclass
class OCRResult:
    """Result from OCR processing"""
    text: str
    confidence: float
    bbox: Optional[List[List[int]]] = None  # Bounding boxes for detected text
    raw_result: Optional[Any] = None  # Engine-specific raw result
    error: Optional[str] = None
    processing_time: float = 0.0  # Processing time in seconds


@dataclass
class OCRConfig:
    """Configuration for OCR processing"""
    # Engine selection
    engine: Literal["paddleocr", "tesseract", "auto"] = "auto"

    # Hardware settings
    use_gpu: bool = True  # Will auto-detect and fallback to CPU if unavailable
    gpu_id: int = 0

    # Language settings
    languages: List[str] = field(default_factory=lambda: ["en"])  # Language code (en for English)

    # Processing settings
    batch_size: int = 10  # Pages to process at once
    max_memory_mb: int = 2048  # Maximum memory usage
    confidence_threshold: float = 0.5  # Minimum confidence to accept results

    # Text detection settings
    enable_text_detection: bool = True
    enable_angle_classification: bool = True  # Detect and correct text rotation

    # Performance settings
    num_threads: int = 4  # For CPU processing

    # Hybrid GPU/CPU processing settings
    enable_hybrid_mode: bool = False  # Enable hybrid GPU/CPU processing
    cpu_batch_size: Optional[int] = None  # Separate batch size for CPU (auto-calculated if None)
    enable_vram_monitoring: bool = True  # Monitor VRAM and adapt batch sizes
    enable_adaptive_batch_sizing: bool = True  # Automatically adjust batch size based on memory pressure

    # Model paths (for bundled models)
    model_dir: Optional[Path] = None

    # Engine-specific settings
    engine_settings: Dict[str, Any] = field(default_factory=dict)

    # Logging
    verbose: bool = False


class OCREngine(ABC):
    """Abstract base class for OCR engines"""

    def __init__(self, config: OCRConfig):
        self.config = config
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the OCR engine, load models, etc.
        Should be called before first use.
        """
        pass

    @abstractmethod
    def process_image(self, image: np.ndarray) -> OCRResult:
        """
        Process a single image and return OCR result.

        Args:
            image: Image as numpy array (RGB or grayscale)

        Returns:
            OCRResult with extracted text and metadata
        """
        pass

    @abstractmethod
    def process_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """
        Process multiple images in batch for better performance.

        Args:
            images: List of images as numpy arrays

        Returns:
            List of OCRResult objects
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Release resources, unload models, free memory.
        Should be called when OCR engine is no longer needed.
        """
        pass

    @abstractmethod
    def get_memory_usage(self) -> int:
        """
        Get current memory usage in bytes.

        Returns:
            Memory usage in bytes
        """
        pass

    @abstractmethod
    def supports_gpu(self) -> bool:
        """
        Check if this engine supports GPU acceleration.

        Returns:
            True if GPU is supported and available
        """
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if engine has been initialized"""
        return self._initialized

    def __enter__(self):
        """Context manager support"""
        if not self._initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.cleanup()
        return False
