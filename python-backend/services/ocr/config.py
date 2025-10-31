"""
OCR Configuration and Hardware Detection
"""

import logging
import platform
import psutil
from pathlib import Path
from typing import Optional, Dict, Any
from .base import OCRConfig

logger = logging.getLogger(__name__)


def detect_gpu_cuda() -> bool:
    """Detect if CUDA-compatible GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def detect_gpu_directml() -> bool:
    """Detect if DirectML is available (Windows AMD/Intel GPUs)"""
    if platform.system() != "Windows":
        return False

    try:
        # Check if DirectML libraries are available
        import torch_directml
        return True
    except ImportError:
        return False


def get_gpu_memory_gb() -> float:
    """Get available GPU memory in GB"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            return gpu_memory / (1024 ** 3)  # Convert to GB
    except:
        pass
    return 0.0


def get_system_memory_gb() -> float:
    """Get total system memory in GB"""
    return psutil.virtual_memory().total / (1024 ** 3)


def get_available_memory_gb() -> float:
    """Get currently available system memory in GB"""
    return psutil.virtual_memory().available / (1024 ** 3)


def get_optimal_batch_size(use_gpu: bool, gpu_memory_gb: float = 0, system_memory_gb: Optional[float] = None) -> int:
    """
    Calculate optimal batch size based on available hardware.

    Optimized for 4GB VRAM systems with improved utilization.

    Args:
        use_gpu: Whether GPU will be used
        gpu_memory_gb: Available GPU memory in GB
        system_memory_gb: System RAM in GB (auto-detected if None)

    Returns:
        Recommended batch size
    """
    if system_memory_gb is None:
        system_memory_gb = get_system_memory_gb()

    if use_gpu and gpu_memory_gb > 0:
        # GPU batch sizing - optimized for better VRAM utilization
        if gpu_memory_gb < 1.5:
            return 5  # Ultra low-end GPUs
        elif gpu_memory_gb < 2.5:
            return 8  # 2GB GPUs
        elif gpu_memory_gb < 3.5:
            return 15  # 3GB GPUs
        elif gpu_memory_gb < 5:
            # 4GB GPUs - optimal for target hardware
            # Can handle 25-30 pages but be conservative
            return 25
        elif gpu_memory_gb < 7:
            return 35  # 6GB GPUs
        elif gpu_memory_gb < 10:
            return 50  # 8GB GPUs
        else:
            return 60  # High-end GPUs (10GB+)
    else:
        # CPU batch sizing - depends on available RAM
        if system_memory_gb < 6:
            return 3  # Minimal RAM
        elif system_memory_gb < 10:
            return 5  # 8GB RAM
        elif system_memory_gb < 20:
            return 10  # 16GB RAM
        elif system_memory_gb < 28:
            return 15  # 24GB RAM
        else:
            return 20  # 32GB+ RAM


def detect_hardware_capabilities() -> Dict[str, Any]:
    """
    Detect available hardware capabilities.

    Returns:
        Dictionary with hardware information
    """
    cuda_available = detect_gpu_cuda()
    directml_available = detect_gpu_directml()
    gpu_available = cuda_available or directml_available

    gpu_memory = get_gpu_memory_gb() if cuda_available else 0.0
    system_memory = get_system_memory_gb()

    capabilities = {
        'gpu_available': gpu_available,
        'cuda_available': cuda_available,
        'directml_available': directml_available,
        'gpu_memory_gb': gpu_memory,
        'system_memory_gb': system_memory,
        'cpu_count': psutil.cpu_count(logical=False) or 1,
        'platform': platform.system(),
    }

    logger.info(f"Hardware capabilities detected: {capabilities}")
    return capabilities


def get_default_config(
    engine: Optional[str] = None,
    prefer_gpu: bool = True,
    model_dir: Optional[Path] = None
) -> OCRConfig:
    """
    Get default OCR configuration with hardware auto-detection.

    Args:
        engine: Specific engine to use, or None for auto-selection
        prefer_gpu: Whether to prefer GPU if available
        model_dir: Path to bundled models directory

    Returns:
        OCRConfig with optimal settings for current hardware
    """
    capabilities = detect_hardware_capabilities()

    # Determine if we should use GPU
    use_gpu = False
    if prefer_gpu and capabilities['gpu_available']:
        use_gpu = True
        logger.info("GPU detected and enabled for OCR processing")
    else:
        if prefer_gpu:
            logger.info("GPU not available, falling back to CPU")
        else:
            logger.info("CPU mode selected")

    # Calculate optimal batch size for GPU
    gpu_batch_size = get_optimal_batch_size(
        use_gpu,
        capabilities['gpu_memory_gb'],
        capabilities['system_memory_gb']
    )

    # Calculate CPU batch size (for hybrid mode or CPU-only)
    cpu_batch_size = get_optimal_batch_size(
        False,  # CPU mode
        0,
        capabilities['system_memory_gb']
    )

    # Determine if hybrid mode should be enabled
    # Enable hybrid for systems with limited VRAM (≤4GB) and sufficient RAM (≥15GB)
    enable_hybrid = (
        use_gpu and
        capabilities['gpu_memory_gb'] <= 4.5 and
        capabilities['system_memory_gb'] >= 15
    )

    # Determine engine
    if engine is None:
        # Auto-select: PaddleOCR if possible, else Tesseract
        engine = "paddleocr"  # Default to PaddleOCR

    # Set max memory based on available RAM (use 50% of available)
    max_memory_mb = int(capabilities['system_memory_gb'] * 1024 * 0.5)

    # Number of CPU threads
    num_threads = max(1, capabilities['cpu_count'] - 1)  # Leave one core free

    config = OCRConfig(
        engine=engine,
        use_gpu=use_gpu,
        batch_size=gpu_batch_size,
        max_memory_mb=max_memory_mb,
        num_threads=num_threads,
        model_dir=model_dir,
        enable_hybrid_mode=enable_hybrid,
        cpu_batch_size=cpu_batch_size,
        enable_vram_monitoring=use_gpu,  # Only monitor VRAM if using GPU
        enable_adaptive_batch_sizing=True,
    )

    logger.info(
        f"Default OCR config: engine={engine}, gpu={use_gpu}, "
        f"gpu_batch_size={gpu_batch_size}, cpu_batch_size={cpu_batch_size}, "
        f"hybrid_mode={enable_hybrid}, threads={num_threads}"
    )

    return config


def get_adaptive_dpi(
    use_gpu: bool,
    gpu_memory_gb: float = 0,
    system_memory_gb: Optional[float] = None,
    target_quality: str = "balanced"
) -> int:
    """
    Calculate adaptive DPI based on available hardware and quality target.

    Args:
        use_gpu: Whether GPU is being used
        gpu_memory_gb: Available GPU memory in GB
        system_memory_gb: System RAM in GB (auto-detected if None)
        target_quality: Quality target ("low", "balanced", "high")

    Returns:
        Recommended DPI (150-600)
    """
    if system_memory_gb is None:
        system_memory_gb = get_system_memory_gb()

    # Base DPI recommendations by quality target
    quality_dpi = {
        "low": 150,       # Fast, lower quality
        "balanced": 300,  # Good balance (default)
        "high": 450,      # High quality
        "max": 600,       # Maximum quality
    }

    base_dpi = quality_dpi.get(target_quality, 300)

    # Adjust based on available memory
    if use_gpu and gpu_memory_gb > 0:
        # GPU processing - can handle higher DPI better
        if gpu_memory_gb < 2:
            # Very limited VRAM - reduce DPI
            max_dpi = 200
        elif gpu_memory_gb < 4:
            # Limited VRAM (2-4GB) - moderate DPI
            max_dpi = 300
        elif gpu_memory_gb < 6:
            # 4-6GB VRAM - good DPI
            max_dpi = 400
        else:
            # Ample VRAM - high DPI
            max_dpi = 600
    else:
        # CPU processing - more conservative
        if system_memory_gb < 6:
            max_dpi = 150
        elif system_memory_gb < 10:
            max_dpi = 200
        elif system_memory_gb < 20:
            max_dpi = 300
        else:
            max_dpi = 450

    # Return the minimum of requested and maximum allowed
    return min(base_dpi, max_dpi)


def get_model_directory() -> Path:
    """
    Get the directory where bundled OCR models are stored.

    Returns:
        Path to models directory
    """
    # Models are bundled in python-backend/models/
    current_file = Path(__file__)
    models_dir = current_file.parent.parent.parent / "models"

    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Created models directory: {models_dir}")

    return models_dir
