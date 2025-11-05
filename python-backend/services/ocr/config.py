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


def get_optimal_batch_size(
    use_gpu: bool,
    gpu_memory_gb: float = 0,
    system_memory_gb: Optional[float] = None,
    dpi: int = 300
) -> int:
    """
    Calculate optimal batch size based on available hardware and DPI.

    Optimized for 4GB VRAM systems with improved utilization.
    Accounts for DPI impact on memory usage (higher DPI = larger images = smaller batches).

    Args:
        use_gpu: Whether GPU will be used
        gpu_memory_gb: Available GPU memory in GB
        system_memory_gb: System RAM in GB (auto-detected if None)
        dpi: Target DPI for rendering (default: 300)

    Returns:
        Recommended batch size (adjusted for DPI)
    """
    if system_memory_gb is None:
        system_memory_gb = get_system_memory_gb()

    # Calculate DPI multiplier (relative to 300 DPI baseline)
    # Memory usage scales approximately with (DPI/300)^2
    dpi_multiplier = (dpi / 300.0) ** 2

    if use_gpu and gpu_memory_gb > 0:
        # GPU batch sizing - optimized for better VRAM utilization
        if gpu_memory_gb < 1.5:
            base_batch = 5  # Ultra low-end GPUs
        elif gpu_memory_gb < 2.5:
            base_batch = 8  # 2GB GPUs
        elif gpu_memory_gb < 3.5:
            base_batch = 15  # 3GB GPUs
        elif gpu_memory_gb < 5:
            # 4GB GPUs - optimal for target hardware
            # Can handle 25-30 pages at 300 DPI but be conservative
            base_batch = 25
        elif gpu_memory_gb < 7:
            base_batch = 35  # 6GB GPUs
        elif gpu_memory_gb < 10:
            base_batch = 50  # 8GB GPUs
        else:
            base_batch = 60  # High-end GPUs (10GB+)
        
        # Adjust for DPI
        adjusted_batch = max(1, int(base_batch / dpi_multiplier))
        
    else:
        # CPU batch sizing - depends on available RAM
        if system_memory_gb < 6:
            base_batch = 3  # Minimal RAM
        elif system_memory_gb < 10:
            base_batch = 5  # 8GB RAM
        elif system_memory_gb < 20:
            base_batch = 10  # 16GB RAM
        elif system_memory_gb < 28:
            base_batch = 15  # 24GB RAM
        else:
            base_batch = 20  # 32GB+ RAM
        
        # Adjust for DPI
        adjusted_batch = max(1, int(base_batch / dpi_multiplier))
    
    # Log warning for very high DPI
    if dpi >= 1200 and adjusted_batch == 1:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"High DPI ({dpi}) requires batch_size=1 for {gpu_memory_gb}GB VRAM. "
            f"Consider reducing DPI or processing in smaller chunks."
        )
    
    return adjusted_batch  # 32GB+ RAM


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
        enable_engine_pooling=True,  # Enable engine pooling for reuse (major performance boost)
        enable_warmup=True,  # Warm up engines after initialization
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
        target_quality: Quality target ("low", "balanced", "high", "max", "ultra", "maximum")

    Returns:
        Recommended DPI (150-1600)
    """
    if system_memory_gb is None:
        system_memory_gb = get_system_memory_gb()

    # Base DPI recommendations by quality target
    quality_dpi = {
        "low": 150,       # Fast, lower quality
        "balanced": 300,  # Good balance (default)
        "high": 450,      # High quality
        "max": 600,       # Maximum quality (standard)
        "ultra": 1200,    # Ultra-high quality (requires significant memory)
        "maximum": 1600,  # Maximum quality (requires 6GB+ VRAM)
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
            # 4-6GB VRAM - good DPI, limited high-DPI support
            # Can handle up to 600 DPI comfortably, 1200 DPI with batch_size=1-2
            max_dpi = 400 if base_dpi <= 600 else 1200
        elif gpu_memory_gb < 8:
            # 6-8GB VRAM - can handle 1200 DPI with small batches
            max_dpi = 1200
        else:
            # Ample VRAM (8GB+) - full high-DPI support
            max_dpi = 1600
    else:
        # CPU processing - more conservative
        if system_memory_gb < 6:
            max_dpi = 150
        elif system_memory_gb < 10:
            max_dpi = 200
        elif system_memory_gb < 20:
            max_dpi = 300
        elif system_memory_gb < 32:
            max_dpi = 600
        else:
            # High RAM systems can handle higher DPI, but CPU processing is slow
            max_dpi = 1200

    # Return the minimum of requested and maximum allowed
    recommended_dpi = min(base_dpi, max_dpi)
    
    # Log warnings for high DPI with limited memory
    if base_dpi > recommended_dpi:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Requested DPI ({base_dpi}) reduced to {recommended_dpi} due to memory constraints. "
            f"Available: {'GPU ' + str(gpu_memory_gb) + 'GB VRAM' if use_gpu else 'CPU ' + str(system_memory_gb) + 'GB RAM'}"
        )
    
    return recommended_dpi


def validate_dpi_for_memory(
    dpi: int,
    use_gpu: bool,
    gpu_memory_gb: float = 0,
    system_memory_gb: Optional[float] = None,
    batch_size: int = 1
) -> tuple[bool, str]:
    """
    Validate if the given DPI is safe for the available memory.

    Args:
        dpi: Target DPI
        use_gpu: Whether GPU is being used
        gpu_memory_gb: Available GPU memory in GB
        system_memory_gb: System RAM in GB (auto-detected if None)
        batch_size: Batch size to validate

    Returns:
        Tuple of (is_safe, warning_message)
        is_safe is False if DPI is too high for available memory
    """
    if system_memory_gb is None:
        system_memory_gb = get_system_memory_gb()

    # Estimate memory usage per page at given DPI
    # Letter page (8.5" x 11") in pixels: (8.5*dpi) x (11*dpi)
    # Memory per page (rough estimate): width * height * 3 (RGB) * overhead_factor
    width_px = int(8.5 * dpi)
    height_px = int(11 * dpi)
    pixels_per_page = width_px * height_px
    
    # Memory overhead factor accounts for:
    # - Image data (3 bytes per pixel for RGB)
    # - Processing overhead (2-3x for intermediate buffers)
    # - Model memory (constant ~100-200MB)
    bytes_per_page = pixels_per_page * 3 * 2.5  # 2.5x overhead factor
    mb_per_page = bytes_per_page / (1024 ** 2)
    gb_per_page = mb_per_page / 1024

    total_memory_needed = gb_per_page * batch_size

    if use_gpu and gpu_memory_gb > 0:
        # Check VRAM constraints
        # Reserve 20% for model and system overhead
        available_vram = gpu_memory_gb * 0.8
        
        if total_memory_needed > available_vram:
            return False, (
                f"DPI {dpi} with batch_size {batch_size} requires ~{total_memory_needed:.2f}GB VRAM, "
                f"but only {available_vram:.2f}GB available (after overhead). "
                f"Reduce DPI to {int(dpi * 0.7)} or set batch_size=1"
            )
        elif total_memory_needed > available_vram * 0.85:
            # Close to limit - warn but allow
            return True, (
                f"DPI {dpi} with batch_size {batch_size} uses ~{total_memory_needed:.2f}GB VRAM "
                f"(~{(total_memory_needed/gpu_memory_gb)*100:.0f}% of {gpu_memory_gb:.1f}GB). "
                f"Processing may be slower due to memory pressure."
            )
    else:
        # Check system RAM constraints
        available_ram = system_memory_gb * 0.5  # Use up to 50% of system RAM
        
        if total_memory_needed > available_ram:
            return False, (
                f"DPI {dpi} with batch_size {batch_size} requires ~{total_memory_needed:.2f}GB RAM, "
                f"but only {available_ram:.2f}GB available (50% of {system_memory_gb:.1f}GB). "
                f"Reduce DPI to {int(dpi * 0.7)} or set batch_size=1"
            )
        elif total_memory_needed > available_ram * 0.85:
            return True, (
                f"DPI {dpi} with batch_size {batch_size} uses ~{total_memory_needed:.2f}GB RAM "
                f"(~{(total_memory_needed/(system_memory_gb*0.5))*100:.0f}% of available). "
                f"Processing may be slower."
            )

    # Safe to proceed
    return True, f"DPI {dpi} with batch_size {batch_size} is within safe memory limits (~{total_memory_needed:.2f}GB)"



class QualityPreset:
    """Predefined OCR quality presets with optimized DPI and PaddleOCR parameters"""
    
    FAST = "fast"           # Fast processing, lower quality (300 DPI)
    BALANCED = "balanced"   # Good balance (600 DPI, tuned params)
    HIGH = "high"           # High quality (600 DPI, aggressive params)
    MAXIMUM = "maximum"     # Maximum quality (1200 DPI, aggressive params)


def get_quality_preset(
    preset: str = "balanced",
    prefer_gpu: bool = True,
    model_dir: Optional[Path] = None
) -> tuple[OCRConfig, int]:
    """
    Get OCR configuration and DPI for a quality preset.

    Args:
        preset: Quality preset ("fast", "balanced", "high", "maximum")
        prefer_gpu: Whether to prefer GPU if available
        model_dir: Path to bundled models directory

    Returns:
        Tuple of (OCRConfig, target_dpi)

    Example:
        config, dpi = get_quality_preset("high", prefer_gpu=True)
        ocr = OCRService(gpu=True, config=config)
        # Render PDF at `dpi` and process with `ocr`
    """
    capabilities = detect_hardware_capabilities()
    
    # Determine if we should use GPU
    use_gpu = False
    if prefer_gpu and capabilities['gpu_available']:
        use_gpu = True
    
    # Base configuration from hardware detection
    base_config = get_default_config(
        engine="paddleocr",
        prefer_gpu=prefer_gpu,
        model_dir=model_dir
    )
    
    # Preset-specific settings
    if preset == QualityPreset.FAST:
        # Fast: 300 DPI, minimal processing
        target_dpi = 300
        engine_settings = {
            'use_space_char': True,
        }
        batch_size = get_optimal_batch_size(
            use_gpu,
            capabilities['gpu_memory_gb'],
            capabilities['system_memory_gb'],
            dpi=target_dpi
        )
    
    elif preset == QualityPreset.BALANCED:
        # Balanced: 600 DPI, tuned detection
        target_dpi = 600
        engine_settings = {
            'text_det_box_thresh': 0.4,      # More sensitive detection
            'text_det_unclip_ratio': 2.0,    # Prevent char fragmentation
            'text_rec_score_thresh': 0.4,    # Keep more results
        }
        batch_size = get_optimal_batch_size(
            use_gpu,
            capabilities['gpu_memory_gb'],
            capabilities['system_memory_gb'],
            dpi=target_dpi
        )
    
    elif preset == QualityPreset.HIGH:
        # High: 600 DPI, aggressive tuning
        target_dpi = 600
        engine_settings = {
            'text_det_box_thresh': 0.3,      # Very sensitive detection
            'text_det_unclip_ratio': 2.2,    # Maximum expansion
            'text_rec_score_thresh': 0.3,    # Accept lower confidence
            'text_det_thresh': 0.2,          # Lower detection threshold
        }
        batch_size = get_optimal_batch_size(
            use_gpu,
            capabilities['gpu_memory_gb'],
            capabilities['system_memory_gb'],
            dpi=target_dpi
        )
    
    elif preset == QualityPreset.MAXIMUM:
        # Maximum: 1200 DPI, aggressive tuning
        target_dpi = 1200
        engine_settings = {
            'text_det_box_thresh': 0.3,      # Very sensitive detection
            'text_det_unclip_ratio': 2.2,    # Maximum expansion
            'text_rec_score_thresh': 0.3,    # Accept lower confidence
            'text_det_thresh': 0.2,          # Lower detection threshold
        }
        batch_size = get_optimal_batch_size(
            use_gpu,
            capabilities['gpu_memory_gb'],
            capabilities['system_memory_gb'],
            dpi=target_dpi
        )
        
        # Warn if hardware can't handle maximum quality
        if use_gpu and capabilities['gpu_memory_gb'] < 6:
            logger.warning(
                f"Maximum quality preset (1200 DPI) with {capabilities['gpu_memory_gb']:.1f}GB VRAM "
                f"will use batch_size={batch_size}. Consider using 'high' preset for better performance."
            )
    
    else:
        # Unknown preset, use balanced
        logger.warning(f"Unknown preset '{preset}', using 'balanced'")
        return get_quality_preset("balanced", prefer_gpu, model_dir)
    
    # Create config with preset settings
    config = OCRConfig(
        engine="paddleocr",
        use_gpu=use_gpu,
        batch_size=batch_size,
        max_memory_mb=base_config.max_memory_mb,
        num_threads=base_config.num_threads,
        model_dir=model_dir,
        enable_hybrid_mode=base_config.enable_hybrid_mode,
        cpu_batch_size=base_config.cpu_batch_size,
        enable_vram_monitoring=use_gpu,
        enable_adaptive_batch_sizing=True,
        enable_engine_pooling=True,
        enable_warmup=True,
        engine_settings=engine_settings
    )
    
    logger.info(
        f"Quality preset '{preset}': DPI={target_dpi}, batch_size={batch_size}, "
        f"gpu={use_gpu}, params={len(engine_settings)} custom settings"
    )
    
    return config, target_dpi


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
