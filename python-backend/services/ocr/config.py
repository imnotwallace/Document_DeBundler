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


def get_gpu_memory_gb_directml() -> float:
    """Get GPU memory via DirectML (Windows AMD/Intel/NVIDIA GPUs)"""
    if platform.system() != "Windows":
        return 0.0
    
    try:
        import torch_directml
        # DirectML doesn't expose memory directly, try WMI if available
        try:
            import wmi
            computer = wmi.WMI()
            for gpu in computer.Win32_VideoController():
                if gpu.AdapterRAM:
                    # AdapterRAM is in bytes
                    memory_gb = gpu.AdapterRAM / (1024 ** 3)
                    if memory_gb > 0.5:  # Filter out virtual/integrated GPUs with minimal memory
                        return memory_gb
        except ImportError:
            # WMI not installed, that's okay
            pass
    except:
        pass
    
    return 0.0


def get_gpu_memory_gb_wmi() -> float:
    """Get GPU memory via WMI (Windows fallback)"""
    if platform.system() != "Windows":
        return 0.0
    
    try:
        import wmi
        computer = wmi.WMI()
        for gpu in computer.Win32_VideoController():
            if gpu.AdapterRAM:
                memory_gb = gpu.AdapterRAM / (1024 ** 3)
                if memory_gb > 0.5:  # Filter out virtual/integrated GPUs
                    return memory_gb
    except:
        pass
    
    return 0.0


def get_gpu_memory_gb_nvidia_smi() -> float:
    """Get GPU memory via nvidia-smi command (Linux/Windows NVIDIA fallback)"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Output is in MiB
            memory_mib = float(result.stdout.strip().split('\n')[0])
            return memory_mib / 1024  # Convert to GB
    except:
        pass
    
    return 0.0


def get_gpu_memory_gb_all_methods() -> float:
    """Try all methods to detect GPU memory, return first successful result"""
    # Try CUDA first (most accurate for NVIDIA)
    memory = get_gpu_memory_gb()
    if memory > 0:
        logger.info(f"Detected {memory:.2f}GB GPU memory via CUDA")
        return memory
    
    # Try DirectML/WMI for Windows
    if platform.system() == "Windows":
        memory = get_gpu_memory_gb_directml()
        if memory > 0:
            logger.info(f"Detected {memory:.2f}GB GPU memory via DirectML/WMI")
            return memory
        
        memory = get_gpu_memory_gb_wmi()
        if memory > 0:
            logger.info(f"Detected {memory:.2f}GB GPU memory via WMI")
            return memory
    
    # Try nvidia-smi as last resort
    memory = get_gpu_memory_gb_nvidia_smi()
    if memory > 0:
        logger.info(f"Detected {memory:.2f}GB GPU memory via nvidia-smi")
        return memory
    
    logger.warning("Could not detect GPU memory via any method")
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
    dpi: int = 300,
    model_type: str = "mobile"
) -> int:
    """
    Calculate optimal batch size based on available hardware and DPI.

    Optimized for 4GB VRAM systems with improved utilization.
    Accounts for DPI impact on memory usage (higher DPI = larger images = smaller batches).
    Accounts for PP-OCRv5 model type (server models use 4-5x more VRAM than mobile).

    Args:
        use_gpu: Whether GPU will be used
        gpu_memory_gb: Available GPU memory in GB
        system_memory_gb: System RAM in GB (auto-detected if None)
        dpi: Target DPI for rendering (default: 300)
        model_type: PP-OCRv5 model type ("mobile" or "server", default: "mobile")

    Returns:
        Recommended batch size (adjusted for DPI and model type)
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
    
    # Apply model type multiplier for server models
    # PP-OCRv5 server models use ~4-5x more VRAM than mobile models
    # Server: ~160MB file, ~400-600MB VRAM during inference
    # Mobile: ~20MB file, ~100-150MB VRAM during inference
    if model_type == "server":
        # Reduce batch size by ~25% for server models
        multiplier = 0.75
        adjusted_batch = max(1, int(adjusted_batch * multiplier))
        logger.debug(
            f"Applied server model multiplier ({multiplier:.0%}) to batch size: "
            f"{int(adjusted_batch / multiplier)} → {adjusted_batch}"
        )
    
    # Log warning for very high DPI
    if dpi >= 1200 and adjusted_batch == 1:
        logger.warning(
            f"High DPI ({dpi}) requires batch_size=1 for {gpu_memory_gb}GB VRAM. "
            f"Consider reducing DPI or processing in smaller chunks."
        )
    
    return adjusted_batch


def calculate_batch_size_for_dpi(
    target_dpi: int,
    use_gpu: bool,
    gpu_memory_gb: float = 0,
    system_memory_gb: Optional[float] = None,
    preprocessing_enabled: bool = True,
    model_type: str = "mobile"
) -> int:
    """
    Calculate batch size needed to achieve target DPI.
    
    DPI-FIRST approach: Calculate how many pages we can fit in memory at the requested DPI,
    rather than calculating DPI based on a fixed batch size.
    
    Memory calculation:
    - Image size at DPI: (8.5*DPI) x (11*DPI) pixels
    - RGB image: width * height * 3 bytes
    - Grayscale: width * height * 1 byte
    - Preprocessing: 2 images per page (deskewed RGB + preprocessed grayscale)
    - Without preprocessing: 1 image per page (original RGB)
    
    Args:
        target_dpi: User-requested target DPI
        use_gpu: Whether GPU will be used
        gpu_memory_gb: Available GPU memory in GB
        system_memory_gb: System RAM in GB (auto-detected if None)
        preprocessing_enabled: Whether intelligent preprocessing is enabled
        model_type: PP-OCRv5 model type ("mobile" or "server")
    
    Returns:
        Maximum batch size that fits in memory at target DPI (minimum 1)
    """
    if system_memory_gb is None:
        system_memory_gb = get_system_memory_gb()
    
    # Calculate memory per page at target DPI
    # Letter page: 8.5" x 11"
    width_px = int(8.5 * target_dpi)
    height_px = int(11 * target_dpi)
    pixels_per_page = width_px * height_px
    
    # Memory footprint calculation
    if preprocessing_enabled:
        # 2 images: deskewed RGB + preprocessed grayscale
        rgb_bytes = pixels_per_page * 3  # Deskewed RGB
        gray_bytes = pixels_per_page * 1  # Preprocessed grayscale
        bytes_per_page = rgb_bytes + gray_bytes
    else:
        # 1 image: original RGB
        bytes_per_page = pixels_per_page * 3
    
    # Add processing overhead (buffers, intermediate arrays, etc.)
    # PaddleOCR needs extra memory for:
    # - Input preprocessing (normalization, padding)
    # - Model inference (activations, gradients)
    # - Output postprocessing (NMS, text decoding)
    # Overhead factor: ~2.5x for mobile models, ~4x for server models
    overhead_factor = 4.0 if model_type == "server" else 2.5
    bytes_per_page_with_overhead = int(bytes_per_page * overhead_factor)
    
    mb_per_page = bytes_per_page_with_overhead / (1024 ** 2)
    
    # Calculate available memory for batching
    if use_gpu and gpu_memory_gb > 0:
        # GPU processing
        # Reserve memory for:
        # - PaddleOCR model: ~150MB (mobile) or ~600MB (server)
        # - System overhead: ~200MB
        model_memory_mb = 600 if model_type == "server" else 150
        system_overhead_mb = 200
        reserved_mb = model_memory_mb + system_overhead_mb
        
        available_mb = (gpu_memory_gb * 1024) - reserved_mb
        
        # Use 85% of available memory to avoid OOM
        usable_mb = available_mb * 0.85
    else:
        # CPU processing
        # Use 50% of system RAM for batch processing
        available_mb = system_memory_gb * 1024 * 0.5
        usable_mb = available_mb
    
    # Calculate batch size
    batch_size = int(usable_mb / mb_per_page)
    batch_size = max(1, batch_size)  # Minimum 1 page
    
    # Log calculation details
    logger.info(
        f"DPI-first batch sizing: target_dpi={target_dpi}, "
        f"memory_per_page={mb_per_page:.1f}MB, "
        f"available_memory={usable_mb:.1f}MB, "
        f"batch_size={batch_size}"
    )
    
    # Warn if batch size is very small (slow processing)
    if batch_size == 1:
        logger.warning(
            f"Target DPI ({target_dpi}) requires batch_size=1 "
            f"({mb_per_page:.1f}MB per page, {usable_mb:.1f}MB available). "
            f"Processing will be slower but will achieve target quality."
        )
    elif batch_size <= 5:
        logger.info(
            f"Target DPI ({target_dpi}) allows small batch_size={batch_size}. "
            f"Consider reducing DPI for faster processing if quality is acceptable."
        )
    
    return batch_size


def detect_hardware_capabilities() -> Dict[str, Any]:
    """
    Detect available hardware capabilities.

    Returns:
        Dictionary with hardware information
    """
    cuda_available = detect_gpu_cuda()
    directml_available = detect_gpu_directml()
    gpu_available = cuda_available or directml_available

    # Use comprehensive GPU memory detection
    gpu_memory = get_gpu_memory_gb_all_methods() if gpu_available else 0.0
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
    target_quality: str = "balanced",
    preprocessing_enabled: bool = True,
    model_type: str = "mobile"
) -> int:
    """
    Calculate adaptive DPI based on available hardware and quality target.
    
    IMPORTANT: Automatically adjusts for preprocessing memory overhead.
    - Without preprocessing: 1 image per page (original)
    - With preprocessing: 2 images per page (deskewed + fully preprocessed)
    - PP-OCRv5 server models use 4-5x more VRAM than mobile models
    
    Args:
        use_gpu: Whether GPU is being used
        gpu_memory_gb: Available GPU memory in GB
        system_memory_gb: System RAM in GB (auto-detected if None)
        target_quality: Quality target ("low", "balanced", "high", "max", "ultra", "maximum")
        preprocessing_enabled: Whether intelligent preprocessing is enabled (default: True)
        model_type: PP-OCRv5 model type ("mobile" or "server", default: "mobile")

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

    # Apply preprocessing memory multiplier if enabled
    if preprocessing_enabled:
        # Memory footprint increases from 1 image to 2 images per page
        # OPTIMIZED: Original images eliminated, only deskewed + preprocessed kept
        # Deskewed (RGB): ~7.5 MB, Preprocessed (grayscale): ~2.5 MB = 10 MB total
        # Without preprocessing: ~7.5 MB (original only)
        # Reduction factor: 7.5 / 10 = 0.75 (~25% reduction)
        multiplier = 0.75
        max_dpi = int(max_dpi * multiplier)
        
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            f"Adjusted max DPI for 2-image preprocessing pipeline: "
            f"{int(max_dpi / multiplier)} → {max_dpi} ({multiplier:.0%} multiplier)"
        )

    # Apply model type multiplier for server models
    if model_type == "server":
        # PP-OCRv5 server models use ~4-5x more VRAM than mobile models
        # Reduce max DPI by ~25% for server models
        multiplier = 0.75
        max_dpi = int(max_dpi * multiplier)
        
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            f"Adjusted max DPI for server model type: "
            f"{int(max_dpi / multiplier)} → {max_dpi} ({multiplier:.0%} multiplier)"
        )

    # Return the minimum of requested and maximum allowed
    recommended_dpi = min(base_dpi, max_dpi)
    
    # Log warnings for high DPI with limited memory
    if base_dpi > recommended_dpi:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Requested DPI ({base_dpi}) reduced to {recommended_dpi} due to memory constraints. "
            f"Available: {'GPU ' + str(gpu_memory_gb) + 'GB VRAM' if use_gpu else 'CPU ' + str(system_memory_gb) + 'GB RAM'}"
            f"{' (2-image preprocessing enabled)' if preprocessing_enabled else ''}"
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
        # Phase 3 finding: use_space_char doesn't exist in PaddleOCR 3.3.1
        target_dpi = 300
        engine_settings = {
            # No custom parameters for FAST preset - use PaddleOCR defaults
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
        # High: 600 DPI, VERY AGGRESSIVE detection (Phase 3 tuning)
        # Goal: Detect ALL text regions, even faint/low-contrast text
        target_dpi = 600
        engine_settings = {
            # VERY AGGRESSIVE detection parameters to catch all text
            'text_det_box_thresh': 0.2,      # Very low - detect more boxes (was 0.35)
            'text_det_unclip_ratio': 2.8,    # Very high - expand boxes more (was 1.9)
            'text_rec_score_thresh': 0.4,    # Lower - accept more recognition results (was 0.5)
            'text_det_thresh': 0.15,         # Very low - detect faint text (was 0.2)
        }
        batch_size = get_optimal_batch_size(
            use_gpu,
            capabilities['gpu_memory_gb'],
            capabilities['system_memory_gb'],
            dpi=target_dpi
        )
    
    elif preset == QualityPreset.MAXIMUM:
        # Maximum: 1200 DPI, balanced tuning (Phase 2 optimized)
        # Phase 3 finding: Spacing parameters don't exist in PaddleOCR 3.3.1
        target_dpi = 1200
        engine_settings = {
            # Core detection parameters (Phase 2 optimized - VALID params only)
            'text_det_box_thresh': 0.35,     # Balanced detection (reduced from 0.28)
            'text_det_unclip_ratio': 1.9,    # Moderate expansion (reduced from 2.2)
            'text_rec_score_thresh': 0.5,    # Filter low-confidence garbage
            'text_det_thresh': 0.2,          # Lower pixel detection threshold
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


def detect_model_type(model_dir: Optional[Path] = None) -> str:
    """
    Detect whether mobile or server PP-OCRv5 models are being used.
    
    Detection strategy:
    1. Check recognition model file size (rec is much larger difference than det)
    2. Mobile models: rec ~5-10MB total (~20MB combined det+rec+cls)
    3. Server models: rec ~100-140MB total (~160MB combined det+rec+cls)
    
    Args:
        model_dir: Path to models directory (auto-detected if None)
    
    Returns:
        "mobile" or "server"
    """
    if model_dir is None:
        model_dir = get_model_directory()
    
    # Check for recognition model file
    rec_model_path = model_dir / "rec" / "inference.pdmodel"
    
    if not rec_model_path.exists():
        # No bundled models, PaddleOCR will auto-download mobile by default
        logger.info("No bundled models found, assuming mobile models (PaddleOCR default)")
        return "mobile"
    
    try:
        # Get model file size in MB
        model_size_bytes = rec_model_path.stat().st_size
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Threshold: 20MB (mobile models are <20MB, server models are >20MB)
        if model_size_mb < 20:
            logger.info(f"Detected mobile model ({model_size_mb:.1f}MB recognition model)")
            return "mobile"
        else:
            logger.info(f"Detected server model ({model_size_mb:.1f}MB recognition model)")
            return "server"
    
    except Exception as e:
        logger.warning(f"Failed to detect model type: {e}, assuming mobile")
        return "mobile"
