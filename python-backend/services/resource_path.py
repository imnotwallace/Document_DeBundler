"""
Resource Path Resolution Utility

Handles path resolution for bundled resources in both development and production modes.
When running as a bundled Tauri app, resources are in a different location than during development.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def get_base_path() -> Path:
    """
    Get the base path for the application.

    Returns:
        Path to the base directory (different in dev vs production)

    In development:
        - Returns the python-backend directory

    In production (bundled):
        - Returns the directory where the Python script is running from
        - Tauri bundles resources relative to the executable
    """
    if getattr(sys, 'frozen', False):
        # Running in a PyInstaller bundle or similar
        base_path = Path(sys.executable).parent
    else:
        # Running in development mode
        # Assume this file is in python-backend/services/
        base_path = Path(__file__).parent.parent

    logger.debug(f"Base path resolved to: {base_path}")
    return base_path


def get_bin_path() -> Path:
    """
    Get the path to the bin directory containing bundled binaries.

    Returns:
        Path to bin/ directory
    """
    base_path = get_base_path()
    bin_path = base_path / "bin"

    logger.debug(f"Bin path resolved to: {bin_path}")
    return bin_path


def get_tesseract_path() -> Optional[Path]:
    """
    Get the path to the bundled Tesseract executable.

    Returns:
        Path to tesseract.exe if found, None otherwise
    """
    # Try bundled version first
    bundled_tesseract = get_bin_path() / "tesseract" / "tesseract.exe"

    if bundled_tesseract.exists():
        logger.info(f"Found bundled Tesseract at: {bundled_tesseract}")
        return bundled_tesseract

    logger.warning(f"Bundled Tesseract not found at: {bundled_tesseract}")

    # In development, might use system Tesseract (let pytesseract auto-detect)
    return None


def get_tessdata_path() -> Optional[Path]:
    """
    Get the path to the tessdata directory containing language data files.

    Returns:
        Path to tessdata/ directory if found, None otherwise
    """
    # Try bundled version first
    bundled_tessdata = get_bin_path() / "tesseract" / "tessdata"

    if bundled_tessdata.exists() and bundled_tessdata.is_dir():
        logger.info(f"Found bundled tessdata at: {bundled_tessdata}")
        return bundled_tessdata

    logger.warning(f"Bundled tessdata not found at: {bundled_tessdata}")

    # In development, might use system tessdata (let pytesseract auto-detect)
    return None


def is_production_mode() -> bool:
    """
    Detect if running in production (bundled) mode.

    Returns:
        True if running as bundled app, False if in development
    """
    return getattr(sys, 'frozen', False)


def setup_tesseract_environment() -> Dict[str, Optional[str]]:
    """
    Configure environment variables for Tesseract.
    Sets both the executable path and TESSDATA_PREFIX.

    Returns:
        Dictionary with configuration:
        - 'tesseract_cmd': Path to tesseract executable (or None for system default)
        - 'tessdata_prefix': Path to tessdata directory (or None for system default)
        - 'mode': 'bundled' or 'system'
    """
    config = {
        'tesseract_cmd': None,
        'tessdata_prefix': None,
        'mode': 'system'
    }

    # Check for bundled Tesseract
    tesseract_path = get_tesseract_path()
    tessdata_path = get_tessdata_path()

    if tesseract_path and tesseract_path.exists():
        config['tesseract_cmd'] = str(tesseract_path.absolute())
        config['mode'] = 'bundled'
        logger.info(f"Using bundled Tesseract: {config['tesseract_cmd']}")

        # Set tessdata path if available
        if tessdata_path and tessdata_path.exists():
            config['tessdata_prefix'] = str(tessdata_path.absolute())

            # Also set environment variable (some Tesseract versions need this)
            os.environ['TESSDATA_PREFIX'] = config['tessdata_prefix']
            logger.info(f"Set TESSDATA_PREFIX to: {config['tessdata_prefix']}")
        else:
            logger.warning("Bundled Tesseract found but tessdata directory missing!")
    else:
        logger.info("Using system Tesseract (if available)")
        config['mode'] = 'system'

    return config


def verify_tesseract_setup() -> tuple[bool, str]:
    """
    Verify that Tesseract is properly configured.

    Returns:
        Tuple of (success: bool, message: str)
    """
    config = setup_tesseract_environment()

    if config['mode'] == 'bundled':
        tesseract_exe = Path(config['tesseract_cmd'])
        tessdata_dir = Path(config['tessdata_prefix']) if config['tessdata_prefix'] else None

        # Check executable exists
        if not tesseract_exe.exists():
            return False, f"Tesseract executable not found: {tesseract_exe}"

        # Check tessdata exists
        if not tessdata_dir or not tessdata_dir.exists():
            return False, f"Tessdata directory not found: {tessdata_dir}"

        # Check for at least one language file
        traineddata_files = list(tessdata_dir.glob("*.traineddata"))
        if not traineddata_files:
            return False, f"No .traineddata files found in: {tessdata_dir}"

        return True, f"Bundled Tesseract configured successfully ({len(traineddata_files)} languages)"

    else:
        # System Tesseract - will be verified by pytesseract
        return True, "Using system Tesseract (will be verified on initialization)"


def get_embedding_models_dir() -> Path:
    """
    Get the path to the embedding models directory.

    Returns:
        Path to models/embeddings/ directory
    """
    base_path = get_base_path()
    models_dir = base_path / "models" / "embeddings"

    logger.debug(f"Embedding models directory resolved to: {models_dir}")
    return models_dir


def get_text_embedding_path() -> Optional[Path]:
    """
    Get the path to the bundled Nomic text embedding model.

    Returns:
        Path to text model directory if found, None otherwise
    """
    models_dir = get_embedding_models_dir()
    text_model_path = models_dir / "text"

    if text_model_path.exists() and (text_model_path / "config.json").exists():
        logger.info(f"Found bundled text embedding model at: {text_model_path}")
        return text_model_path

    logger.debug(f"Bundled text embedding model not found at: {text_model_path}")
    return None


def get_vision_embedding_path() -> Optional[Path]:
    """
    Get the path to the bundled Nomic vision embedding model.

    Returns:
        Path to vision model directory if found, None otherwise
    """
    models_dir = get_embedding_models_dir()
    vision_model_path = models_dir / "vision"

    if vision_model_path.exists() and (vision_model_path / "config.json").exists():
        logger.info(f"Found bundled vision embedding model at: {vision_model_path}")
        return vision_model_path

    logger.debug(f"Bundled vision embedding model not found at: {vision_model_path}")
    return None


def verify_embedding_models() -> Dict[str, bool]:
    """
    Verify which embedding models are properly installed.

    Returns:
        Dictionary with model availability:
        - 'text': True if text model is available
        - 'vision': True if vision model is available
    """
    result = {
        'text': False,
        'vision': False
    }

    text_path = get_text_embedding_path()
    if text_path and text_path.exists():
        # Check for essential files
        if (text_path / "config.json").exists():
            result['text'] = True
            logger.info("Text embedding model verified")

    vision_path = get_vision_embedding_path()
    if vision_path and vision_path.exists():
        # Check for essential files
        if (vision_path / "config.json").exists():
            result['vision'] = True
            logger.info("Vision embedding model verified")

    return result


def get_llm_models_dir() -> Path:
    """
    Get the path to the LLM models directory.

    Returns:
        Path to models/llm/ directory
    """
    base_path = get_base_path()
    models_dir = base_path / "models" / "llm"

    logger.debug(f"LLM models directory resolved to: {models_dir}")
    return models_dir


def get_phi3_mini_path() -> Optional[Path]:
    """
    Get the path to the bundled Phi-3 Mini model.

    Returns:
        Path to Phi-3 Mini GGUF file if found, None otherwise
    """
    models_dir = get_llm_models_dir()
    model_path = models_dir / "phi-3-mini-4k-instruct-q4.gguf"

    if model_path.exists() and model_path.stat().st_size > 1024**3:  # > 1GB
        logger.info(f"Found bundled Phi-3 Mini at: {model_path}")
        return model_path

    logger.debug(f"Bundled Phi-3 Mini not found at: {model_path}")
    return None


def get_gemma2_2b_path() -> Optional[Path]:
    """
    Get the path to the bundled Gemma 2 2B model.

    Returns:
        Path to Gemma 2 2B GGUF file if found, None otherwise
    """
    models_dir = get_llm_models_dir()
    model_path = models_dir / "gemma-2-2b-it-q4_k_m.gguf"

    if model_path.exists() and model_path.stat().st_size > 1024**3:  # > 1GB
        logger.info(f"Found bundled Gemma 2 2B at: {model_path}")
        return model_path

    logger.debug(f"Bundled Gemma 2 2B not found at: {model_path}")
    return None


def verify_llm_models() -> Dict[str, bool]:
    """
    Verify which LLM models are properly installed.

    Returns:
        Dictionary with model availability:
        - 'phi3_mini': True if Phi-3 Mini model is available
        - 'gemma2_2b': True if Gemma 2 2B model is available
    """
    result = {
        'phi3_mini': False,
        'gemma2_2b': False
    }

    phi3_path = get_phi3_mini_path()
    if phi3_path and phi3_path.exists():
        result['phi3_mini'] = True
        logger.info("Phi-3 Mini model verified")

    gemma_path = get_gemma2_2b_path()
    if gemma_path and gemma_path.exists():
        result['gemma2_2b'] = True
        logger.info("Gemma 2 2B model verified")

    return result


def get_llama_cpp_bin_dir() -> Path:
    """
    Get the path to the llama.cpp binaries directory.

    Returns:
        Path to bin/llama-cpp/ directory
    """
    bin_path = get_bin_path()
    llama_cpp_dir = bin_path / "llama-cpp"

    logger.debug(f"llama.cpp bin directory resolved to: {llama_cpp_dir}")
    return llama_cpp_dir


def get_llama_cpp_binary_path() -> Optional[Path]:
    """
    Get the path to the llama-cpp binary for the current platform.

    Returns:
        Path to llama-server executable if found, None otherwise
    """
    import platform

    llama_cpp_dir = get_llama_cpp_bin_dir()
    system = platform.system().lower()

    # Determine platform-specific path
    if system == "windows":
        binary_path = llama_cpp_dir / "windows" / "llama-server.exe"
    elif system == "linux":
        binary_path = llama_cpp_dir / "linux" / "llama-server"
    elif system == "darwin":  # macOS
        binary_path = llama_cpp_dir / "macos" / "llama-server"
    else:
        logger.warning(f"Unsupported platform for llama.cpp binary: {system}")
        return None

    if binary_path.exists():
        logger.info(f"Found llama.cpp binary at: {binary_path}")
        return binary_path

    logger.debug(f"llama.cpp binary not found at: {binary_path}")
    return None


def verify_llama_cpp_binary() -> tuple[bool, str]:
    """
    Verify that llama.cpp binary is properly configured.

    Returns:
        Tuple of (success: bool, message: str)
    """
    binary_path = get_llama_cpp_binary_path()

    if not binary_path:
        return False, "llama.cpp binary not found for this platform"

    if not binary_path.exists():
        return False, f"llama.cpp binary does not exist: {binary_path}"

    # Check if executable
    import stat
    if not (binary_path.stat().st_mode & stat.S_IXUSR):
        return False, f"llama.cpp binary is not executable: {binary_path}"

    return True, f"llama.cpp binary ready: {binary_path}"


# Convenience function for other modules
def get_resource_path(relative_path: str) -> Path:
    """
    Get absolute path to a bundled resource.

    Args:
        relative_path: Path relative to python-backend/ (e.g., "bin/tesseract/tesseract.exe")

    Returns:
        Absolute path to the resource
    """
    base_path = get_base_path()
    return base_path / relative_path
