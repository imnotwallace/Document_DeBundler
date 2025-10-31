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
