"""
Download LLM Models for Document De-Bundler

This script downloads optimized GGUF models for local LLM inference.
Models are saved to python-backend/models/llm/ for bundling with the application.

Target Hardware: 4GB VRAM + 16GB RAM
Primary Model: Phi-3 Mini (Q4_K_M quantization, ~2.3GB)
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_models_dir() -> Path:
    """Get the models/llm directory."""
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models" / "llm"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def check_existing_model(model_path: Path) -> bool:
    """Check if model already exists and is valid."""
    if not model_path.exists():
        return False

    # Check file size (should be > 1GB for GGUF models)
    size_gb = model_path.stat().st_size / (1024**3)
    if size_gb < 1.0:
        logger.warning(f"Model file too small ({size_gb:.2f}GB), re-downloading...")
        return False

    logger.info(f"Found existing model: {model_path.name} ({size_gb:.2f}GB)")
    return True


def download_from_huggingface(
    repo_id: str,
    filename: str,
    output_path: Path,
    resume: bool = True
) -> bool:
    """
    Download a model file from HuggingFace.

    Args:
        repo_id: HuggingFace repo (e.g., "microsoft/Phi-3-mini-4k-instruct-gguf")
        filename: Model filename in the repo
        output_path: Local path to save the file
        resume: Whether to resume interrupted downloads

    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError

        logger.info(f"Downloading {filename} from {repo_id}...")
        logger.info(f"This may take several minutes depending on your connection...")

        # Download with progress bar
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=None,  # Use default cache
            resume_download=resume,
            local_dir=output_path.parent,
            local_dir_use_symlinks=False  # Copy file instead of symlink
        )

        # If downloaded to a different location, move it
        downloaded_path = Path(downloaded_path)
        if downloaded_path != output_path:
            if output_path.exists():
                output_path.unlink()
            downloaded_path.rename(output_path)

        size_gb = output_path.stat().st_size / (1024**3)
        logger.info(f"✓ Downloaded successfully: {output_path.name} ({size_gb:.2f}GB)")
        return True

    except HfHubHTTPError as e:
        logger.error(f"Download failed: {e}")
        logger.error("Make sure you have internet connection and the model exists")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        return False


def download_phi3_mini() -> bool:
    """
    Download Phi-3 Mini Q4 model (recommended for 4GB VRAM).

    Model Details:
    - Size: ~2.4GB
    - Quantization: Q4 (balanced quality/size)
    - VRAM Usage: ~2.3GB with 28 GPU layers
    - Context: 4096 tokens
    """
    models_dir = get_models_dir()
    model_path = models_dir / "phi-3-mini-4k-instruct-q4.gguf"

    if check_existing_model(model_path):
        return True

    logger.info("=" * 60)
    logger.info("Downloading Phi-3 Mini (Q4)")
    logger.info("=" * 60)
    logger.info("Model: microsoft/Phi-3-mini-4k-instruct-gguf")
    logger.info("Size: ~2.4GB")
    logger.info("Optimized for: 4GB VRAM + 16GB RAM")
    logger.info("=" * 60)

    return download_from_huggingface(
        repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
        filename="Phi-3-mini-4k-instruct-q4.gguf",
        output_path=model_path
    )


def download_gemma2_2b() -> bool:
    """
    Download Gemma 2 2B Q4_K_M model (lighter alternative for low VRAM).

    Model Details:
    - Size: ~1.5GB
    - Quantization: Q4_K_M
    - VRAM Usage: ~1.5GB with 20 GPU layers
    - Context: 4096 tokens
    """
    models_dir = get_models_dir()
    model_path = models_dir / "gemma-2-2b-it-q4_k_m.gguf"

    if check_existing_model(model_path):
        return True

    logger.info("=" * 60)
    logger.info("Downloading Gemma 2 2B (Q4_K_M) - Optional")
    logger.info("=" * 60)
    logger.info("Model: google/gemma-2-2b-it-gguf")
    logger.info("Size: ~1.5GB")
    logger.info("Optimized for: 2GB VRAM systems")
    logger.info("=" * 60)

    return download_from_huggingface(
        repo_id="google/gemma-2-2b-it-gguf",
        filename="gemma-2-2b-it-q4_k_m.gguf",
        output_path=model_path
    )


def verify_models() -> Dict[str, bool]:
    """
    Verify which models are properly installed.

    Returns:
        Dictionary with model availability
    """
    models_dir = get_models_dir()

    result = {
        'phi3_mini': False,
        'gemma2_2b': False
    }

    # Check Phi-3 Mini
    phi3_path = models_dir / "phi-3-mini-4k-instruct-q4.gguf"
    if phi3_path.exists() and phi3_path.stat().st_size > 1024**3:
        result['phi3_mini'] = True
        logger.info(f"✓ Phi-3 Mini found: {phi3_path}")

    # Check Gemma 2 2B
    gemma_path = models_dir / "gemma-2-2b-it-q4_k_m.gguf"
    if gemma_path.exists() and gemma_path.stat().st_size > 1024**3:
        result['gemma2_2b'] = True
        logger.info(f"✓ Gemma 2 2B found: {gemma_path}")

    return result


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("Document De-Bundler - LLM Model Downloader")
    print("=" * 60)
    print()

    # Check for huggingface_hub
    try:
        import huggingface_hub
        logger.info(f"Using huggingface_hub v{huggingface_hub.__version__}")
    except ImportError:
        logger.error("ERROR: huggingface-hub not installed")
        logger.error("Please install it: pip install huggingface-hub")
        sys.exit(1)

    models_dir = get_models_dir()
    logger.info(f"Models directory: {models_dir}")
    print()

    # Ask user which models to download
    print("Available models:")
    print("  1. Phi-3 Mini Q4_K_M (recommended for 4GB VRAM) - ~2.3GB")
    print("  2. Gemma 2 2B Q4_K_M (lighter, for 2GB VRAM) - ~1.5GB")
    print("  3. Both models")
    print()

    choice = input("Which model(s) to download? [1/2/3, default=1]: ").strip() or "1"

    success = True

    if choice in ["1", "3"]:
        print()
        success &= download_phi3_mini()

    if choice in ["2", "3"]:
        print()
        success &= download_gemma2_2b()

    # Verify downloads
    print()
    print("=" * 60)
    print("Verification")
    print("=" * 60)
    models = verify_models()

    if models['phi3_mini']:
        print("✓ Phi-3 Mini ready for use")
    if models['gemma2_2b']:
        print("✓ Gemma 2 2B ready for use")

    if not any(models.values()):
        print("✗ No models successfully downloaded")
        success = False

    print()
    if success:
        print("=" * 60)
        print("✓ Download complete!")
        print("=" * 60)
        print()
        print("Models are ready for use with Document De-Bundler.")
        print("The application will automatically detect and use these bundled models.")
        print()
        print("Next steps:")
        print("  1. Run the application: npm run tauri:dev")
        print("  2. Enable AI features in processing options")
        print()
    else:
        print("=" * 60)
        print("✗ Download failed")
        print("=" * 60)
        print()
        print("Some models failed to download. Check the errors above.")
        print("You can:")
        print("  - Check your internet connection")
        print("  - Try again later")
        print("  - Download models manually from HuggingFace")
        print()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
