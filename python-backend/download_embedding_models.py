#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nomic Embedding Model Downloader
Automatically downloads Nomic Embed v1.5 models for offline operation.

IMPORTANT: Run this script from within the activated virtual environment:
  Windows: venv\\Scripts\\activate && python download_embedding_models.py
  Unix/Mac: source venv/bin/activate && python download_embedding_models.py
"""

import os
import sys
from pathlib import Path

# Set UTF-8 encoding for console output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from typing import List, Tuple, Optional

# Model definitions: (name, model_id, target_directory)
MODELS = [
    (
        "Nomic Embed Text v1.5",
        "nomic-ai/nomic-embed-text-v1.5",
        "text",
        "~550MB - Text embeddings for document analysis"
    ),
    (
        "Nomic Embed Vision v1.5",
        "nomic-ai/nomic-embed-vision-v1.5",
        "vision",
        "~600MB - Visual embeddings for image understanding"
    ),
]


def check_virtual_env():
    """Check if running in a virtual environment and warn if not"""
    in_venv = (
        hasattr(sys, 'real_prefix') or  # virtualenv
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or  # venv
        os.environ.get('VIRTUAL_ENV') is not None  # environment variable
    )

    if not in_venv:
        print("WARNING: You don't appear to be in a virtual environment!")
        print("It's recommended to run this script from within the venv:")
        print()
        if os.name == 'nt':  # Windows
            print("  .venv\\Scripts\\activate")
            print("  python download_embedding_models.py")
        else:  # Unix/Mac
            print("  source .venv/bin/activate")
            print("  python download_embedding_models.py")
        print()
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response not in ('y', 'yes'):
            print("Cancelled by user.")
            return False

    return True


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import huggingface_hub
        return True
    except ImportError:
        print("ERROR: Required dependency 'huggingface_hub' not found.")
        print("\nThis means project dependencies haven't been installed yet.")
        print("\nPlease install all dependencies first:")
        print()
        if os.name == 'nt':  # Windows
            print("  cd python-backend")
            print("  .venv\\Scripts\\activate")
            print("  uv pip sync requirements.txt")
            print("  # OR: pip install -r requirements.txt")
        else:  # Unix/Mac
            print("  cd python-backend")
            print("  source .venv/bin/activate")
            print("  uv pip sync requirements.txt")
            print("  # OR: pip install -r requirements.txt")
        print()
        print("Then run this script again:")
        print("  python download_embedding_models.py")
        return False


def get_models_dir() -> Path:
    """Get the embedding models directory path"""
    # This script should be in python-backend/
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models" / "embeddings"

    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created models directory: {models_dir}")

    return models_dir


def check_existing_models(models_dir: Path) -> List[str]:
    """Check which models are already installed"""
    existing = []

    for name, _, target_dir, _ in MODELS:
        model_path = models_dir / target_dir
        # Check if model directory exists and contains files
        if model_path.exists() and any(model_path.iterdir()):
            # Check for essential files (config.json is always present)
            config_file = model_path / "config.json"
            if config_file.exists():
                existing.append(target_dir)

    return existing


def download_model(
    name: str,
    model_id: str,
    models_dir: Path,
    target_dir: str,
    description: str
) -> bool:
    """
    Download a single model from HuggingFace Hub.

    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import snapshot_download
        from tqdm import tqdm

        print(f"\n[{name}]")
        print(f"  Model ID: {model_id}")
        print(f"  Description: {description}")

        # Target path for the model
        target_path = models_dir / target_dir
        target_path.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading to: {target_path}")
        print(f"  This may take a few minutes...")

        # Download model with progress bar
        try:
            downloaded_path = snapshot_download(
                repo_id=model_id,
                local_dir=str(target_path),
                local_dir_use_symlinks=False,  # Copy files instead of symlinks
                resume_download=True,  # Resume if interrupted
                tqdm_class=tqdm  # Use tqdm for progress
            )

            # Verify download
            config_file = target_path / "config.json"
            model_file = target_path / "model.safetensors"

            if config_file.exists():
                # Calculate total size
                total_size = sum(
                    f.stat().st_size for f in target_path.rglob('*') if f.is_file()
                )
                size_mb = total_size / (1024 * 1024)

                print(f"  ✓ Successfully downloaded to {target_path}")
                print(f"    Total size: {size_mb:.1f} MB")

                # List key files
                key_files = ["config.json", "model.safetensors", "tokenizer_config.json"]
                for key_file in key_files:
                    file_path = target_path / key_file
                    if file_path.exists():
                        file_size = file_path.stat().st_size / (1024 * 1024)
                        print(f"    - {key_file}: {file_size:.1f} MB")

                return True
            else:
                print(f"  ✗ Download incomplete - config.json not found")
                return False

        except Exception as e:
            print(f"  ✗ Download error: {e}")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def verify_installation(models_dir: Path) -> bool:
    """Verify all models are correctly installed"""
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    all_ok = True

    for name, _, target_dir, _ in MODELS:
        model_path = models_dir / target_dir
        config_file = model_path / "config.json"

        if config_file.exists():
            # Calculate size
            total_size = sum(
                f.stat().st_size for f in model_path.rglob('*') if f.is_file()
            )
            size_mb = total_size / (1024 * 1024)
            print(f"✓ {name}: OK ({size_mb:.0f} MB)")
        else:
            print(f"✗ {name}: MISSING")
            all_ok = False

    return all_ok


def show_disk_space_warning():
    """Show disk space requirements"""
    print("\n" + "="*70)
    print("DISK SPACE REQUIREMENTS")
    print("="*70)
    print("The following models will be downloaded:")
    print()

    total_size = 0
    for name, _, _, description in MODELS:
        # Extract size from description
        size_str = description.split(" - ")[0]
        print(f"  • {name}: {size_str}")
        # Parse size (rough estimate)
        if "MB" in size_str:
            size_mb = int(size_str.replace("~", "").replace("MB", "").strip())
            total_size += size_mb

    print(f"\nTotal approximate size: ~{total_size} MB ({total_size/1024:.1f} GB)")
    print("\nMake sure you have enough disk space available.")


def main():
    """Main entry point"""
    print("="*70)
    print("Nomic Embedding Model Downloader")
    print("="*70)
    print()
    print("This script downloads Nomic Embed v1.5 models for offline use.")
    print("Models will be used by the embedding service for document analysis.")
    print()

    # Check virtual environment
    if not check_virtual_env():
        return 1

    # Check dependencies
    if not check_dependencies():
        return 1

    # Show disk space requirements
    show_disk_space_warning()

    # Ask for confirmation
    print()
    response = input("Do you want to proceed with the download? (Y/n): ").strip().lower()
    if response in ('n', 'no'):
        print("Download cancelled by user.")
        return 0

    # Get models directory
    models_dir = get_models_dir()
    print(f"\nModels directory: {models_dir}")

    # Check existing models
    existing = check_existing_models(models_dir)
    if existing:
        print(f"\nExisting models found: {', '.join(existing)}")
        response = input("Do you want to re-download existing models? (y/N): ").strip().lower()
        skip_existing = response not in ('y', 'yes')
    else:
        skip_existing = False

    # Download models
    print("\n" + "="*70)
    print("DOWNLOADING MODELS")
    print("="*70)

    success_count = 0
    total_count = len(MODELS)

    for name, model_id, target_dir, description in MODELS:
        if skip_existing and target_dir in existing:
            print(f"\n[{name}]")
            print(f"  Skipping (already exists)")
            success_count += 1
            continue

        if download_model(name, model_id, models_dir, target_dir, description):
            success_count += 1

    # Verify installation
    all_ok = verify_installation(models_dir)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Successfully installed: {success_count}/{total_count} models")

    if all_ok:
        print("\n✓ All embedding models are ready for use!")
        print("\nThe application will now use these bundled models")
        print("instead of downloading them on first run.")
        print("\nModel locations:")
        for _, _, target_dir, _ in MODELS:
            print(f"  • {target_dir}: {models_dir / target_dir}")
        return 0
    else:
        print("\n✗ Some models failed to install.")
        print("Please check the errors above and try again.")
        print("\nNote: The application will still work - it will auto-download")
        print("missing models from HuggingFace on first use.")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
