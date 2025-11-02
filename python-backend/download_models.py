#!/usr/bin/env python3
"""
PaddleOCR Model Downloader
Automatically downloads and extracts PaddleOCR models for offline operation.
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

# Model definitions: (name, url, target_directory)
MODELS = [
    (
        "Detection Model (English PP-OCRv3)",
        "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
        "det",
        "en_PP-OCRv3_det_infer"
    ),
    (
        "Recognition Model (English PP-OCRv4)",
        "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
        "rec",
        "en_PP-OCRv4_rec_infer"
    ),
    (
        "Angle Classification Model (Multilingual)",
        "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
        "cls",
        "ch_ppocr_mobile_v2.0_cls_infer"
    ),
]


class DownloadProgress:
    """Simple progress reporter for downloads"""

    def __init__(self, name: str):
        self.name = name
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        """Progress callback for urllib.request.urlretrieve"""
        if total_size <= 0:
            return

        downloaded = block_num * block_size
        percent = min(int(downloaded * 100 / total_size), 100)

        # Only print every 10% to reduce output noise
        if percent >= self.last_percent + 10 or percent == 100:
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"  {percent:3d}% - {mb_downloaded:.1f} MB / {mb_total:.1f} MB", end='\r')
            self.last_percent = percent

            if percent == 100:
                print()  # New line after completion


def get_models_dir() -> Path:
    """Get the models directory path"""
    # This script should be in python-backend/
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models"

    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created models directory: {models_dir}")

    return models_dir


def check_existing_models(models_dir: Path) -> List[str]:
    """Check which models are already installed"""
    existing = []

    for name, _, target_dir, _ in MODELS:
        model_path = models_dir / target_dir
        # Check if model files exist
        if model_path.exists():
            pdmodel = model_path / "inference.pdmodel"
            pdiparams = model_path / "inference.pdiparams"

            if pdmodel.exists() and pdiparams.exists():
                existing.append(target_dir)

    return existing


def download_and_extract_model(
    name: str,
    url: str,
    models_dir: Path,
    target_dir: str,
    extracted_dir_name: str
) -> bool:
    """
    Download and extract a single model.

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n[{name}]")
        print(f"  URL: {url}")

        # Download to temp file
        temp_file = models_dir / f"{target_dir}_temp.tar"
        print(f"  Downloading...")

        progress = DownloadProgress(name)
        urllib.request.urlretrieve(url, temp_file, reporthook=progress)

        # Extract
        print(f"  Extracting...")
        target_path = models_dir / target_dir
        target_path.mkdir(parents=True, exist_ok=True)

        with tarfile.open(temp_file, 'r') as tar:
            # Extract to temporary location first
            extract_temp = models_dir / "temp_extract"
            extract_temp.mkdir(parents=True, exist_ok=True)
            tar.extractall(extract_temp)

            # Move files from extracted directory to target
            extracted_model_dir = extract_temp / extracted_dir_name
            if extracted_model_dir.exists():
                # Move all files from extracted directory to target
                import shutil
                for item in extracted_model_dir.iterdir():
                    dest = target_path / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(target_path))

            # Cleanup
            shutil.rmtree(extract_temp)

        # Remove temp file
        temp_file.unlink()

        # Verify extraction
        pdmodel = target_path / "inference.pdmodel"
        pdiparams = target_path / "inference.pdiparams"

        if pdmodel.exists() and pdiparams.exists():
            pdmodel_size = pdmodel.stat().st_size / (1024 * 1024)
            pdiparams_size = pdiparams.stat().st_size / (1024 * 1024)
            print(f"  ✓ Successfully installed to {target_path}")
            print(f"    - inference.pdmodel: {pdmodel_size:.1f} MB")
            print(f"    - inference.pdiparams: {pdiparams_size:.1f} MB")
            return True
        else:
            print(f"  ✗ Extraction failed - model files not found")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def verify_installation(models_dir: Path) -> bool:
    """Verify all models are correctly installed"""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    all_ok = True

    for name, _, target_dir, _ in MODELS:
        model_path = models_dir / target_dir
        pdmodel = model_path / "inference.pdmodel"
        pdiparams = model_path / "inference.pdiparams"

        if pdmodel.exists() and pdiparams.exists():
            print(f"✓ {name}: OK")
        else:
            print(f"✗ {name}: MISSING")
            all_ok = False

    return all_ok


def main():
    """Main entry point"""
    print("="*60)
    print("PaddleOCR Model Downloader")
    print("="*60)
    print()

    # Get models directory
    models_dir = get_models_dir()
    print(f"Models directory: {models_dir}")

    # Check existing models
    existing = check_existing_models(models_dir)
    if existing:
        print(f"\nExisting models found: {', '.join(existing)}")
        response = input("Do you want to re-download existing models? (y/N): ").strip().lower()
        skip_existing = response not in ('y', 'yes')
    else:
        skip_existing = False

    # Download models
    print("\n" + "="*60)
    print("DOWNLOADING MODELS")
    print("="*60)

    success_count = 0
    total_count = len(MODELS)

    for name, url, target_dir, extracted_dir_name in MODELS:
        if skip_existing and target_dir in existing:
            print(f"\n[{name}]")
            print(f"  Skipping (already exists)")
            success_count += 1
            continue

        if download_and_extract_model(name, url, models_dir, target_dir, extracted_dir_name):
            success_count += 1

    # Verify installation
    all_ok = verify_installation(models_dir)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Successfully installed: {success_count}/{total_count} models")

    if all_ok:
        print("\n✓ All models are ready for use!")
        print("\nThe application will now use these bundled models")
        print("instead of downloading them on first run.")
        return 0
    else:
        print("\n✗ Some models failed to install.")
        print("Please check the errors above and try again.")
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
