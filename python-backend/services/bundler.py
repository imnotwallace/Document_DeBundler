"""
Bundler Service
Creates ZIP archives or organizes files into folders
"""

import logging
import zipfile
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class Bundler:
    """Handles bundling of split documents"""

    @staticmethod
    def create_zip(files: List[Path], output_path: Path) -> Path:
        """Create a ZIP archive containing all files"""
        logger.info(f"Creating ZIP archive: {output_path}")

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files:
                if file_path.exists():
                    zipf.write(file_path, file_path.name)
                    logger.debug(f"Added to ZIP: {file_path.name}")

        logger.info(f"ZIP archive created: {output_path}")
        return output_path

    @staticmethod
    def organize_to_folder(files: List[Path], output_dir: Path) -> Path:
        """Copy files to organized folder structure"""
        logger.info(f"Organizing files to: {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            if file_path.exists():
                dest = output_dir / file_path.name
                dest.write_bytes(file_path.read_bytes())
                logger.debug(f"Copied: {file_path.name} -> {dest}")

        logger.info(f"Files organized to: {output_dir}")
        return output_dir
