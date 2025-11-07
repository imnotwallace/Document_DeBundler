"""
Language Pack Manager for PaddleOCR 3.x

Manages language packs by triggering PaddleOCR's automatic model download.
PaddleOCR 3.x auto-downloads models from Hugging Face on first use.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict

from .language_pack_metadata import (
    get_language_pack,
    get_all_languages,
    get_supported_language_codes,
    get_script_info,
    check_model_installed,
    LanguagePackInfo,
    ScriptModelInfo
)

logger = logging.getLogger(__name__)


@dataclass
class LanguageStatus:
    """Status of a language pack installation"""
    code: str
    name: str
    installed: bool  # True if ANY version is installed
    script_name: str  # "latin", "arabic", "cyrillic", etc.
    script_description: str
    total_size_mb: float
    detection_installed: bool
    recognition_installed: bool  # For current model_version
    model_version: str = "mobile"  # "server" or "mobile"
    has_server_version: bool = False  # Whether server version is available
    available_versions: List[str] = None  # List of available versions
    # NEW: Per-version installation status
    server_installed: bool = False  # Whether server version is installed
    mobile_installed: bool = False  # Whether mobile version is installed  # List of available versions


@dataclass
class DownloadProgress:
    """Progress information for a download operation"""
    language: str
    language_name: str
    phase: str  # "initializing" | "downloading" | "complete" | "error"
    progress_percent: float  # 0-100
    message: str
    speed_mbps: Optional[float] = None
    error: Optional[str] = None


class LanguagePackManager:
    """Manages language pack installations via PaddleOCR auto-download"""

    def __init__(self):
        """Initialize language pack manager."""
        logger.info("Language Pack Manager initialized for PaddleOCR 3.x auto-download")

    def get_language_status(self, language_code: str, version: str = "mobile") -> Optional[LanguageStatus]:
        """
        Get installation status for a specific language with a specific version.

        Args:
            language_code: Language code (e.g., "en", "french", "ch")
            version: Model version ("server" or "mobile")

        Returns:
            LanguageStatus with current installation state
        """
        from .language_pack_metadata import get_language_pack_with_version
        
        lang_pack = get_language_pack_with_version(language_code, version)
        if not lang_pack:
            return None

        detection_installed = check_model_installed(lang_pack.detection_model_name)
        recognition_model_name = lang_pack.get_recognition_model_name()
        recognition_installed = check_model_installed(recognition_model_name)

        return LanguageStatus(
            code=lang_pack.code,
            name=lang_pack.name,
            installed=lang_pack.installed,
            script_name=lang_pack.script_model.script_name,
            script_description=lang_pack.script_model.description,
            total_size_mb=lang_pack.total_size_mb,
            detection_installed=detection_installed,
            recognition_installed=recognition_installed,
            model_version=lang_pack.model_version,
            has_server_version=lang_pack.can_use_server_version(),
            available_versions=lang_pack.get_available_versions()
        )

    def get_all_language_statuses(self) -> List[LanguageStatus]:
        """
        Get installation status for all languages with per-version status.

        Returns:
            List of LanguageStatus for all available languages
        """
        from .language_pack_metadata import get_language_pack_with_version
        
        statuses = []
        for lang_pack in get_all_languages():
            detection_installed = check_model_installed(lang_pack.detection_model_name)
            
            # Check installation status for BOTH versions (if available)
            mobile_installed = False
            server_installed = False
            
            # Always check mobile version
            mobile_pack = get_language_pack_with_version(lang_pack.code, "mobile")
            if mobile_pack:
                mobile_model_name = mobile_pack.get_recognition_model_name()
                mobile_installed = check_model_installed(mobile_model_name)
            
            # Check server version if available
            if lang_pack.can_use_server_version():
                server_pack = get_language_pack_with_version(lang_pack.code, "server")
                if server_pack:
                    server_model_name = server_pack.get_recognition_model_name()
                    server_installed = check_model_installed(server_model_name)
            
            # Determine which version to report as "current"
            # Use the method to get the correct model name (server or mobile)
            recognition_model_name = lang_pack.get_recognition_model_name()
            recognition_installed = check_model_installed(recognition_model_name)
            
            # Language is "installed" if ANY version is installed
            any_version_installed = mobile_installed or server_installed

            status = LanguageStatus(
                code=lang_pack.code,
                name=lang_pack.name,
                installed=any_version_installed,
                script_name=lang_pack.script_model.script_name,
                script_description=lang_pack.script_model.description,
                total_size_mb=lang_pack.total_size_mb,
                detection_installed=detection_installed,
                recognition_installed=recognition_installed,
                model_version=lang_pack.model_version,
                has_server_version=lang_pack.can_use_server_version(),
                available_versions=lang_pack.get_available_versions(),
                server_installed=server_installed,
                mobile_installed=mobile_installed
            )
            statuses.append(status)

        return statuses

    def trigger_language_download(
        self,
        language_code: str,
        version: str = "mobile",
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None
    ) -> bool:
        """
        Trigger PaddleOCR to download models for a language with a specific version.

        This initializes PaddleOCR with the specified language and version, which triggers
        automatic model download from Hugging Face if models aren't already installed.

        Args:
            language_code: Language code to download
            version: Model version ("server" or "mobile")
            progress_callback: Optional callback for progress updates

        Returns:
            True if successful, False otherwise
        """
        from .language_pack_metadata import get_language_pack_with_version
        
        lang_pack = get_language_pack_with_version(language_code, version)
        if not lang_pack:
            logger.error(f"Unknown language code: {language_code}")
            if progress_callback:
                progress_callback(DownloadProgress(
                    language=language_code,
                    language_name=language_code,
                    phase="error",
                    progress_percent=0,
                    message=f"Unknown language: {language_code}",
                    error=f"Language code '{language_code}' is not supported"
                ))
            return False

        try:
            if progress_callback:
                version_str = f" ({version} version)" if lang_pack.can_use_server_version() else ""
                progress_callback(DownloadProgress(
                    language=language_code,
                    language_name=lang_pack.name,
                    phase="initializing",
                    progress_percent=0,
                    message=f"Initializing PaddleOCR for {lang_pack.name}{version_str}..."
                ))

            # Import PaddleOCR here to avoid startup overhead
            from paddleocr import PaddleOCR

            logger.info(f"Initializing PaddleOCR for language: {language_code} (version: {version})")

            if progress_callback:
                version_str = f" ({version} version)" if lang_pack.can_use_server_version() else ""
                progress_callback(DownloadProgress(
                    language=language_code,
                    language_name=lang_pack.name,
                    phase="downloading",
                    progress_percent=50,
                    message=f"Downloading {lang_pack.script_model.script_name} models{version_str} (this may take a few minutes)..."
                ))

            # Initialize PaddleOCR with lang parameter to trigger auto-download
            # PaddleOCR 3.x auto-downloads models based on language code
            rec_model_name = lang_pack.get_recognition_model_name()
            det_model_name = lang_pack.detection_model_name
            
            logger.info(f"Downloading PaddleOCR models for language: {language_code}")
            logger.info(f"  - Expected detection: {det_model_name}")
            logger.info(f"  - Expected recognition: {rec_model_name}")
            logger.warning(f"NOTE: PaddleOCR may auto-select mobile version regardless of requested '{version}' version")
            
            # Initialize with lang only - triggers auto-download
            # LIMITATION: PaddleOCR 3.x may not respect version preference during download
            ocr = PaddleOCR(
                lang=language_code,
                device='cpu',  # Use CPU for download
                show_log=True  # Show download progress
            )

            # Verify models were downloaded
            detection_model_name = lang_pack.detection_model_name
            detection_exists = check_model_installed(detection_model_name)
            recognition_exists = check_model_installed(rec_model_name)
            
            logger.info(f"Verification - Detection model ({detection_model_name}): {detection_exists}")
            logger.info(f"Verification - Recognition model ({rec_model_name}): {recognition_exists}")
            
            if not detection_exists:
                raise Exception(f"Detection model not found: {detection_model_name}")
            if not recognition_exists:
                raise Exception(f"Recognition model not found: {rec_model_name}")

            if progress_callback:
                version_str = f" ({version} version)" if lang_pack.can_use_server_version() else ""
                progress_callback(DownloadProgress(
                    language=language_code,
                    language_name=lang_pack.name,
                    phase="complete",
                    progress_percent=100,
                    message=f"Successfully installed {lang_pack.name}{version_str}"
                ))

            logger.info(f"Successfully downloaded models for {language_code} ({version})")
            return True

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to download models for {language_code} ({version}): {error_msg}")

            if progress_callback:
                progress_callback(DownloadProgress(
                    language=language_code,
                    language_name=lang_pack.name,
                    phase="error",
                    progress_percent=0,
                    message=f"Failed to download {lang_pack.name}",
                    error=error_msg
                ))

            return False

    def check_language_installed(self, language_code: str) -> bool:
        """
        Check if a language pack is fully installed.

        Args:
            language_code: Language code to check

        Returns:
            True if both detection and recognition models are installed
        """
        lang_pack = get_language_pack(language_code)
        if not lang_pack:
            return False

        return lang_pack.installed

    def get_installed_languages(self) -> List[str]:
        """
        Get list of fully installed language codes.

        Returns:
            List of language codes that are installed
        """
        return [
            lang.code
            for lang in get_all_languages()
            if lang.installed
        ]

    def get_missing_languages(self) -> List[str]:
        """
        Get list of language codes that are not yet installed.

        Returns:
            List of language codes that need to be downloaded
        """
        return [
            lang.code
            for lang in get_all_languages()
            if not lang.installed
        ]


# Singleton instance
_manager_instance: Optional[LanguagePackManager] = None


def get_language_pack_manager() -> LanguagePackManager:
    """
    Get the singleton instance of LanguagePackManager.

    Returns:
        LanguagePackManager instance
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = LanguagePackManager()
    return _manager_instance
