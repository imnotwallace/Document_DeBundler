"""
Language Pack Metadata for PaddleOCR 3.x (PP-OCRv5)

Defines available languages, their codes, script types, and model information.
PaddleOCR 3.x auto-downloads models from Hugging Face on first use.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class ScriptModelInfo:
    """Information about a script-based recognition model"""
    script_name: str  # e.g., "latin", "arabic", "cyrillic"
    server_model_name: Optional[str]  # e.g., "PP-OCRv5_server_rec" (if available)
    mobile_model_name: str   # e.g., "latin_PP-OCRv5_mobile_rec"
    approximate_size_mb: float  # Approximate download size
    description: str  # What languages use this script
    has_server_version: bool = False  # Whether server version exists


@dataclass
class LanguagePackInfo:
    """Complete information about a language pack"""
    code: str  # PaddleOCR language code (e.g., "en", "french", "ch")
    name: str  # Display name (e.g., "English", "French")
    script_model: ScriptModelInfo  # Script-based recognition model
    detection_model_name: str  # Detection model (usually PP-OCRv5_server_det)
    total_size_mb: float  # Total approximate download size
    model_version: str = "mobile"  # "server" or "mobile" - preferred version
    installed: bool = False  # Whether models are downloaded
    supported: bool = True  # Whether this language is fully supported

    def get_recognition_model_name(self) -> str:
        """
        Get the recognition model name based on the selected version.
        
        Returns:
            Model name for the selected version (server or mobile)
        """
        if self.model_version == "server" and self.script_model.has_server_version:
            return self.script_model.server_model_name
        else:
            # Fall back to mobile if server not available or mobile selected
            return self.script_model.mobile_model_name
    
    def get_detection_model_name(self) -> str:
        """
        Get the detection model name based on the selected version.
        
        Returns:
            Detection model name for the selected version (server or mobile)
        """
        if self.model_version == "server":
            return DETECTION_MODEL_SERVER
        else:
            return DETECTION_MODEL_MOBILE
    
    def can_use_server_version(self) -> bool:
        """Check if server version is available for this language"""
        return self.script_model.has_server_version
    
    def get_available_versions(self) -> List[str]:
        """Get list of available model versions for this language"""
        versions = ["mobile"]  # Mobile always available
        if self.script_model.has_server_version:
            versions.append("server")
        return versions


# Script-based recognition models
# NOTE: Some models have both server and mobile versions, others only mobile
SCRIPT_MODELS = {
    "latin": ScriptModelInfo(
        script_name="latin",
        server_model_name=None,  # No server version available
        mobile_model_name="latin_PP-OCRv5_mobile_rec",
        approximate_size_mb=8.5,
        description="French, German, Spanish, Portuguese, Italian, Dutch, and other Latin-script languages",
        has_server_version=False
    ),
    "english": ScriptModelInfo(
        script_name="english",
        server_model_name="PP-OCRv4_server_rec",  # CRITICAL: v4 detects 15x more text than v5!
        mobile_model_name="en_PP-OCRv4_mobile_rec",  # CRITICAL: v4 detects 15x more text than v5!
        approximate_size_mb=9.0,
        description="English (optimized)",
        has_server_version=True
    ),
    "chinese": ScriptModelInfo(
        script_name="chinese",
        server_model_name="PP-OCRv4_server_rec",  # CRITICAL: v4 detects 15x more text than v5!
        mobile_model_name="PP-OCRv4_mobile_rec",  # CRITICAL: v4 detects 15x more text than v5!
        approximate_size_mb=10.5,
        description="Chinese (Simplified and Traditional)",
        has_server_version=True
    ),
    "arabic": ScriptModelInfo(
        script_name="arabic",
        server_model_name=None,  # No server version available
        mobile_model_name="arabic_PP-OCRv3_mobile_rec",  # NOTE: Using v3, v5 not available
        approximate_size_mb=9.0,
        description="Arabic",
        has_server_version=False
    ),
    "cyrillic": ScriptModelInfo(
        script_name="cyrillic",
        server_model_name=None,  # No server version available
        mobile_model_name="eslav_PP-OCRv5_mobile_rec",
        approximate_size_mb=9.0,
        description="Russian, Ukrainian, Bulgarian, and other Cyrillic-script languages",
        has_server_version=False
    ),
    "devanagari": ScriptModelInfo(
        script_name="devanagari",
        server_model_name=None,  # No server version available
        mobile_model_name="devanagari_PP-OCRv3_mobile_rec",  # NOTE: Using v3, v5 not available
        approximate_size_mb=9.5,
        description="Hindi, Sanskrit, Marathi, and other Devanagari-script languages",
        has_server_version=False
    ),
    "korean": ScriptModelInfo(
        script_name="korean",
        server_model_name=None,  # No server version available
        mobile_model_name="korean_PP-OCRv5_mobile_rec",
        approximate_size_mb=11.0,
        description="Korean",
        has_server_version=False
    ),
    "japanese": ScriptModelInfo(
        script_name="japanese",
        server_model_name="PP-OCRv4_server_rec",  # CRITICAL: v4 detects 15x more text than v5!
        mobile_model_name="PP-OCRv4_mobile_rec",  # CRITICAL: v4 detects 15x more text than v5!
        approximate_size_mb=11.5,
        description="Japanese",
        has_server_version=True
    ),
}

# Shared detection models (used across all languages)
# Server model: More accurate and better quality, but more conservative (fewer false positives)
# Mobile model: Smaller/faster but lower quality (more gibberish)
# CRITICAL: Using PP-OCRv4 instead of PP-OCRv5 - v4 detects 15x more text regions on small body text!
DETECTION_MODEL_SERVER = "PP-OCRv4_server_det"
DETECTION_MODEL_MOBILE = "PP-OCRv4_mobile_det"
DETECTION_MODEL_SIZE_SERVER_MB = 12.0
DETECTION_MODEL_SIZE_MOBILE_MB = 1.1  # Mobile is ~11x smaller

# Default detection model: Server has better quality, even if more conservative
DETECTION_MODEL_NAME = DETECTION_MODEL_SERVER
DETECTION_MODEL_SIZE_MB = DETECTION_MODEL_SIZE_SERVER_MB

# Language pack definitions
# Format: code -> LanguagePackInfo
LANGUAGE_PACKS: Dict[str, LanguagePackInfo] = {
    "en": LanguagePackInfo(
        code="en",
        name="English",
        script_model=SCRIPT_MODELS["english"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["english"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "ch": LanguagePackInfo(
        code="ch",
        name="Chinese (Simplified)",
        script_model=SCRIPT_MODELS["chinese"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["chinese"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "chinese_cht": LanguagePackInfo(
        code="chinese_cht",
        name="Chinese (Traditional)",
        script_model=SCRIPT_MODELS["chinese"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["chinese"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "french": LanguagePackInfo(
        code="french",
        name="French",
        script_model=SCRIPT_MODELS["latin"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["latin"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "german": LanguagePackInfo(
        code="german",
        name="German",
        script_model=SCRIPT_MODELS["latin"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["latin"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "es": LanguagePackInfo(
        code="es",
        name="Spanish",
        script_model=SCRIPT_MODELS["latin"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["latin"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "pt": LanguagePackInfo(
        code="pt",
        name="Portuguese",
        script_model=SCRIPT_MODELS["latin"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["latin"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "it": LanguagePackInfo(
        code="it",
        name="Italian",
        script_model=SCRIPT_MODELS["latin"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["latin"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "nl": LanguagePackInfo(
        code="nl",
        name="Dutch",
        script_model=SCRIPT_MODELS["latin"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["latin"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "ru": LanguagePackInfo(
        code="ru",
        name="Russian",
        script_model=SCRIPT_MODELS["cyrillic"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["cyrillic"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "ar": LanguagePackInfo(
        code="ar",
        name="Arabic",
        script_model=SCRIPT_MODELS["arabic"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["arabic"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "hi": LanguagePackInfo(
        code="hi",
        name="Hindi",
        script_model=SCRIPT_MODELS["devanagari"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["devanagari"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "korean": LanguagePackInfo(
        code="korean",
        name="Korean",
        script_model=SCRIPT_MODELS["korean"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["korean"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "japan": LanguagePackInfo(
        code="japan",
        name="Japanese",
        script_model=SCRIPT_MODELS["japanese"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["japanese"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "pl": LanguagePackInfo(
        code="pl",
        name="Polish",
        script_model=SCRIPT_MODELS["latin"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["latin"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "sv": LanguagePackInfo(
        code="sv",
        name="Swedish",
        script_model=SCRIPT_MODELS["latin"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["latin"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "da": LanguagePackInfo(
        code="da",
        name="Danish",
        script_model=SCRIPT_MODELS["latin"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["latin"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "tr": LanguagePackInfo(
        code="tr",
        name="Turkish",
        script_model=SCRIPT_MODELS["latin"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["latin"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
    "vi": LanguagePackInfo(
        code="vi",
        name="Vietnamese",
        script_model=SCRIPT_MODELS["latin"],
        detection_model_name=DETECTION_MODEL_NAME,
        total_size_mb=SCRIPT_MODELS["latin"].approximate_size_mb + DETECTION_MODEL_SIZE_MB
    ),
}


def get_paddlex_models_dir() -> Path:
    """
    Get the directory where PaddleOCR stores auto-downloaded models.

    Returns:
        Path to ~/.paddlex/official_models/
    """
    home = Path.home()
    return home / ".paddlex" / "official_models"


def check_model_installed(model_name: str) -> bool:
    """
    Check if a specific model is already downloaded.

    Args:
        model_name: Model name (e.g., "latin_PP-OCRv5_mobile_rec")

    Returns:
        True if model directory exists
    """
    models_dir = get_paddlex_models_dir()
    model_path = models_dir / model_name
    return model_path.exists() and model_path.is_dir()


def update_installed_status(lang_pack: LanguagePackInfo) -> LanguagePackInfo:
    """
    Update the installed status of a language pack by checking if models exist.

    Args:
        lang_pack: Language pack to check

    Returns:
        Updated language pack with installed status
    """
    detection_model_name = lang_pack.get_detection_model_name()
    detection_installed = check_model_installed(detection_model_name)
    recognition_model_name = lang_pack.get_recognition_model_name()
    recognition_installed = check_model_installed(recognition_model_name)

    lang_pack.installed = detection_installed and recognition_installed
    return lang_pack


def get_language_pack(code: str) -> Optional[LanguagePackInfo]:
    """
    Get language pack information by code.

    Args:
        code: Language code (e.g., "en", "french", "ch")

    Returns:
        LanguagePackInfo if found, None otherwise
    """
    lang_pack = LANGUAGE_PACKS.get(code)
    if lang_pack:
        return update_installed_status(lang_pack)
    return None


def get_language_pack_with_version(code: str, version: str = "mobile") -> Optional[LanguagePackInfo]:
    """
    Get language pack information by code with a specific model version.

    Args:
        code: Language code (e.g., "en", "french", "ch")
        version: Model version ("server" or "mobile")

    Returns:
        LanguagePackInfo if found, None otherwise
    """
    lang_pack = LANGUAGE_PACKS.get(code)
    if lang_pack:
        # Create a copy with the specified version
        lang_pack_copy = LanguagePackInfo(
            code=lang_pack.code,
            name=lang_pack.name,
            script_model=lang_pack.script_model,
            detection_model_name=lang_pack.detection_model_name,
            total_size_mb=lang_pack.total_size_mb,
            model_version=version if lang_pack.can_use_server_version() and version == "server" else "mobile",
            installed=lang_pack.installed,
            supported=lang_pack.supported
        )
        return update_installed_status(lang_pack_copy)
    return None


def get_all_languages() -> List[LanguagePackInfo]:
    """
    Get all available language packs with updated installation status.

    Returns:
        List of all language pack info objects
    """
    return [update_installed_status(pack) for pack in LANGUAGE_PACKS.values()]


def get_supported_language_codes() -> List[str]:
    """
    Get list of all supported language codes.

    Returns:
        List of language codes
    """
    return list(LANGUAGE_PACKS.keys())


def get_script_info(code: str) -> Optional[ScriptModelInfo]:
    """
    Get script model information for a language.

    Args:
        code: Language code

    Returns:
        ScriptModelInfo if found, None otherwise
    """
    lang_pack = get_language_pack(code)
    if lang_pack:
        return lang_pack.script_model
    return None
