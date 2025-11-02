"""
LLM Settings - User Configuration Management

Manages user-configurable settings for LLM features including:
- Enable/disable LLM features
- Model selection
- Performance tuning
- VRAM allocation
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class LLMSettings:
    """User-configurable LLM settings."""

    # Feature toggles
    enabled: bool = True
    split_refinement_enabled: bool = True
    naming_enabled: bool = True

    # Model selection
    model_preference: str = "auto"  # "auto", "phi3_mini", "gemma2_2b"
    use_gpu: bool = True

    # Performance tuning
    max_gpu_layers: Optional[int] = None  # None for auto, or specific number
    context_size: int = 4096
    batch_size: int = 256

    # Quality/confidence thresholds
    split_confidence_threshold: float = 0.6  # Min confidence to accept LLM split decision
    naming_fallback_enabled: bool = True  # Use heuristic fallback if LLM fails

    # Memory management
    auto_cleanup_enabled: bool = True  # Auto cleanup after processing
    sequential_processing: bool = True  # Process OCR/Embeddings/LLM sequentially

    # Caching
    cache_llm_results: bool = True
    cache_ttl_days: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMSettings':
        """Create from dictionary."""
        # Filter out unknown keys
        valid_keys = set(cls.__annotations__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


class LLMSettingsManager:
    """
    Manager for loading, saving, and validating LLM settings.

    Settings are stored in a JSON file for persistence across sessions.
    """

    def __init__(self, settings_file: Optional[Path] = None):
        """
        Initialize settings manager.

        Args:
            settings_file: Path to settings JSON file (None for default)
        """
        if settings_file is None:
            # Default to user config directory
            from services.resource_path import get_base_path
            base_path = get_base_path()
            self.settings_file = base_path / "config" / "llm_settings.json"
        else:
            self.settings_file = settings_file

        self.settings = LLMSettings()
        self.load()

        logger.info(f"LLM Settings Manager initialized (file: {self.settings_file})")

    def load(self) -> bool:
        """
        Load settings from file.

        Returns:
            True if loaded successfully
        """
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    data = json.load(f)

                self.settings = LLMSettings.from_dict(data)
                logger.info(f"Loaded settings from {self.settings_file}")
                return True
            else:
                logger.info("No settings file found, using defaults")
                # Save defaults
                self.save()
                return False

        except Exception as e:
            logger.error(f"Failed to load settings: {e}", exc_info=True)
            logger.info("Using default settings")
            return False

    def save(self) -> bool:
        """
        Save settings to file.

        Returns:
            True if saved successfully
        """
        try:
            # Ensure directory exists
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.settings_file, 'w') as f:
                json.dump(self.settings.to_dict(), f, indent=2)

            logger.info(f"Saved settings to {self.settings_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save settings: {e}", exc_info=True)
            return False

    def get(self) -> LLMSettings:
        """Get current settings."""
        return self.settings

    def update(self, **kwargs) -> bool:
        """
        Update settings.

        Args:
            **kwargs: Settings to update

        Returns:
            True if updated successfully
        """
        try:
            # Update settings
            for key, value in kwargs.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
                    logger.debug(f"Updated {key} = {value}")
                else:
                    logger.warning(f"Unknown setting: {key}")

            # Save to file
            return self.save()

        except Exception as e:
            logger.error(f"Failed to update settings: {e}", exc_info=True)
            return False

    def reset(self) -> bool:
        """
        Reset to default settings.

        Returns:
            True if reset successfully
        """
        try:
            self.settings = LLMSettings()
            logger.info("Settings reset to defaults")
            return self.save()
        except Exception as e:
            logger.error(f"Failed to reset settings: {e}", exc_info=True)
            return False

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate current settings.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            settings = self.settings

            # Validate thresholds
            if not 0 <= settings.split_confidence_threshold <= 1:
                return False, "split_confidence_threshold must be between 0 and 1"

            # Validate cache TTL
            if settings.cache_ttl_days < 1:
                return False, "cache_ttl_days must be at least 1"

            # Validate context size
            valid_context_sizes = [512, 1024, 2048, 4096, 8192]
            if settings.context_size not in valid_context_sizes:
                return False, f"context_size must be one of {valid_context_sizes}"

            # Validate model preference
            valid_models = ["auto", "phi3_mini", "gemma2_2b"]
            if settings.model_preference not in valid_models:
                return False, f"model_preference must be one of {valid_models}"

            # Validate GPU layers if specified
            if settings.max_gpu_layers is not None:
                if settings.max_gpu_layers < 0 or settings.max_gpu_layers > 40:
                    return False, "max_gpu_layers must be between 0 and 40 (or None for auto)"

            return True, None

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def get_effective_config(self) -> Dict[str, Any]:
        """
        Get effective configuration after applying auto-detection.

        Returns:
            Dictionary with resolved configuration
        """
        from services.llm.config import select_optimal_llm_config
        from services.ocr.config import detect_hardware_capabilities

        # Detect hardware
        capabilities = detect_hardware_capabilities()

        # Get optimal LLM config
        gpu_memory_gb = capabilities['gpu_memory_gb'] if self.settings.use_gpu else 0
        llm_config = select_optimal_llm_config(gpu_memory_gb)

        # Override with user settings if specified
        if self.settings.max_gpu_layers is not None:
            llm_config['n_gpu_layers'] = self.settings.max_gpu_layers

        llm_config['n_ctx'] = self.settings.context_size
        llm_config['n_batch'] = self.settings.batch_size

        return {
            'hardware': {
                'gpu_memory_gb': capabilities['gpu_memory_gb'],
                'system_memory_gb': capabilities['system_memory_gb'],
                'cpu_count': capabilities['cpu_count'],
                'platform': capabilities['platform']
            },
            'llm_config': llm_config,
            'user_settings': self.settings.to_dict()
        }

    def print_settings(self):
        """Print current settings in human-readable format."""
        print("=" * 60)
        print("LLM Settings")
        print("=" * 60)

        settings = self.settings.to_dict()

        print("\nFeature Toggles:")
        print(f"  LLM Enabled: {settings['enabled']}")
        print(f"  Split Refinement: {settings['split_refinement_enabled']}")
        print(f"  AI Naming: {settings['naming_enabled']}")

        print("\nModel Configuration:")
        print(f"  Model Preference: {settings['model_preference']}")
        print(f"  Use GPU: {settings['use_gpu']}")
        print(f"  Max GPU Layers: {settings['max_gpu_layers'] or 'auto'}")

        print("\nPerformance:")
        print(f"  Context Size: {settings['context_size']}")
        print(f"  Batch Size: {settings['batch_size']}")
        print(f"  Sequential Processing: {settings['sequential_processing']}")

        print("\nQuality Thresholds:")
        print(f"  Split Confidence: {settings['split_confidence_threshold']}")
        print(f"  Naming Fallback: {settings['naming_fallback_enabled']}")

        print("\nMemory Management:")
        print(f"  Auto Cleanup: {settings['auto_cleanup_enabled']}")

        print("\nCaching:")
        print(f"  Cache Results: {settings['cache_llm_results']}")
        print(f"  Cache TTL: {settings['cache_ttl_days']} days")

        print("=" * 60)


# Singleton instance
_settings_manager_instance = None


def get_settings_manager() -> LLMSettingsManager:
    """
    Get singleton settings manager instance.

    Returns:
        LLMSettingsManager instance
    """
    global _settings_manager_instance
    if _settings_manager_instance is None:
        _settings_manager_instance = LLMSettingsManager()
    return _settings_manager_instance


def get_settings() -> LLMSettings:
    """
    Get current LLM settings.

    Returns:
        LLMSettings instance
    """
    manager = get_settings_manager()
    return manager.get()


def update_settings(**kwargs) -> bool:
    """
    Update LLM settings.

    Args:
        **kwargs: Settings to update

    Returns:
        True if updated successfully
    """
    manager = get_settings_manager()
    return manager.update(**kwargs)


def test_settings():
    """Test function for LLM settings."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("LLM Settings Test")
    print("=" * 60)

    # Get manager
    manager = get_settings_manager()

    # Print default settings
    print("\n1. Default Settings:")
    manager.print_settings()

    # Validate
    print("\n2. Validation:")
    valid, error = manager.validate()
    print(f"  Valid: {valid}")
    if error:
        print(f"  Error: {error}")

    # Update settings
    print("\n3. Updating settings...")
    manager.update(
        split_refinement_enabled=True,
        naming_enabled=True,
        use_gpu=True,
        split_confidence_threshold=0.7
    )

    # Print updated
    print("\n4. Updated Settings:")
    manager.print_settings()

    # Get effective config
    print("\n5. Effective Configuration:")
    try:
        config = manager.get_effective_config()
        print(f"  Hardware: {config['hardware']}")
        print(f"  LLM Config: {config['llm_config']['model_name']}")
        print(f"  GPU Layers: {config['llm_config']['n_gpu_layers']}")
    except Exception as e:
        print(f"  Error: {e}")

    # Reset
    print("\n6. Resetting to defaults...")
    manager.reset()
    manager.print_settings()

    print("\n" + "=" * 60)
    print("✓ Test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_settings()
