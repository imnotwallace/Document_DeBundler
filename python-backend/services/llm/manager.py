"""
LLM Manager Service

Singleton manager for LLM lifecycle with lazy loading, thread safety, and cleanup.
Coordinates with OCR and embedding services for sequential GPU usage.
"""

import logging
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Singleton manager for LLM operations.

    Features:
    - Lazy loading (initialize only when needed)
    - Thread-safe generation with queuing
    - Automatic cleanup and memory management
    - Sequential GPU processing coordination
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize manager (only once)."""
        if self._initialized:
            return

        self._initialized = True
        self.loader = None
        self.enabled = True
        self.generation_lock = threading.Lock()
        self.stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_tokens_generated': 0
        }

        logger.info("LLMManager initialized (singleton)")

    @classmethod
    def get_instance(cls) -> 'LLMManager':
        """Get singleton instance."""
        return cls()

    def initialize(
        self,
        use_gpu: bool = True,
        force_reload: bool = False,
        model_name: Optional[str] = None
    ) -> bool:
        """
        Initialize LLM (lazy loading).

        Args:
            use_gpu: Whether to use GPU acceleration
            force_reload: Force reload even if already loaded
            model_name: Specific model to load

        Returns:
            True if initialization successful
        """
        with self._lock:
            # Check if already loaded
            if self.loader and not force_reload:
                logger.info("LLM already initialized")
                return True

            # Clean up existing loader if reloading
            if self.loader and force_reload:
                logger.info("Force reload requested, cleaning up existing loader...")
                self._cleanup_internal()

            try:
                from services.llm.loader import LlamaLoader

                logger.info("Initializing LLM loader...")
                self.loader = LlamaLoader(
                    use_gpu=use_gpu,
                    use_binary_fallback=True,
                    verbose=False
                )

                # Load model
                success = self.loader.load_model(model_name=model_name)

                if success:
                    info = self.loader.get_model_info()
                    logger.info(f"✓ LLM initialized: {info['model_name']}")
                    logger.info(f"  Mode: {info['mode']}")
                    logger.info(f"  GPU: {info['gpu_enabled']} ({info['gpu_layers']} layers)")
                    logger.info(f"  VRAM: {info['expected_vram_gb']:.1f}GB")
                    return True
                else:
                    logger.error("LLM initialization failed")
                    self.loader = None
                    return False

            except Exception as e:
                logger.error(f"LLM initialization error: {e}", exc_info=True)
                self.loader = None
                return False

    def generate(
        self,
        prompt: str,
        task_type: str = "general",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate text with automatic initialization.

        Args:
            prompt: Input prompt
            task_type: Type of task ("split_refinement", "naming", "general")
            max_tokens: Max tokens to generate (None for task default)
            temperature: Sampling temperature (None for task default)
            **kwargs: Additional generation parameters

        Returns:
            Generated text or None if generation fails
        """
        # Lazy initialization
        if not self.loader:
            logger.info("LLM not initialized, initializing now...")
            if not self.initialize():
                logger.error("Failed to initialize LLM for generation")
                return None

        # Check if enabled
        if not self.enabled:
            logger.warning("LLM is disabled")
            return None

        try:
            # Get task-specific parameters
            from services.llm.config import get_generation_params

            params = get_generation_params(task_type)

            # Override with explicit parameters
            if max_tokens is not None:
                params['max_tokens'] = max_tokens
            if temperature is not None:
                params['temperature'] = temperature

            # Merge additional kwargs
            params.update(kwargs)

            # Thread-safe generation
            with self.generation_lock:
                self.stats['total_generations'] += 1

                logger.debug(f"Generating ({task_type}): {prompt[:100]}...")

                response = self.loader.generate(
                    prompt=prompt,
                    **params
                )

                if response:
                    self.stats['successful_generations'] += 1
                    # Rough token count estimate
                    token_estimate = len(response.split())
                    self.stats['total_tokens_generated'] += token_estimate

                    logger.debug(f"Generated: {response[:100]}...")
                    return response
                else:
                    self.stats['failed_generations'] += 1
                    logger.warning("Generation returned None")
                    return None

        except Exception as e:
            self.stats['failed_generations'] += 1
            logger.error(f"Generation error: {e}", exc_info=True)
            return None

    def generate_batch(
        self,
        prompts: List[str],
        task_type: str = "general",
        progress_callback=None,
        **kwargs
    ) -> List[Optional[str]]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            task_type: Type of task
            progress_callback: Optional callback(current, total, message)
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts (None for failed generations)
        """
        results = []

        for i, prompt in enumerate(prompts):
            if progress_callback:
                progress_callback(i, len(prompts), f"Generating {i+1}/{len(prompts)}")

            result = self.generate(
                prompt=prompt,
                task_type=task_type,
                **kwargs
            )
            results.append(result)

        if progress_callback:
            progress_callback(len(prompts), len(prompts), "Batch generation complete")

        return results

    def is_available(self) -> bool:
        """
        Check if LLM is available for use.

        Returns:
            True if LLM can be used
        """
        if not self.enabled:
            return False

        # If not initialized, check if models are available
        if not self.loader:
            from services.resource_path import verify_llm_models
            models = verify_llm_models()
            return models['phi3_mini'] or models['gemma2_2b']

        # If initialized, loader should be valid
        return True

    def is_initialized(self) -> bool:
        """Check if LLM is currently loaded."""
        return self.loader is not None

    def get_info(self) -> Dict[str, Any]:
        """
        Get LLM information and statistics.

        Returns:
            Dictionary with LLM info
        """
        info = {
            'enabled': self.enabled,
            'initialized': self.is_initialized(),
            'available': self.is_available(),
            'stats': self.stats.copy()
        }

        if self.loader:
            model_info = self.loader.get_model_info()
            info.update(model_info)

        return info

    def set_enabled(self, enabled: bool):
        """
        Enable or disable LLM features.

        Args:
            enabled: Whether to enable LLM
        """
        self.enabled = enabled
        logger.info(f"LLM {'enabled' if enabled else 'disabled'}")

    def cleanup(self):
        """
        Clean up LLM resources.

        Call this when done with LLM processing to free GPU memory.
        """
        with self._lock:
            self._cleanup_internal()

    def _cleanup_internal(self):
        """Internal cleanup (assumes lock is held)."""
        if self.loader:
            logger.info("Cleaning up LLM resources...")
            try:
                self.loader.cleanup()
                self.loader = None

                # Force garbage collection
                import gc
                gc.collect()

                # Clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug("CUDA cache cleared")
                except:
                    pass

                logger.info("✓ LLM cleanup complete")

            except Exception as e:
                logger.error(f"Cleanup error: {e}", exc_info=True)

    def reset_stats(self):
        """Reset generation statistics."""
        self.stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_tokens_generated': 0
        }
        logger.info("Statistics reset")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False


# Convenience functions for common operations

def get_llm_manager() -> LLMManager:
    """
    Get the singleton LLM manager instance.

    Returns:
        LLMManager instance
    """
    return LLMManager.get_instance()


def is_llm_available() -> bool:
    """
    Check if LLM is available for use.

    Returns:
        True if LLM can be used
    """
    manager = get_llm_manager()
    return manager.is_available()


def initialize_llm(use_gpu: bool = True, force_reload: bool = False) -> bool:
    """
    Initialize LLM with automatic configuration.

    Args:
        use_gpu: Whether to use GPU acceleration
        force_reload: Force reload even if already loaded

    Returns:
        True if initialization successful
    """
    manager = get_llm_manager()
    return manager.initialize(use_gpu=use_gpu, force_reload=force_reload)


def cleanup_llm():
    """Clean up LLM resources to free GPU memory."""
    manager = get_llm_manager()
    manager.cleanup()


def test_manager():
    """Test function for LLMManager."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("LLMManager Test")
    print("=" * 60)

    # Check availability
    print("\n1. Checking availability...")
    manager = get_llm_manager()
    print(f"  Available: {manager.is_available()}")

    # Initialize
    print("\n2. Initializing LLM...")
    if not manager.initialize(use_gpu=True):
        print("✗ Initialization failed")
        return False

    # Get info
    print("\n3. LLM Information:")
    info = manager.get_info()
    for key, value in info.items():
        if key != 'stats':
            print(f"  {key}: {value}")

    # Test generation
    print("\n4. Testing generation...")
    prompt = "The capital of France is"
    response = manager.generate(prompt, task_type="general", max_tokens=20)

    if response:
        print(f"  Prompt: {prompt}")
        print(f"  Response: {response}")
        print("✓ Generation successful")
    else:
        print("✗ Generation failed")
        return False

    # Test batch generation
    print("\n5. Testing batch generation...")
    prompts = [
        "Paris is the capital of",
        "London is the capital of",
        "Berlin is the capital of"
    ]

    results = manager.generate_batch(
        prompts,
        task_type="general",
        max_tokens=10,
        progress_callback=lambda c, t, m: print(f"  Progress: {m}")
    )

    print("  Results:")
    for prompt, result in zip(prompts, results):
        print(f"    {prompt} → {result}")

    # Show stats
    print("\n6. Statistics:")
    stats = manager.stats
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Cleanup
    print("\n7. Cleaning up...")
    manager.cleanup()

    print("\n" + "=" * 60)
    print("✓ All tests passed")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_manager()
