"""
LLM Loader Service

Handles loading and initialization of llama.cpp models with dual integration:
- Primary: llama-cpp-python (Python bindings)
- Fallback: Standalone llama.cpp binary (subprocess)

Optimized for 4GB VRAM + 16GB RAM target hardware.
"""

import logging
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import psutil

logger = logging.getLogger(__name__)


class LlamaLoader:
    """
    Manages llama.cpp model loading with Python bindings and binary fallback.

    Features:
    - Auto-detection of bundled vs downloaded models
    - GPU layer optimization based on VRAM
    - Memory monitoring during initialization
    - Graceful fallback to binary if Python bindings fail
    """

    def __init__(
        self,
        use_gpu: bool = True,
        use_binary_fallback: bool = True,
        verbose: bool = False
    ):
        """
        Initialize LlamaLoader.

        Args:
            use_gpu: Whether to use GPU acceleration
            use_binary_fallback: Fall back to standalone binary if Python bindings fail
            verbose: Enable verbose logging
        """
        self.use_gpu = use_gpu
        self.use_binary_fallback = use_binary_fallback
        self.verbose = verbose

        self.llm = None
        self.model_path = None
        self.config = None
        self.binary_process = None
        self.mode = None  # 'python' or 'binary'

        logger.info(f"LlamaLoader initialized (GPU: {use_gpu}, Binary fallback: {use_binary_fallback})")

    def load_model(
        self,
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Load LLM model using optimal configuration.

        Args:
            model_name: Specific model to load ('phi3_mini' or 'gemma2_2b'), None for auto
            config: Manual configuration override, None for auto-detection

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Get configuration
            if config is None:
                from services.llm.config import select_optimal_llm_config
                from services.ocr.config import detect_hardware_capabilities

                capabilities = detect_hardware_capabilities()
                gpu_memory_gb = capabilities['gpu_memory_gb'] if self.use_gpu else 0

                self.config = select_optimal_llm_config(gpu_memory_gb)
                logger.info(f"Auto-selected configuration: {self.config['model_name']}")
            else:
                self.config = config

            # Get model path
            self.model_path = self._find_model_path(model_name)

            if not self.model_path:
                logger.error("No LLM model found. Run download_llm_models.py first.")
                return False

            logger.info(f"Loading model from: {self.model_path}")
            logger.info(f"Expected VRAM: {self.config['expected_vram_gb']:.1f}GB")

            # Try Python bindings first
            if self._load_with_python_bindings():
                self.mode = 'python'
                logger.info("✓ Model loaded successfully with Python bindings")
                return True

            # Fallback to binary if enabled
            if self.use_binary_fallback:
                logger.warning("Python bindings failed, trying binary fallback...")
                if self._load_with_binary():
                    self.mode = 'binary'
                    logger.info("✓ Model loaded successfully with binary fallback")
                    return True

            logger.error("Failed to load model with all methods")
            return False

        except Exception as e:
            logger.error(f"Model loading failed: {e}", exc_info=True)
            return False

    def _find_model_path(self, model_name: Optional[str] = None) -> Optional[Path]:
        """
        Find model file path (bundled or downloaded).

        Args:
            model_name: Specific model name or None for auto-detection

        Returns:
            Path to model file or None if not found
        """
        from services.resource_path import (
            get_phi3_mini_path,
            get_gemma2_2b_path,
            verify_llm_models
        )

        # Check which models are available
        available = verify_llm_models()

        if model_name == 'phi3_mini' or model_name is None:
            if available['phi3_mini']:
                return get_phi3_mini_path()

        if model_name == 'gemma2_2b' or model_name is None:
            if available['gemma2_2b']:
                return get_gemma2_2b_path()

        # If specific model requested but not found
        if model_name:
            logger.warning(f"Requested model '{model_name}' not found")

        # Try any available model
        if available['phi3_mini']:
            return get_phi3_mini_path()
        if available['gemma2_2b']:
            return get_gemma2_2b_path()

        return None

    def _load_with_python_bindings(self) -> bool:
        """
        Load model using llama-cpp-python bindings.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Import llama-cpp-python
            try:
                from llama_cpp import Llama
            except ImportError as e:
                logger.warning(f"llama-cpp-python not available: {e}")
                return False

            # Check memory before loading
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            logger.info(f"Available RAM: {available_ram_gb:.1f}GB")

            if available_ram_gb < 4:
                logger.warning(f"Low RAM ({available_ram_gb:.1f}GB), model may fail to load")

            # Determine GPU layers
            n_gpu_layers = self.config.get('n_gpu_layers', 0) if self.use_gpu else 0

            logger.info(f"Loading with {n_gpu_layers} GPU layers...")

            # Load model
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.config.get('n_ctx', 4096),
                n_batch=self.config.get('n_batch', 512),
                n_gpu_layers=n_gpu_layers,
                n_threads=self.config.get('n_threads', psutil.cpu_count() // 2),
                use_mlock=False,  # Don't lock memory (can cause issues on Windows)
                use_mmap=True,    # Use memory mapping for efficiency
                verbose=self.verbose,
                # For newer llama-cpp-python versions
                flash_attn=False,  # Disable flash attention (can be unstable)
            )

            logger.info(f"Model loaded: {self.llm.model_path}")
            logger.info(f"Context size: {self.llm.n_ctx()}")

            # Test generation to verify
            test_output = self.llm(
                prompt="Test",
                max_tokens=5,
                temperature=0.1,
                echo=False
            )

            logger.debug(f"Test generation successful: {test_output}")

            return True

        except Exception as e:
            logger.error(f"Python bindings loading failed: {e}", exc_info=True)
            self.llm = None
            return False

    def _load_with_binary(self) -> bool:
        """
        Load model using standalone llama.cpp binary (server mode).

        Returns:
            True if successful, False otherwise
        """
        try:
            from services.resource_path import get_llama_cpp_binary_path, verify_llama_cpp_binary

            # Check binary availability
            success, message = verify_llama_cpp_binary()
            if not success:
                logger.warning(f"Binary not available: {message}")
                return False

            binary_path = get_llama_cpp_binary_path()
            logger.info(f"Using binary: {binary_path}")

            # Construct server command
            n_gpu_layers = self.config.get('n_gpu_layers', 0) if self.use_gpu else 0

            cmd = [
                str(binary_path),
                "--model", str(self.model_path),
                "--ctx-size", str(self.config.get('n_ctx', 4096)),
                "--batch-size", str(self.config.get('n_batch', 512)),
                "--n-gpu-layers", str(n_gpu_layers),
                "--threads", str(self.config.get('n_threads', psutil.cpu_count() // 2)),
                "--port", "8080",  # Default port
                "--host", "127.0.0.1",  # Localhost only
            ]

            logger.info(f"Starting llama-server: {' '.join(cmd)}")

            # Start server process
            self.binary_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for server to start (check for "HTTP server listening" in output)
            import time
            max_wait = 30  # seconds
            start_time = time.time()

            while time.time() - start_time < max_wait:
                # Check if process crashed
                if self.binary_process.poll() is not None:
                    stderr = self.binary_process.stderr.read()
                    logger.error(f"Server process crashed: {stderr}")
                    return False

                # Try to connect to server
                try:
                    import requests
                    response = requests.get("http://127.0.0.1:8080/health", timeout=1)
                    if response.status_code == 200:
                        logger.info("✓ llama-server ready")
                        return True
                except:
                    time.sleep(1)

            logger.error("Server failed to start within timeout")
            self._cleanup_binary()
            return False

        except Exception as e:
            logger.error(f"Binary loading failed: {e}", exc_info=True)
            self._cleanup_binary()
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            **kwargs: Additional generation parameters

        Returns:
            Generated text or None if generation fails
        """
        try:
            if self.mode == 'python':
                return self._generate_python(prompt, max_tokens, temperature, stop, **kwargs)
            elif self.mode == 'binary':
                return self._generate_binary(prompt, max_tokens, temperature, stop, **kwargs)
            else:
                logger.error("No model loaded")
                return None
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return None

    def _generate_python(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]],
        **kwargs
    ) -> Optional[str]:
        """Generate using Python bindings."""
        if not self.llm:
            return None

        output = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
            echo=False,
            **kwargs
        )

        # Extract text from output
        if isinstance(output, dict) and 'choices' in output:
            return output['choices'][0]['text'].strip()

        return str(output).strip()

    def _generate_binary(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]],
        **kwargs
    ) -> Optional[str]:
        """Generate using binary server."""
        try:
            import requests

            response = requests.post(
                "http://127.0.0.1:8080/completion",
                json={
                    "prompt": prompt,
                    "n_predict": max_tokens,
                    "temperature": temperature,
                    "stop": stop or [],
                    **kwargs
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('content', '').strip()
            else:
                logger.error(f"Server returned error: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Binary generation failed: {e}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded model.

        Returns:
            Dictionary with model information
        """
        if not self.model_path or not self.config:
            return {
                'loaded': False,
                'error': 'No model loaded'
            }

        return {
            'loaded': True,
            'mode': self.mode,
            'model_path': str(self.model_path),
            'model_name': self.config['model_name'],
            'gpu_enabled': self.use_gpu,
            'gpu_layers': self.config.get('n_gpu_layers', 0),
            'context_size': self.config.get('n_ctx', 4096),
            'expected_vram_gb': self.config.get('expected_vram_gb', 0),
            'offload_strategy': self.config.get('offload_strategy', 'unknown')
        }

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.mode == 'python' and self.llm:
                # llama-cpp-python cleanup
                del self.llm
                self.llm = None
                logger.info("Python bindings cleaned up")

            elif self.mode == 'binary':
                self._cleanup_binary()

            # Force garbage collection
            import gc
            gc.collect()

            # Try to free GPU memory if using CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared")
            except:
                pass

            self.mode = None
            logger.info("LLM cleanup complete")

        except Exception as e:
            logger.error(f"Cleanup error: {e}", exc_info=True)

    def _cleanup_binary(self):
        """Clean up binary server process."""
        if self.binary_process:
            try:
                self.binary_process.terminate()
                self.binary_process.wait(timeout=5)
                logger.info("Binary server terminated")
            except:
                try:
                    self.binary_process.kill()
                    logger.warning("Binary server killed (force)")
                except:
                    pass
            finally:
                self.binary_process = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False


def test_loader():
    """Test function for LlamaLoader."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("LlamaLoader Test")
    print("=" * 60)

    with LlamaLoader(use_gpu=True) as loader:
        # Load model
        print("\n1. Loading model...")
        if not loader.load_model():
            print("✗ Failed to load model")
            return False

        # Get model info
        print("\n2. Model information:")
        info = loader.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Test generation
        print("\n3. Testing generation...")
        prompt = "The capital of France is"
        response = loader.generate(prompt, max_tokens=20, temperature=0.1)

        if response:
            print(f"  Prompt: {prompt}")
            print(f"  Response: {response}")
            print("✓ Generation successful")
        else:
            print("✗ Generation failed")
            return False

    print("\n" + "=" * 60)
    print("✓ All tests passed")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_loader()
