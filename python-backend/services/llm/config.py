"""
LLM Configuration with VRAM-based Model Selection
Optimized for 4GB VRAM target hardware
"""

import logging
from typing import Dict, Any, Optional
from services.ocr.config import detect_hardware_capabilities

logger = logging.getLogger(__name__)


def select_optimal_llm_config(gpu_memory_gb: Optional[float] = None) -> Dict[str, Any]:
    """
    Automatically select best LLM configuration for available VRAM.

    Args:
        gpu_memory_gb: Manual override for GPU memory (auto-detected if None)

    Returns:
        LLM configuration dictionary with model settings
    """
    if gpu_memory_gb is None:
        capabilities = detect_hardware_capabilities()
        gpu_memory_gb = capabilities['gpu_memory_gb']

    logger.info(f"Detecting optimal LLM for {gpu_memory_gb:.1f}GB VRAM")

    if gpu_memory_gb >= 8:
        # High-end GPU: Better quality
        return {
            'model_name': 'Phi-3 Mini (High Quality)',
            'model_id': 'microsoft/Phi-3-mini-4k-instruct-gguf',
            'model_file': 'Phi-3-mini-4k-instruct-q5_k_m.gguf',
            'quantization': 'Q5_K_M',
            'n_gpu_layers': 32,
            'n_batch': 512,
            'n_ctx': 4096,
            'expected_vram_gb': 3.2,
            'offload_strategy': 'gpu_only'
        }
    elif gpu_memory_gb >= 6:
        # Mid-range GPU
        return {
            'model_name': 'Phi-3 Mini (Balanced)',
            'model_id': 'microsoft/Phi-3-mini-4k-instruct-gguf',
            'model_file': 'Phi-3-mini-4k-instruct-q4_k_m.gguf',
            'quantization': 'Q4_K_M',
            'n_gpu_layers': 32,
            'n_batch': 512,
            'n_ctx': 4096,
            'expected_vram_gb': 2.5,
            'offload_strategy': 'gpu_only'
        }
    elif gpu_memory_gb >= 4:
        # TARGET: 4GB VRAM
        return {
            'model_name': 'Phi-3 Mini (4GB Optimized)',
            'model_id': 'microsoft/Phi-3-mini-4k-instruct-gguf',
            'model_file': 'Phi-3-mini-4k-instruct-q4_k_m.gguf',
            'quantization': 'Q4_K_M',
            'n_gpu_layers': 28,  # 28 GPU + 4 CPU layers
            'n_batch': 256,
            'n_ctx': 4096,
            'expected_vram_gb': 2.3,
            'offload_strategy': 'hybrid',
            'cpu_layers': 4
        }
    elif gpu_memory_gb >= 2:
        # Low VRAM: Lighter model
        return {
            'model_name': 'Gemma 2 2B',
            'model_id': 'google/gemma-2-2b-it-gguf',
            'model_file': 'gemma-2-2b-it-q4_k_m.gguf',
            'quantization': 'Q4_K_M',
            'n_gpu_layers': 20,
            'n_batch': 256,
            'n_ctx': 4096,
            'expected_vram_gb': 1.5,
            'offload_strategy': 'hybrid',
            'cpu_layers': 8
        }
    else:
        # CPU-only fallback
        return {
            'model_name': 'Phi-3 Mini (CPU)',
            'model_id': 'microsoft/Phi-3-mini-4k-instruct-gguf',
            'model_file': 'Phi-3-mini-4k-instruct-q4_k_m.gguf',
            'quantization': 'Q4_K_M',
            'n_gpu_layers': 0,
            'n_batch': 128,
            'n_ctx': 4096,
            'expected_vram_gb': 0,
            'offload_strategy': 'cpu_only',
            'warning': 'LLM running on CPU - will be slow (5-10x slower)'
        }


def get_model_download_info(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Get model download information.

    Args:
        config: LLM configuration from select_optimal_llm_config()

    Returns:
        Dictionary with download URLs and paths
    """
    # HuggingFace download URLs
    base_url = f"https://huggingface.co/{config['model_id']}/resolve/main"

    return {
        'model_url': f"{base_url}/{config['model_file']}",
        'model_file': config['model_file'],
        'expected_size_gb': config['expected_vram_gb']
    }


def get_generation_params(task_type: str = "split_refinement") -> Dict[str, Any]:
    """
    Get generation parameters optimized for specific tasks.

    Args:
        task_type: Type of task ("split_refinement" or "naming")

    Returns:
        Generation parameters for llama-cpp-python
    """
    if task_type == "split_refinement":
        # Split refinement: Need precise YES/NO answer
        return {
            'max_tokens': 150,
            'temperature': 0.1,  # Very low for consistent YES/NO
            'top_p': 0.9,
            'top_k': 20,
            'repeat_penalty': 1.1,
            'stop': ['\n\n', 'Question:', 'Context:'],
        }
    elif task_type == "naming":
        # Document naming: Need creative but structured output
        return {
            'max_tokens': 100,
            'temperature': 0.3,  # Low but allows some variation
            'top_p': 0.9,
            'top_k': 40,
            'repeat_penalty': 1.15,
            'stop': ['\n\n', '.pdf', 'Examples:', 'Output:'],
        }
    else:
        # Default conservative settings
        return {
            'max_tokens': 200,
            'temperature': 0.2,
            'top_p': 0.9,
            'top_k': 40,
            'repeat_penalty': 1.1,
        }


def estimate_inference_time(
    config: Dict[str, Any],
    num_tokens: int = 100
) -> float:
    """
    Estimate inference time based on configuration.

    Args:
        config: LLM configuration
        num_tokens: Number of tokens to generate

    Returns:
        Estimated time in seconds
    """
    # Base speeds (tokens per second)
    if config['offload_strategy'] == 'gpu_only':
        if config['expected_vram_gb'] >= 3:
            tokens_per_sec = 50  # High quality on 8GB VRAM
            prompt_overhead = 1.0  # Prompt processing time
        else:
            tokens_per_sec = 40  # Balanced on 6GB VRAM
            prompt_overhead = 1.5
    elif config['offload_strategy'] == 'hybrid':
        tokens_per_sec = 25  # 4GB VRAM hybrid mode
        prompt_overhead = 2.0  # More overhead due to layer switching
    else:
        tokens_per_sec = 5  # CPU-only (much slower)
        prompt_overhead = 5.0  # Significant prompt processing on CPU

    # Total time = prompt processing + token generation
    generation_time = num_tokens / tokens_per_sec
    return generation_time + prompt_overhead
