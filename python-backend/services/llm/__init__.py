"""
LLM Module for Document De-bundling

This module provides LLM-based analysis for document de-bundling:
- Split refinement (validating detected boundaries)
- Document naming (generating descriptive filenames)
- VRAM-optimized model selection
"""

from .config import select_optimal_llm_config, get_model_download_info
from .prompts import (
    SPLIT_REFINEMENT_PROMPT,
    DOCUMENT_NAMING_PROMPT,
    format_split_prompt,
    format_naming_prompt,
)

__all__ = [
    'select_optimal_llm_config',
    'get_model_download_info',
    'SPLIT_REFINEMENT_PROMPT',
    'DOCUMENT_NAMING_PROMPT',
    'format_split_prompt',
    'format_naming_prompt',
]
