"""
Naming Service
Generates intelligent names for split documents
"""

import logging
from typing import List, Optional
import re

logger = logging.getLogger(__name__)


class NamingService:
    """Generates meaningful names for split documents"""

    @staticmethod
    def suggest_name(
        text_content: str,
        page_num: int,
        fallback_prefix: str = "Document",
        second_page_text: Optional[str] = None,
        use_llm: bool = True
    ) -> str:
        """
        Suggest a name based on document content.

        Uses LLM for intelligent naming if available, with heuristic fallback.

        Args:
            text_content: First page text content
            page_num: Page number for fallback naming
            fallback_prefix: Prefix for fallback naming
            second_page_text: Optional second page text for better context
            use_llm: Whether to attempt LLM naming (default: True)

        Returns:
            Suggested filename (without .pdf extension)
        """
        # Try LLM naming first if enabled
        if use_llm:
            try:
                from .llm.name_generator import NameGenerator
                from .llm.settings import get_settings

                settings = get_settings()

                # Check if LLM naming is enabled
                if settings.enabled and settings.naming_enabled:
                    logger.debug("Attempting LLM-based naming...")

                    generator = NameGenerator()
                    llm_name = generator.generate_name(
                        first_page_text=text_content,
                        second_page_text=second_page_text,
                        start_page=page_num,
                        fallback_prefix=fallback_prefix
                    )

                    if llm_name and llm_name != generator._fallback_name(fallback_prefix, page_num):
                        logger.info(f"LLM generated name: {llm_name}")
                        return llm_name
                    else:
                        logger.debug("LLM returned fallback, using heuristic")
                else:
                    logger.debug("LLM naming disabled in settings")

            except Exception as e:
                logger.warning(f"LLM naming failed, using heuristic: {e}")

        # Heuristic fallback
        return NamingService._heuristic_name(text_content, page_num, fallback_prefix)

    @staticmethod
    def _heuristic_name(text_content: str, page_num: int, fallback_prefix: str = "Document") -> str:
        """
        Generate name using heuristic rules (fallback method).

        Args:
            text_content: First page text
            page_num: Page number
            fallback_prefix: Fallback prefix

        Returns:
            Filename generated from heuristics
        """
        clean_text = NamingService._clean_text(text_content)

        if len(clean_text) > 10:
            # Try to extract first meaningful line as title
            first_line = clean_text.split('\n')[0][:50]
            if first_line:
                safe_name = NamingService._make_safe_filename(first_line)
                if safe_name:
                    return safe_name

        # Fallback to numbered naming
        return f"{fallback_prefix}_{page_num:04d}"

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text for processing"""
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def _make_safe_filename(name: str) -> str:
        """Convert text to safe filename"""
        # Remove invalid filename characters
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        # Replace spaces with underscores
        name = name.replace(' ', '_')
        # Remove multiple underscores
        name = re.sub(r'_+', '_', name)
        # Trim
        name = name.strip('_')
        return name[:100]  # Limit length
