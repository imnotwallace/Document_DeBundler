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
    def suggest_name(text_content: str, page_num: int, fallback_prefix: str = "Document") -> str:
        """Suggest a name based on document content"""

        # TODO: Implement intelligent naming
        # - Extract title from first page
        # - Use dates if found
        # - Use keywords
        # - Pattern matching for common document types

        # For now, use simple numbering
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
