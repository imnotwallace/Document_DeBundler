"""
Name Generator - LLM-based Document Naming

Uses LLM to generate intelligent, structured filenames for extracted documents
based on content analysis.
"""

import logging
from typing import Optional, Dict, Any, List
import re

logger = logging.getLogger(__name__)


class NameGenerator:
    """
    LLM-based generator for document filenames.

    Analyzes document content to extract:
    - Date (document date, not today's date)
    - Document type (Invoice, Contract, Report, etc.)
    - Description (parties, subject, identifiers)

    Format: {DATE}_{DOCTYPE}_{DESCRIPTION}
    """

    def __init__(self, cache_manager=None):
        """
        Initialize NameGenerator.

        Args:
            cache_manager: Optional cache manager for caching LLM-generated names
        """
        self.cache = cache_manager
        logger.info("NameGenerator initialized")

    def generate_name(
        self,
        first_page_text: str,
        second_page_text: Optional[str] = None,
        start_page: int = 0,
        end_page: Optional[int] = None,
        fallback_prefix: str = "Document"
    ) -> str:
        """
        Generate intelligent filename using LLM.

        Args:
            first_page_text: Text content of first page
            second_page_text: Optional text content of second page
            start_page: Starting page number
            end_page: Ending page number
            fallback_prefix: Prefix to use if LLM fails

        Returns:
            Generated filename (without .pdf extension)
        """
        try:
            from services.llm.manager import get_llm_manager
            from services.llm.prompts import format_naming_prompt, parse_filename, validate_filename

            # Check if LLM is available
            manager = get_llm_manager()
            if not manager.is_available():
                logger.warning("LLM not available, using fallback naming")
                return self._fallback_name(fallback_prefix, start_page)

            # Check cache first
            if self.cache:
                cached = self._get_cached_name(first_page_text, second_page_text)
                if cached:
                    logger.debug(f"Using cached name: {cached}")
                    return cached

            # Format prompt
            prompt = format_naming_prompt(
                start_page=start_page,
                end_page=end_page or start_page,
                first_page_text=first_page_text,
                second_page_text=second_page_text or ""
            )

            logger.info(f"Generating name for document (pages {start_page}-{end_page or start_page})...")

            # Generate name
            response = manager.generate(
                prompt=prompt,
                task_type="naming"
            )

            if not response:
                logger.error("LLM returned empty response")
                return self._fallback_name(fallback_prefix, start_page)

            # Parse and validate filename
            filename = parse_filename(response)

            if not filename:
                logger.warning("Failed to parse filename from LLM response")
                return self._fallback_name(fallback_prefix, start_page)

            # Validate format
            if not validate_filename(filename):
                logger.warning(f"Generated filename doesn't match format: {filename}")
                # Try to fix it
                filename = self._fix_filename(filename, fallback_prefix, start_page)

            logger.info(f"Generated name: {filename}")

            # Cache the name
            if self.cache:
                self._cache_name(first_page_text, second_page_text, filename)

            return filename

        except Exception as e:
            logger.error(f"Name generation error: {e}", exc_info=True)
            return self._fallback_name(fallback_prefix, start_page)

    def generate_names_batch(
        self,
        documents: List[Dict[str, Any]],
        progress_callback=None
    ) -> List[str]:
        """
        Generate names for multiple documents.

        Args:
            documents: List of document dicts with 'first_page_text', 'second_page_text', etc.
            progress_callback: Optional callback(current, total, message)

        Returns:
            List of generated filenames
        """
        names = []

        for i, doc in enumerate(documents):
            if progress_callback:
                progress_callback(
                    i,
                    len(documents),
                    f"LLM naming document {i+1}/{len(documents)}"
                )

            name = self.generate_name(
                first_page_text=doc.get('first_page_text', ''),
                second_page_text=doc.get('second_page_text'),
                start_page=doc.get('start_page', i),
                end_page=doc.get('end_page'),
                fallback_prefix=doc.get('fallback_prefix', 'Document')
            )

            names.append(name)

        if progress_callback:
            progress_callback(
                len(documents),
                len(documents),
                "LLM naming complete"
            )

        return names

    def _fallback_name(self, prefix: str, page_num: int) -> str:
        """
        Generate fallback name when LLM unavailable.

        Args:
            prefix: Prefix for filename
            page_num: Page number

        Returns:
            Fallback filename
        """
        from datetime import datetime

        # Format: UNDATED_Other_{Prefix} {page}
        filename = f"UNDATED_Other_{prefix} {page_num + 1}"

        logger.debug(f"Using fallback name: {filename}")
        return filename

    def _fix_filename(self, filename: str, fallback_prefix: str, page_num: int) -> str:
        """
        Attempt to fix malformed filename.

        Args:
            filename: Malformed filename
            fallback_prefix: Fallback prefix
            page_num: Page number

        Returns:
            Fixed filename or fallback
        """
        try:
            # Split on underscores
            parts = filename.split('_')

            # Need at least 3 parts
            if len(parts) < 3:
                logger.warning(f"Cannot fix filename (too few parts): {filename}")
                return self._fallback_name(fallback_prefix, page_num)

            date_part = parts[0]
            doctype_part = parts[1]
            description_parts = parts[2:]

            # Fix date
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_part) and date_part != "UNDATED":
                date_part = "UNDATED"

            # Fix doctype (single word)
            if not re.match(r'^\w+$', doctype_part):
                doctype_part = "Other"

            # Fix description (join remaining parts)
            description = ' '.join(description_parts)
            # Clean it
            description = re.sub(r'[^\w\s\-]', '', description)
            description = re.sub(r'\s+', ' ', description).strip()

            # Ensure 2-5 words
            words = description.split()
            if len(words) < 2:
                description = f"{fallback_prefix} Document"
            elif len(words) > 5:
                description = ' '.join(words[:5])

            fixed = f"{date_part}_{doctype_part}_{description}"
            logger.info(f"Fixed filename: {filename} → {fixed}")

            return fixed

        except Exception as e:
            logger.error(f"Error fixing filename: {e}")
            return self._fallback_name(fallback_prefix, page_num)

    def _get_cached_name(
        self,
        first_page_text: str,
        second_page_text: Optional[str]
    ) -> Optional[str]:
        """Get cached LLM-generated name if available."""
        if not self.cache:
            return None

        try:
            import hashlib

            # Create cache key from text content
            content = first_page_text[:2000]
            if second_page_text:
                content += second_page_text[:1000]

            cache_key = f"llm_name_{hashlib.md5(content.encode()).hexdigest()}"

            # Try to get from cache
            cached = self.cache.get(cache_key)

            if cached:
                return cached.get('filename')

        except Exception as e:
            logger.debug(f"Cache lookup failed: {e}")

        return None

    def _cache_name(
        self,
        first_page_text: str,
        second_page_text: Optional[str],
        filename: str
    ):
        """Cache LLM-generated name for future use."""
        if not self.cache:
            return

        try:
            import hashlib

            content = first_page_text[:2000]
            if second_page_text:
                content += second_page_text[:1000]

            cache_key = f"llm_name_{hashlib.md5(content.encode()).hexdigest()}"

            # Cache the name
            self.cache.set(cache_key, {
                'filename': filename
            }, ttl=86400 * 30)  # 30 days

            logger.debug(f"Cached name: {filename}")

        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    def extract_metadata(self, filename: str) -> Dict[str, str]:
        """
        Extract metadata from generated filename.

        Args:
            filename: Filename in {DATE}_{DOCTYPE}_{DESCRIPTION} format

        Returns:
            Dictionary with 'date', 'doctype', 'description'
        """
        parts = filename.split('_', 2)

        if len(parts) != 3:
            return {
                'date': 'UNDATED',
                'doctype': 'Other',
                'description': filename
            }

        return {
            'date': parts[0],
            'doctype': parts[1],
            'description': parts[2]
        }


def test_name_generator():
    """Test function for NameGenerator."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("NameGenerator Test")
    print("=" * 60)

    # Mock document texts
    invoice_text = """
    INVOICE

    Invoice Number: INV-2024-001
    Date: January 15, 2024

    Bill To:
    Acme Corporation
    123 Business St
    New York, NY 10001

    Services Rendered:
    Consulting Services - Q4 2023
    """

    contract_text = """
    SOFTWARE LICENSE AGREEMENT

    This Agreement is entered into as of March 20, 2024
    between TechCorp Inc. ("Licensor") and Client Company ("Licensee").

    1. Grant of License
    Licensor hereby grants to Licensee a non-exclusive...
    """

    # Create generator
    generator = NameGenerator()

    # Test invoice naming
    print("\n1. Testing invoice naming...")
    invoice_name = generator.generate_name(
        first_page_text=invoice_text,
        start_page=0,
        end_page=2,
        fallback_prefix="Invoice"
    )
    print(f"  Generated: {invoice_name}")

    # Extract metadata
    metadata = generator.extract_metadata(invoice_name)
    print(f"  Metadata: {metadata}")

    # Test contract naming
    print("\n2. Testing contract naming...")
    contract_name = generator.generate_name(
        first_page_text=contract_text,
        start_page=0,
        end_page=5,
        fallback_prefix="Contract"
    )
    print(f"  Generated: {contract_name}")
    metadata = generator.extract_metadata(contract_name)
    print(f"  Metadata: {metadata}")

    # Test batch naming
    print("\n3. Testing batch naming...")
    documents = [
        {
            'first_page_text': invoice_text,
            'start_page': 0,
            'end_page': 2,
            'fallback_prefix': 'Invoice'
        },
        {
            'first_page_text': contract_text,
            'start_page': 3,
            'end_page': 7,
            'fallback_prefix': 'Contract'
        }
    ]

    names = generator.generate_names_batch(
        documents,
        progress_callback=lambda c, t, m: print(f"  {m}")
    )

    print("  Generated names:")
    for name in names:
        print(f"    - {name}")

    print("\n" + "=" * 60)
    print("✓ Test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_name_generator()
