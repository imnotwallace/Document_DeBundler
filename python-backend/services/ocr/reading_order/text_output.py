"""
Layer 5: Text Output

Generates final ordered text from processed regions.
"""

import logging
from typing import List, Dict, Any
from .data_structures import Region, Block
from .config import ReadingOrderConfig
from .utils import normalize_whitespace

logger = logging.getLogger(__name__)


def generate_ordered_text(regions: List[Region],
                         headers: List[Block],
                         footers: List[Block],
                         config: ReadingOrderConfig) -> str:
    """
    Generate ordered text output from regions.

    Args:
        regions: List of Region instances with reading_order assigned
        headers: List of header blocks
        footers: List of footer blocks
        config: Configuration

    Returns:
        Ordered text string
    """
    logger.info(f"Generating text output from {len(regions)} regions, "
               f"{len(headers)} headers, {len(footers)} footers")

    output_parts = []

    # Process headers first
    if headers:
        sorted_headers = sorted(headers, key=lambda b: b.y0)
        for header in sorted_headers:
            text = header.get_text()
            output_parts.append(text)

        if config.preserve_structure_markers:
            output_parts.append("\n---HEADER END---\n")

    # Process main content in reading order
    sorted_regions = sorted(regions, key=lambda r: r.reading_order or 0)

    for region in sorted_regions:
        # Sort blocks within region by reading order
        sorted_blocks = sorted(region.blocks, key=lambda b: b.reading_order or 0)

        for block in sorted_blocks:
            text = block.get_text()
            output_parts.append(text)

            if config.preserve_structure_markers:
                output_parts.append("\n")  # Block separator

        if config.preserve_structure_markers and len(sorted_regions) > 1:
            output_parts.append("\n---COLUMN END---\n")

    # Process footers last
    if footers:
        if config.preserve_structure_markers:
            output_parts.append("\n---FOOTER START---\n")

        sorted_footers = sorted(footers, key=lambda b: b.y0)
        for footer in sorted_footers:
            text = footer.get_text()
            output_parts.append(text)

    # Join all parts
    final_text = "\n".join(output_parts)

    # Clean up whitespace
    if config.normalize_whitespace:
        final_text = normalize_whitespace(final_text, config.max_consecutive_newlines)

    logger.info(f"Generated text output: {len(final_text)} characters")

    return final_text


def generate_ordered_text_with_coordinates(regions: List[Region],
                                          headers: List[Block],
                                          footers: List[Block]) -> List[Dict[str, Any]]:
    """
    Generate ordered text with coordinate information preserved.

    Useful for applications that need to know where text came from
    in the original document.

    Args:
        regions: List of Region instances with reading_order assigned
        headers: List of header blocks
        footers: List of footer blocks

    Returns:
        List of dictionaries with text and coordinate data
    """
    logger.info("Generating structured output with coordinates")

    output = []

    # Collect all blocks (headers + regions + footers)
    all_blocks = []

    # Headers
    for header in headers:
        header_data = {
            "block": header,
            "category": "header"
        }
        all_blocks.append(header_data)

    # Main content
    for region in regions:
        for block in region.blocks:
            block_data = {
                "block": block,
                "category": "main",
                "region_id": region.region_id
            }
            all_blocks.append(block_data)

    # Footers
    for footer in footers:
        footer_data = {
            "block": footer,
            "category": "footer"
        }
        all_blocks.append(footer_data)

    # Sort by reading order
    all_blocks.sort(key=lambda x: x["block"].reading_order or 0)

    # Generate output
    for item in all_blocks:
        block = item["block"]

        block_data = {
            "text": block.get_text(),
            "bbox": [block.x0, block.y0, block.x1, block.y1],
            "reading_order": block.reading_order,
            "type": block.block_type,
            "category": item["category"],
            "lines": []
        }

        if "region_id" in item:
            block_data["region_id"] = item["region_id"]

        # Add line-level data
        for line in block.lines:
            line_data = {
                "text": line.get_text(),
                "bbox": [line.x0, line.y0, line.x1, line.y1],
                "words": []
            }

            # Add word-level data
            for word in line.words:
                word_data = {
                    "text": word.text,
                    "bbox": [word.x0, word.y0, word.x1, word.y1],
                    "confidence": word.confidence,
                    "_original_index": word.original_index  # Preserve original index for coordinate mapping
                }
                line_data["words"].append(word_data)

            block_data["lines"].append(line_data)

        output.append(block_data)

    logger.info(f"Generated structured output: {len(output)} blocks")

    return output
