"""
Layer 3.3: Special Zone Classification

Classifies blocks as headers, footers, or main content
based on vertical position on page.
"""

import logging
from typing import List, Tuple
from .data_structures import Block
from .config import ReadingOrderConfig

logger = logging.getLogger(__name__)


def classify_special_zones(blocks: List[Block],
                          page_height: float,
                          config: ReadingOrderConfig) -> Tuple[List[Block], List[Block], List[Block]]:
    """
    Classify blocks into headers, main content, and footers.

    Based on vertical position:
    - Top N% of page: headers
    - Bottom N% of page: footers
    - Middle: main content

    Args:
        blocks: List of Block instances
        page_height: Page height
        config: Configuration with zone ratios

    Returns:
        Tuple of (headers, main_blocks, footers)
    """
    if not blocks:
        return ([], [], [])

    logger.info(f"Classifying {len(blocks)} blocks into zones")

    # Define zone boundaries
    header_threshold = page_height * config.header_zone_ratio
    footer_threshold = page_height * config.footer_zone_ratio

    logger.debug(f"Header zone: Y < {header_threshold:.0f}px ({config.header_zone_ratio:.0%} of page)")
    logger.debug(f"Footer zone: Y > {footer_threshold:.0f}px ({config.footer_zone_ratio:.0%} of page)")

    headers = []
    main_blocks = []
    footers = []

    for block in blocks:
        # Classify based on vertical position
        # Use bottom of block (y1) for headers, top of block (y0) for footers

        if block.y1 < header_threshold:
            # Block ends in header zone
            block.block_type = "header"
            headers.append(block)
        elif block.y0 > footer_threshold:
            # Block starts in footer zone
            block.block_type = "footer"
            footers.append(block)
        else:
            # Block in main content area
            block.block_type = "main"
            main_blocks.append(block)

    logger.info(f"Classified: {len(headers)} headers, {len(main_blocks)} main, {len(footers)} footers")

    return (headers, main_blocks, footers)


def detect_table_regions(blocks: List[Block], config: ReadingOrderConfig) -> List[Block]:
    """
    Detect blocks that are tables.

    Tables have regular row/column structure with consistent spacing.

    Args:
        blocks: List of Block instances
        config: Configuration

    Returns:
        List of blocks classified as tables
    """
    table_blocks = []

    for block in blocks:
        if _is_table_block(block, config):
            block.block_type = "table"
            table_blocks.append(block)

    if table_blocks:
        logger.info(f"Detected {len(table_blocks)} table blocks")

    return table_blocks


def _is_table_block(block: Block, config: ReadingOrderConfig) -> bool:
    """
    Check if a block is a table.

    Tables have:
    - Multiple lines
    - Consistent multi-segment structure (columns) across lines
    - Regular spacing

    Args:
        block: Block to check
        config: Configuration

    Returns:
        True if block appears to be a table
    """
    lines = block.lines

    if len(lines) < config.table_min_rows:
        return False

    # Check if lines have consistent multi-segment structure
    segments_per_line = []

    for line in lines:
        segments = _count_horizontal_segments(line.words, config)
        segments_per_line.append(segments)

    # Calculate average segments per line
    avg_segments = sum(segments_per_line) / len(segments_per_line)

    # If most lines have multiple segments, likely a table
    is_table = avg_segments >= config.table_min_columns

    if is_table:
        logger.debug(f"Table block detected: avg_segments={avg_segments:.1f}, "
                    f"lines={len(lines)}")

    return is_table


def _count_horizontal_segments(words: List, config: ReadingOrderConfig) -> int:
    """
    Count distinct horizontal groups (columns) in a line.

    Large gaps between words indicate column boundaries.

    Args:
        words: List of WordBox instances in the line
        config: Configuration

    Returns:
        Number of segments (columns) in the line
    """
    if len(words) <= 1:
        return 1

    # Sort words left to right
    sorted_words = sorted(words, key=lambda w: w.x0)

    # Calculate average word width
    avg_width = sum(w.width for w in sorted_words) / len(sorted_words)

    # Count segments based on horizontal gaps
    segments = 1

    for i in range(len(sorted_words) - 1):
        gap = sorted_words[i+1].x0 - sorted_words[i].x1

        # Large gap = new segment (column)
        gap_threshold = avg_width * config.table_horizontal_gap_multiplier

        if gap > gap_threshold:
            segments += 1

    return segments
