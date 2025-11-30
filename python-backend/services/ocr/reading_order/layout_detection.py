"""
Layer 3.1: Layout Type Detection

Determines the overall layout type of the page to guide processing strategy.
"""

import logging
from typing import List
from .data_structures import WordBox
from .config import ReadingOrderConfig
from .utils import group_by_vertical_position

logger = logging.getLogger(__name__)


def detect_layout_type(word_boxes: List[WordBox],
                      page_width: float,
                      page_height: float,
                      config: ReadingOrderConfig) -> str:
    """
    Detect the layout type of the page.

    Returns one of:
    - "SINGLE_COLUMN": Single column of text
    - "MULTI_COLUMN": Multiple columns (newspaper, etc.)
    - "TABLE": Tabular structure
    - "COMPLEX": Mixed or complex layout

    Args:
        word_boxes: List of WordBox instances
        page_width, page_height: Page dimensions
        config: Configuration

    Returns:
        Layout type string
    """
    if not word_boxes:
        return "SINGLE_COLUMN"

    logger.info(f"Detecting layout type for {len(word_boxes)} words")

    # Analyze horizontal distribution
    left_third = page_width / 3
    right_third = page_width * 2 / 3

    left_count = sum(1 for w in word_boxes if w.center_x < left_third)
    middle_count = sum(1 for w in word_boxes if left_third <= w.center_x < right_third)
    right_count = sum(1 for w in word_boxes if w.center_x >= right_third)

    total_count = len(word_boxes)

    left_ratio = left_count / total_count
    middle_ratio = middle_count / total_count
    right_ratio = right_count / total_count

    logger.debug(f"Horizontal distribution: left={left_ratio:.2f}, middle={middle_ratio:.2f}, right={right_ratio:.2f}")

    # Decision tree

    # Single column: most words in center third
    if middle_ratio > config.single_column_center_ratio:
        layout_type = "SINGLE_COLUMN"

    # Multi-column: significant words on both left and right, few in middle
    elif (left_ratio > config.multi_column_side_ratio and
          right_ratio > config.multi_column_side_ratio and
          middle_ratio < 0.2):
        layout_type = "MULTI_COLUMN"

    # Check for table structure
    elif is_table_structure(word_boxes, config):
        layout_type = "TABLE"

    else:
        # Complex or ambiguous
        layout_type = "COMPLEX"

    logger.info(f"Detected layout type: {layout_type}")

    return layout_type


def is_table_structure(word_boxes: List[WordBox],
                      config: ReadingOrderConfig) -> bool:
    """
    Detect if the layout is a table.

    Tables have:
    - Multiple rows
    - Consistent multi-column structure across rows

    Args:
        word_boxes: List of WordBox instances
        config: Configuration

    Returns:
        True if table structure detected
    """
    if len(word_boxes) < config.table_min_rows * config.table_min_columns:
        return False

    # Group words by vertical position (rows)
    vertical_bands = group_by_vertical_position(word_boxes, tolerance=10)

    if len(vertical_bands) < config.table_min_rows:
        return False

    # Check if bands have consistent multi-column structure
    multi_column_bands = 0

    for band in vertical_bands:
        # Count "columns" in this band by horizontal gaps
        if len(band) >= config.table_min_columns:
            # Sort by X
            sorted_band = sorted(band, key=lambda w: w.x0)

            # Count gaps that indicate column boundaries
            gaps = []
            for i in range(len(sorted_band) - 1):
                gap = sorted_band[i+1].x0 - sorted_band[i].x1
                gaps.append(gap)

            # Large gaps indicate columns
            if gaps:
                median_gap = sorted(gaps)[len(gaps) // 2]
                large_gaps = sum(1 for g in gaps if g > median_gap * 2)

                # If multiple large gaps, this is a multi-column row
                if large_gaps >= config.table_min_columns - 1:
                    multi_column_bands += 1

    # If most rows are multi-column, it's a table
    table_ratio = multi_column_bands / len(vertical_bands)

    is_table = table_ratio > config.table_row_threshold

    if is_table:
        logger.debug(f"Table structure detected: {multi_column_bands}/{len(vertical_bands)} "
                    f"rows are multi-column ({table_ratio:.2f})")

    return is_table
