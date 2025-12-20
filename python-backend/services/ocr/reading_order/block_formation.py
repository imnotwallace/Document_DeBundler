"""
Layer 2.2: Block Formation

Groups lines into blocks (paragraphs/sections) using vertical
proximity and horizontal alignment.
"""

import logging
from typing import List
from .data_structures import Line, Block
from .config import ReadingOrderConfig
from .utils import calculate_horizontal_overlap, get_average_line_height

logger = logging.getLogger(__name__)


def form_blocks(lines: List[Line],
               page_width: float,
               page_height: float,
               config: ReadingOrderConfig,
               y_variance_ratio: float = 0.0) -> List[Block]:
    """
    Group lines into blocks using vertical proximity and horizontal alignment.

    Key Insight: Lines in the same block have:
    1. Small vertical gaps between them
    2. Similar horizontal extent (alignment)
    3. No large whitespace separating them

    For curved documents (high y_variance_ratio), we relax the horizontal
    overlap requirement since a single visual line can span the entire page
    width with different y-positions at each end.

    Args:
        lines: List of Line instances
        page_width, page_height: Page dimensions
        config: Configuration with thresholds
        y_variance_ratio: Y-coordinate variance ratio (high = curved document)

    Returns:
        List of Block instances
    """
    if not lines:
        logger.warning("No lines to form blocks from")
        return []

    logger.info(f"Forming blocks from {len(lines)} lines (y_variance_ratio={y_variance_ratio:.2f})")

    # Calculate average line height for adaptive thresholds BEFORE sorting
    # (needed for Y-binning during sort)
    avg_line_height = get_average_line_height(lines)
    if avg_line_height == 0:
        avg_line_height = 15.0  # Fallback

    # For curved documents, use larger Y-bin size to group lines at similar heights
    # even if they have different y-positions due to page curvature
    if y_variance_ratio > 0.5:
        # Curved document: use more aggressive y-binning
        y_bin_size = avg_line_height * 1.0  # Full line height tolerance
        logger.info(f"Curved document detected: using larger y-bin size {y_bin_size:.1f}px")
    else:
        # Normal document: half line height
        y_bin_size = avg_line_height * 0.5

    # Sort lines top to bottom, then left to right
    # CRITICAL: Lines within the y_bin_size of each other should be treated
    # as on the same row (handles justified text baseline variance and split lines)
    sorted_lines = sorted(lines, key=lambda l: (round(l.y0 / y_bin_size), l.x0))

    logger.debug(f"Sorting lines with Y-bin size: {y_bin_size:.1f}px, avg line height: {avg_line_height:.1f}px")

    # Initialize with first line
    blocks = []
    current_block_lines = [sorted_lines[0]]
    block_id_counter = 0

    # Process remaining lines
    for i in range(1, len(sorted_lines)):
        current_line = sorted_lines[i]
        previous_line = current_block_lines[-1]

        # Calculate metrics for decision
        same_block = _is_same_block(
            previous_line,
            current_line,
            current_block_lines,
            avg_line_height,
            config,
            y_variance_ratio
        )

        if same_block:
            # Add to current block
            current_block_lines.append(current_line)
        else:
            # Finalize current block
            block = _create_block(current_block_lines, block_id_counter)
            blocks.append(block)
            block_id_counter += 1

            # Start new block
            current_block_lines = [current_line]

    # Don't forget last block
    if current_block_lines:
        block = _create_block(current_block_lines, block_id_counter)
        blocks.append(block)

    logger.info(f"Formed {len(blocks)} blocks from {len(lines)} lines")

    # Log block statistics
    lines_per_block = [len(block.lines) for block in blocks]
    avg_lines = sum(lines_per_block) / len(lines_per_block) if lines_per_block else 0
    logger.debug(f"Average lines per block: {avg_lines:.1f} "
                f"(min={min(lines_per_block)}, max={max(lines_per_block)})")

    return blocks


def _is_same_block(previous_line: Line,
                   current_line: Line,
                   current_block_lines: List[Line],
                   avg_line_height: float,
                   config: ReadingOrderConfig,
                   y_variance_ratio: float = 0.0) -> bool:
    """
    Determine if current line belongs to same block as previous line.

    Uses:
    1. Vertical gap between lines
    2. Horizontal overlap (alignment) - relaxed for curved documents

    Args:
        previous_line: Last line in current block
        current_line: Line being evaluated
        current_block_lines: All lines in current block so far
        avg_line_height: Average line height for threshold calculation
        config: Configuration
        y_variance_ratio: Y-coordinate variance ratio (high = curved document)

    Returns:
        True if same block, False if new block
    """
    # Criterion 1: Vertical gap
    vertical_gap = current_line.y0 - previous_line.y1

    # Criterion 2: Horizontal overlap
    horizontal_overlap = calculate_horizontal_overlap(previous_line, current_line)

    # Calculate dynamic threshold
    # For curved documents, use more lenient gap threshold
    if y_variance_ratio > 0.5:
        gap_threshold = avg_line_height * config.block_vertical_gap_multiplier * 1.5
    else:
        gap_threshold = avg_line_height * config.block_vertical_gap_multiplier

    # Decision logic
    has_small_gap = vertical_gap < gap_threshold

    # For curved documents, relax horizontal overlap requirement
    # since lines from the same visual row may be at different x-positions
    if y_variance_ratio > 0.5:
        # For curved documents: if lines are at similar y-level (small gap),
        # group them even without horizontal overlap
        has_alignment = horizontal_overlap > 0 or vertical_gap < avg_line_height * 0.3
    else:
        has_alignment = horizontal_overlap > config.block_horizontal_overlap_threshold

    same_block = has_small_gap and has_alignment

    # Debug logging for edge cases
    if config.debug and vertical_gap > gap_threshold * 0.8:
        logger.debug(f"\nEvaluating line starting with '{current_line.words[0].text if current_line.words else '?'}' "
                    f"after '{previous_line.words[0].text if previous_line.words else '?'}':")
        logger.debug(f"  Vertical gap: {vertical_gap:.1f}px (threshold: {gap_threshold:.1f}px)")
        logger.debug(f"  Horizontal overlap: {horizontal_overlap:.2f} (threshold: {config.block_horizontal_overlap_threshold})")
        logger.debug(f"  y_variance_ratio: {y_variance_ratio:.2f}")
        logger.debug(f"  => Decision: {'SAME BLOCK' if same_block else 'NEW BLOCK'}")

    return same_block


def _create_block(lines: List[Line], block_id: int) -> Block:
    """
    Create a Block object from lines.

    Args:
        lines: List of Line instances
        block_id: Unique identifier for this block

    Returns:
        Block instance with sorted lines and calculated bounding box
    """
    # Sort lines top to bottom
    sorted_lines = sorted(lines, key=lambda l: l.y0)

    # Create Block object (bounding box calculated in __post_init__)
    block = Block(lines=sorted_lines, block_id=block_id)

    # Assign block_id to constituent lines
    for line in sorted_lines:
        line.block_id = block_id

    return block
