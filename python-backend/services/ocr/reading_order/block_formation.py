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
               config: ReadingOrderConfig) -> List[Block]:
    """
    Group lines into blocks using vertical proximity and horizontal alignment.

    Key Insight: Lines in the same block have:
    1. Small vertical gaps between them
    2. Similar horizontal extent (alignment)
    3. No large whitespace separating them

    Args:
        lines: List of Line instances
        page_width, page_height: Page dimensions
        config: Configuration with thresholds

    Returns:
        List of Block instances
    """
    if not lines:
        logger.warning("No lines to form blocks from")
        return []

    logger.info(f"Forming blocks from {len(lines)} lines")

    # Sort lines top to bottom
    sorted_lines = sorted(lines, key=lambda l: l.y0)

    # Calculate average line height for adaptive thresholds
    avg_line_height = get_average_line_height(sorted_lines)
    if avg_line_height == 0:
        logger.error("Average line height is 0, cannot form blocks")
        return []

    logger.debug(f"Average line height: {avg_line_height:.1f}px")

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
            config
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
                   config: ReadingOrderConfig) -> bool:
    """
    Determine if current line belongs to same block as previous line.

    Uses:
    1. Vertical gap between lines
    2. Horizontal overlap (alignment)

    Args:
        previous_line: Last line in current block
        current_line: Line being evaluated
        current_block_lines: All lines in current block so far
        avg_line_height: Average line height for threshold calculation
        config: Configuration

    Returns:
        True if same block, False if new block
    """
    # Criterion 1: Vertical gap
    vertical_gap = current_line.y0 - previous_line.y1

    # Criterion 2: Horizontal overlap
    horizontal_overlap = calculate_horizontal_overlap(previous_line, current_line)

    # Calculate dynamic threshold
    gap_threshold = avg_line_height * config.block_vertical_gap_multiplier

    # Decision logic
    has_small_gap = vertical_gap < gap_threshold
    has_alignment = horizontal_overlap > config.block_horizontal_overlap_threshold

    same_block = has_small_gap and has_alignment

    # Debug logging for edge cases
    if config.debug and vertical_gap > gap_threshold * 0.8:
        logger.debug(f"\nEvaluating line starting with '{current_line.words[0].text if current_line.words else '?'}' "
                    f"after '{previous_line.words[0].text if previous_line.words else '?'}':")
        logger.debug(f"  Vertical gap: {vertical_gap:.1f}px (threshold: {gap_threshold:.1f}px)")
        logger.debug(f"  Horizontal overlap: {horizontal_overlap:.2f} (threshold: {config.block_horizontal_overlap_threshold})")
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
