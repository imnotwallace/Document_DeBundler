"""
Layer 3.2: Column Detection

Detects multi-column layouts using horizontal projection histogram method.

This is the GOLD STANDARD for column detection, significantly better
than simple clustering approaches.
"""

import logging
import numpy as np
from typing import List
from .data_structures import WordBox, Block, Region
from .config import ReadingOrderConfig
from .utils import gaussian_smooth

logger = logging.getLogger(__name__)


def detect_columns_by_projection(word_boxes: List[WordBox],
                                 blocks: List[Block],
                                 page_width: float,
                                 page_height: float,
                                 config: ReadingOrderConfig) -> List[Region]:
    """
    Detect columns using horizontal projection histogram method.

    This is the industry-standard approach:
    1. Build horizontal density histogram
    2. Smooth with Gaussian filter
    3. Find valleys (low-density regions = column gutters)
    4. Use valleys to define column boundaries

    Args:
        word_boxes: List of WordBox instances (for histogram)
        blocks: List of Block instances (to assign to columns)
        page_width, page_height: Page dimensions
        config: Configuration

    Returns:
        List of Region instances (one per column)
    """
    if not word_boxes or not blocks:
        logger.warning("No words or blocks for column detection")
        if blocks:
            # Single region with all blocks
            return [_create_single_region(blocks, page_width)]
        return []

    logger.info(f"Detecting columns using projection histogram for {len(blocks)} blocks")

    # Step 1: Build horizontal density histogram
    histogram = _build_horizontal_histogram(word_boxes, int(page_width))

    # Step 2: Smooth histogram to reduce noise
    smoothed_histogram = gaussian_smooth(histogram, config.histogram_smooth_sigma)

    # Step 3: Find valleys (column separators)
    valleys = _find_valleys_in_histogram(
        smoothed_histogram,
        config.min_valley_width,
        config.valley_density_threshold,
        page_width
    )

    # Step 4: Define column boundaries and assign blocks
    if len(valleys) == 0:
        # Single column
        logger.info("No valleys found - single column layout")
        return [_create_single_region(blocks, page_width)]

    # Multiple columns
    logger.info(f"Found {len(valleys)} column separators at X positions: {[f'{v:.0f}' for v in valleys]}")

    boundaries = [0.0] + valleys + [page_width]
    regions = []

    for i in range(len(boundaries) - 1):
        left_boundary = boundaries[i]
        right_boundary = boundaries[i + 1]

        # Assign blocks to this column
        # Block belongs to column if its center_x is within boundaries
        column_blocks = [
            block for block in blocks
            if left_boundary <= block.center_x < right_boundary
        ]

        if column_blocks:
            region = _create_region(column_blocks, left_boundary, right_boundary, i)
            regions.append(region)
            logger.debug(f"Column {i+1}: X=[{left_boundary:.0f}, {right_boundary:.0f}], "
                        f"{len(column_blocks)} blocks")

    logger.info(f"Created {len(regions)} column regions")

    return regions


def _build_horizontal_histogram(word_boxes: List[WordBox], page_width: int) -> np.ndarray:
    """
    Build horizontal density histogram.

    Each X position gets a count of how many words cover it.

    Args:
        word_boxes: List of WordBox instances
        page_width: Page width (determines histogram size)

    Returns:
        Numpy array of length page_width with density counts
    """
    histogram = np.zeros(page_width, dtype=np.int32)

    for word in word_boxes:
        # Mark all X positions this word occupies
        x_start = max(0, int(np.floor(word.x0)))
        x_end = min(page_width, int(np.ceil(word.x1)))

        # Increment histogram for each position
        histogram[x_start:x_end] += 1

    logger.debug(f"Built histogram: max_density={np.max(histogram)}, "
                f"mean_density={np.mean(histogram):.1f}")

    return histogram


def _find_valleys_in_histogram(histogram: np.ndarray,
                               min_valley_width: float,
                               valley_threshold_ratio: float,
                               page_width: float) -> List[float]:
    """
    Find valleys (low-density regions) in histogram.

    Valleys represent column gutters (white space between columns).

    Args:
        histogram: Smoothed horizontal density histogram
        min_valley_width: Minimum width (pixels) for a valley to count
        valley_threshold_ratio: Density threshold as fraction of mean
        page_width: Page width for bounds checking

    Returns:
        List of X positions (valley centers) representing column separators
    """
    # Calculate threshold for "low density"
    mean_density = np.mean(histogram)
    threshold = mean_density * valley_threshold_ratio

    logger.debug(f"Valley detection: mean_density={mean_density:.1f}, "
                f"threshold={threshold:.1f} (ratio={valley_threshold_ratio})")

    valleys = []
    in_valley = False
    valley_start = 0

    for x in range(len(histogram)):
        if histogram[x] < threshold:
            # Low density - might be a valley
            if not in_valley:
                in_valley = True
                valley_start = x
        else:
            # High density - end of valley
            if in_valley:
                valley_width = x - valley_start

                # Only count wide valleys (gutters)
                if valley_width >= min_valley_width:
                    valley_center = (valley_start + x) / 2.0
                    valleys.append(valley_center)
                    logger.debug(f"Valley found: X={valley_center:.0f}, width={valley_width:.0f}px")

                in_valley = False

    # Handle valley extending to end of page
    if in_valley:
        valley_width = len(histogram) - valley_start
        if valley_width >= min_valley_width:
            valley_center = (valley_start + len(histogram)) / 2.0
            valleys.append(valley_center)

    return valleys


def _create_region(blocks: List[Block],
                  left_boundary: float,
                  right_boundary: float,
                  region_id: int) -> Region:
    """
    Create a Region (column) from blocks.

    Args:
        blocks: List of Block instances in this column
        left_boundary, right_boundary: Horizontal boundaries
        region_id: Unique identifier

    Returns:
        Region instance
    """
    # Sort blocks top to bottom
    sorted_blocks = sorted(blocks, key=lambda b: b.y0)

    region = Region(
        blocks=sorted_blocks,
        x0=left_boundary,
        x1=right_boundary,
        region_id=region_id,
        region_type="column"
    )

    # Assign region_id to blocks
    for block in sorted_blocks:
        block.region_id = region_id

    return region


def _create_single_region(blocks: List[Block], page_width: float) -> Region:
    """
    Create a single region containing all blocks.

    Args:
        blocks: List of Block instances
        page_width: Page width

    Returns:
        Region instance spanning full page width
    """
    return _create_region(blocks, 0.0, page_width, 0)
