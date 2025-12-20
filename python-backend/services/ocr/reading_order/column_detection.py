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

    For newspaper-style layouts with headers/photos, uses vertical band analysis
    to detect columns in the lower (article) portion of the page.

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

    # Step 1: Try full-page analysis first
    histogram = _build_horizontal_histogram(word_boxes, int(page_width))
    smoothed_histogram = gaussian_smooth(histogram, config.histogram_smooth_sigma)

    valleys = _find_valleys_in_histogram(
        smoothed_histogram,
        config.min_valley_width,
        config.valley_density_threshold,
        page_width
    )

    min_inter_valley_distance = page_width * config.min_inter_valley_ratio
    valleys = _merge_close_valleys(valleys, min_inter_valley_distance)

    # Step 2: If insufficient valleys, try vertical band analysis
    # This helps with newspaper layouts where headers/photos span columns
    if len(valleys) < 2:
        logger.info(f"Full-page analysis found {len(valleys)} valleys, trying vertical band analysis")
        band_valleys = _detect_columns_by_vertical_bands(
            word_boxes, page_width, page_height, config
        )
        if len(band_valleys) >= 2:
            logger.info(f"Vertical band analysis found {len(band_valleys)} valleys at: {[f'{v:.0f}' for v in band_valleys]}")
            valleys = band_valleys
        else:
            logger.info(f"Vertical band analysis also failed, only found {len(band_valleys)} valleys")

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


def _find_local_minima_valleys(histogram: np.ndarray,
                               min_valley_width: float,
                               page_width: float,
                               relative_depth_threshold: float = 0.7,
                               max_valleys: int = 4) -> List[float]:
    """
    Find valleys using local minima detection.

    Unlike absolute threshold method, this finds positions that are
    significantly lower than their surrounding area - works better for
    newspaper layouts where columns have text but at lower density.

    Args:
        histogram: Smoothed horizontal density histogram
        min_valley_width: Minimum width for grouping adjacent minima
        page_width: Page width for bounds checking
        relative_depth_threshold: Valley must be this fraction of local max (0.7 = 30% dip)
        max_valleys: Maximum number of valleys to return (keeps deepest)

    Returns:
        List of X positions (valley centers) representing column separators
    """
    if len(histogram) < 50:
        return []

    # Use a window to find local minima
    window_size = max(30, int(page_width * 0.1))  # 10% of page width or 30px
    half_window = window_size // 2

    minima = []

    # Scan for local minima
    for x in range(half_window, len(histogram) - half_window):
        local_region = histogram[x - half_window:x + half_window]
        local_max = np.max(local_region)
        local_min = np.min(local_region)

        # Check if this position is at or near the local minimum
        if histogram[x] <= local_min * 1.05:  # Within 5% of local min
            # Check if this is a significant dip (valley is X% lower than local peaks)
            if local_max > 0 and histogram[x] / local_max < relative_depth_threshold:
                # Store position, density, and depth ratio (lower = deeper valley)
                depth_ratio = histogram[x] / local_max if local_max > 0 else 1.0
                minima.append((x, histogram[x], depth_ratio))

    if not minima:
        logger.debug(f"No local minima found with depth threshold {relative_depth_threshold}")
        return []

    # Group adjacent minima and find the center of each group
    valleys_with_depth = []
    group_positions = [minima[0][0]]
    group_depths = [minima[0][2]]

    for i in range(1, len(minima)):
        if minima[i][0] - minima[i-1][0] <= 3:  # Adjacent positions
            group_positions.append(minima[i][0])
            group_depths.append(minima[i][2])
        else:
            # End of group - calculate center and average depth
            if len(group_positions) >= 1:
                valley_center = sum(group_positions) / len(group_positions)
                avg_depth = sum(group_depths) / len(group_depths)
                valleys_with_depth.append((valley_center, avg_depth))
                logger.debug(f"Local minimum valley at X={valley_center:.0f}, depth_ratio={avg_depth:.2f}")
            group_positions = [minima[i][0]]
            group_depths = [minima[i][2]]

    # Don't forget last group
    if group_positions:
        valley_center = sum(group_positions) / len(group_positions)
        avg_depth = sum(group_depths) / len(group_depths)
        valleys_with_depth.append((valley_center, avg_depth))
        logger.debug(f"Local minimum valley at X={valley_center:.0f}, depth_ratio={avg_depth:.2f}")

    # Sort by depth (lowest depth_ratio = deepest valley) and keep top N
    valleys_with_depth.sort(key=lambda v: v[1])

    if len(valleys_with_depth) > max_valleys:
        logger.debug(f"Limiting from {len(valleys_with_depth)} to {max_valleys} deepest valleys")
        valleys_with_depth = valleys_with_depth[:max_valleys]

    # Extract positions and sort by X position for output
    valleys = sorted([v[0] for v in valleys_with_depth])

    logger.debug(f"Found {len(valleys)} local minima valleys")
    return valleys


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


def _select_best_valleys(valleys: List[float], n: int, page_width: float) -> List[float]:
    """
    Select the N best valleys that create the most evenly-spaced columns.

    For example, if we have valleys at [128, 250, 374, 500] and need 2,
    we should select [250, 374] which divides the page into 3 roughly equal columns.

    Args:
        valleys: List of valley positions (sorted by X)
        n: Number of valleys to select
        page_width: Page width for calculating ideal spacing

    Returns:
        List of N best valley positions
    """
    if len(valleys) <= n:
        return valleys

    # For N valleys, we want N+1 columns of equal width
    # Ideal spacing: page_width / (n + 1)
    ideal_spacing = page_width / (n + 1)

    # Try all combinations of N valleys and find the one with best spacing
    from itertools import combinations

    best_combo = None
    best_score = float('inf')

    for combo in combinations(valleys, n):
        sorted_combo = sorted(combo)

        # Calculate how well this combo divides the page
        # Add implicit boundaries at 0 and page_width
        boundaries = [0.0] + list(sorted_combo) + [page_width]

        # Calculate widths of resulting columns
        widths = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries) - 1)]

        # Score = sum of squared deviations from ideal width
        score = sum((w - ideal_spacing) ** 2 for w in widths)

        if score < best_score:
            best_score = score
            best_combo = sorted_combo

    logger.debug(f"Selected best {n} valleys from {len(valleys)}: {[f'{v:.0f}' for v in best_combo]}")
    return list(best_combo) if best_combo else valleys[:n]


def _merge_close_valleys(valleys: List[float], min_distance: float) -> List[float]:
    """
    Merge valleys that are too close together.

    When multiple valleys are detected within min_distance of each other,
    they are merged into a single valley at the average position.

    Args:
        valleys: List of valley positions (sorted)
        min_distance: Minimum distance between valleys

    Returns:
        List of merged valley positions
    """
    if len(valleys) <= 1:
        return valleys

    # Ensure sorted
    sorted_valleys = sorted(valleys)

    merged = []
    group = [sorted_valleys[0]]

    for i in range(1, len(sorted_valleys)):
        current = sorted_valleys[i]
        prev = sorted_valleys[i - 1]

        if current - prev < min_distance:
            # Too close - add to current group
            group.append(current)
        else:
            # Far enough - finalize previous group and start new one
            merged.append(sum(group) / len(group))  # Average position
            group = [current]

    # Don't forget the last group
    merged.append(sum(group) / len(group))

    logger.debug(f"Merged {len(valleys)} valleys to {len(merged)} (min_distance={min_distance:.1f})")

    return merged


def _detect_columns_by_vertical_bands(word_boxes: List[WordBox],
                                      page_width: float,
                                      page_height: float,
                                      config: ReadingOrderConfig) -> List[float]:
    """
    Detect columns by analyzing vertical bands of the page.

    Newspaper-style layouts often have:
    - Spanning headers/mastheads at top
    - Large photos in upper portion
    - Column structure only in lower article area

    This function analyzes the lower portion of the page where
    columns are most likely to be visible.

    Args:
        word_boxes: List of WordBox instances
        page_width, page_height: Page dimensions
        config: Configuration

    Returns:
        List of valley positions (column separators)
    """
    if not word_boxes:
        return []

    # Analyze bottom 40% of the page (where articles typically are)
    y_threshold = page_height * 0.6
    lower_words = [w for w in word_boxes if w.center_y >= y_threshold]

    logger.info(f"Vertical band analysis: page_height={page_height:.0f}, y_threshold={y_threshold:.0f}")
    logger.info(f"Vertical band analysis: {len(lower_words)} of {len(word_boxes)} words in lower 40%")

    if len(lower_words) < 20:
        # Not enough words in lower portion
        logger.info(f"Only {len(lower_words)} words in lower 40%, insufficient for band analysis")
        return []

    logger.debug(f"[BAND] Analyzing lower 40% of page: {len(lower_words)} words")

    # Build histogram for lower portion only
    histogram = _build_horizontal_histogram(lower_words, int(page_width))
    smoothed = gaussian_smooth(histogram, config.histogram_smooth_sigma)

    mean_density = np.mean(smoothed)
    threshold = mean_density * config.valley_density_threshold
    logger.debug(f"[BAND] Histogram: mean_density={mean_density:.1f}, threshold={threshold:.1f}, min_valley_width={config.min_valley_width * 0.75:.0f}")

    # First try absolute threshold method
    valleys = _find_valleys_in_histogram(
        smoothed,
        config.min_valley_width * 0.75,  # Allow slightly narrower valleys
        config.valley_density_threshold,
        page_width
    )

    logger.debug(f"[BAND] Absolute threshold found {len(valleys)} raw valleys: {[f'{v:.0f}' for v in valleys]}")

    # If absolute threshold fails, try local minima detection
    if len(valleys) < 2:
        logger.debug(f"[BAND] Absolute threshold failed, trying local minima detection...")

        # First get all valleys without limiting
        all_valleys = _find_local_minima_valleys(
            smoothed,
            config.min_valley_width * 0.75,
            page_width,
            relative_depth_threshold=0.75,  # Valley must be 25% lower than local peak
            max_valleys=10  # Get many, then filter
        )
        logger.debug(f"[BAND] Local minima found {len(all_valleys)} raw valleys: {[f'{v:.0f}' for v in all_valleys]}")

        # Filter margins FIRST before limiting
        # Use 25% margin to exclude false positives near page edges
        margin = page_width * 0.25
        interior_valleys = [v for v in all_valleys if margin < v < (page_width - margin)]
        logger.debug(f"[BAND] After margin filter: {len(interior_valleys)} interior valleys (margin={margin:.0f})")

        # Now limit to expected column count
        estimated_max_columns = max(2, int(page_width / 150))
        max_valleys = estimated_max_columns - 1  # 3 columns = 2 valleys

        if len(interior_valleys) > max_valleys:
            # Keep the most evenly spaced valleys
            interior_valleys = _select_best_valleys(interior_valleys, max_valleys, page_width)

        valleys = interior_valleys
        logger.debug(f"[BAND] Final valleys (max={max_valleys}): {[f'{v:.0f}' for v in valleys]}")

    else:
        # Filter out margin valleys (too close to edges)
        margin = page_width * 0.15
        interior_valleys = [v for v in valleys if margin < v < (page_width - margin)]
        valleys = interior_valleys

    logger.info(f"[BAND] Detected {len(valleys)} column boundaries")

    if len(valleys) < 1:
        logger.debug(f"Band analysis found {len(valleys)} interior valleys, need at least 1")
        return []

    # Merge close valleys
    min_inter_valley_distance = page_width * config.min_inter_valley_ratio
    merged_valleys = _merge_close_valleys(valleys, min_inter_valley_distance)

    logger.debug(f"Vertical band analysis: {len(merged_valleys)} valleys after filtering/merging")

    return merged_valleys
