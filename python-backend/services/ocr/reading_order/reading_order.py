"""
Layer 4: Reading Order Assignment

Assigns sequential reading order to regions, blocks, and lines
based on layout type and document structure.
"""

import logging
from typing import List
from .data_structures import Region, Block
from .config import ReadingOrderConfig

logger = logging.getLogger(__name__)


def assign_reading_order(regions: List[Region],
                        layout_type: str,
                        page_width: float,
                        config: ReadingOrderConfig) -> List[Region]:
    """
    Assign reading order to regions based on layout type.

    Routes to appropriate strategy:
    - SINGLE_COLUMN: top to bottom
    - MULTI_COLUMN: columns left to right, then top to bottom within each
    - TABLE: row by row, left to right within rows
    - COMPLEX: spatial order (top-left to bottom-right)

    Args:
        regions: List of Region instances
        layout_type: Layout type string
        page_width: Page width for context
        config: Configuration

    Returns:
        List of regions with reading_order assigned
    """
    if not regions:
        logger.warning("No regions to assign reading order")
        return []

    logger.info(f"Assigning reading order for {layout_type} layout with {len(regions)} regions")

    # Route to appropriate strategy
    if layout_type == "SINGLE_COLUMN":
        ordered_regions = assign_single_column_order(regions)

    elif layout_type == "MULTI_COLUMN":
        ordered_regions = assign_multi_column_order(regions, page_width, config)

    elif layout_type == "TABLE":
        ordered_regions = assign_table_order(regions)

    else:  # COMPLEX or unknown
        ordered_regions = assign_spatial_order(regions)

    logger.info(f"Reading order assigned to {len(ordered_regions)} regions")

    return ordered_regions


# ============================================================================
# Strategy 1: Single Column
# ============================================================================

def assign_single_column_order(regions: List[Region]) -> List[Region]:
    """
    Assign reading order for single-column layout.

    Simple: top to bottom.

    Args:
        regions: List of Region instances

    Returns:
        Ordered list of regions
    """
    logger.debug("Using single-column ordering strategy")

    # Sort regions top to bottom
    sorted_regions = sorted(regions, key=lambda r: min(b.y0 for b in r.blocks))

    # Assign reading order
    for i, region in enumerate(sorted_regions):
        region.reading_order = i

        # Assign order to blocks within region
        for j, block in enumerate(region.blocks):
            block.reading_order = i * 1000 + j

    return sorted_regions


# ============================================================================
# Strategy 2: Multi-Column
# ============================================================================

def assign_multi_column_order(regions: List[Region],
                              page_width: float,
                              config: ReadingOrderConfig) -> List[Region]:
    """
    Assign reading order for multi-column layout.

    Handles 2-column and 3+ column layouts differently.

    Args:
        regions: List of Region instances
        page_width: Page width
        config: Configuration

    Returns:
        Ordered list of regions
    """
    logger.debug(f"Using multi-column ordering strategy for {len(regions)} columns")

    # Sort regions left to right
    sorted_regions = sorted(regions, key=lambda r: r.center_x)

    num_columns = len(sorted_regions)

    if num_columns == 2:
        return _handle_two_column_layout(sorted_regions, config)
    elif num_columns >= 3:
        return _handle_multi_column_layout(sorted_regions)
    else:
        # Single region - treat as single column
        return assign_single_column_order(sorted_regions)


def _handle_two_column_layout(regions: List[Region],
                               config: ReadingOrderConfig) -> List[Region]:
    """
    Handle two-column layout.

    Determines if columns are:
    - True side-by-side (significant vertical overlap)
    - Stacked regions (little overlap)

    Args:
        regions: List of 2 Region instances
        config: Configuration

    Returns:
        Ordered list of regions
    """
    left_column = regions[0]
    right_column = regions[1]

    # Calculate vertical extent of each column
    left_top = min(b.y0 for b in left_column.blocks)
    left_bottom = max(b.y1 for b in left_column.blocks)
    right_top = min(b.y0 for b in right_column.blocks)
    right_bottom = max(b.y1 for b in right_column.blocks)

    # Calculate vertical overlap
    overlap_top = max(left_top, right_top)
    overlap_bottom = min(left_bottom, right_bottom)
    overlap_height = max(0.0, overlap_bottom - overlap_top)

    total_height = max(left_bottom, right_bottom) - min(left_top, right_top)

    overlap_ratio = overlap_height / total_height if total_height > 0 else 0

    logger.debug(f"Two-column layout: vertical overlap ratio = {overlap_ratio:.2f}")

    if overlap_ratio > config.two_column_overlap_threshold:
        # True side-by-side columns
        logger.debug("True side-by-side columns detected")

        # Reading order: left column fully, then right column
        left_column.reading_order = 0
        for i, block in enumerate(left_column.blocks):
            block.reading_order = i

        right_column.reading_order = 1
        for i, block in enumerate(right_column.blocks):
            block.reading_order = 1000 + i

        return [left_column, right_column]

    else:
        # Stacked regions (not true columns)
        logger.debug("Stacked regions detected (not true side-by-side columns)")

        # Reading order: top to bottom regardless of horizontal position
        all_blocks = left_column.blocks + right_column.blocks
        sorted_blocks = sorted(all_blocks, key=lambda b: (b.y0, b.x0))

        # Assign reading order
        for i, block in enumerate(sorted_blocks):
            block.reading_order = i

        # Merge into single region
        merged_region = Region(
            blocks=sorted_blocks,
            x0=min(left_column.x0, right_column.x0),
            x1=max(left_column.x1, right_column.x1),
            region_id=0,
            region_type="merged"
        )
        merged_region.reading_order = 0

        return [merged_region]


def _handle_multi_column_layout(regions: List[Region]) -> List[Region]:
    """
    Handle 3+ column layout with spanning header detection.

    First detects blocks that span across columns at similar Y positions,
    then reads remaining blocks column by column.

    Reading order:
    1. Spanning headers (blocks at same Y across columns)
    2. Column 0 content top-to-bottom
    3. Column 1 content top-to-bottom
    4. etc.

    Args:
        regions: List of Region instances (already sorted left to right)

    Returns:
        Ordered list of regions
    """
    logger.debug(f"Multi-column layout: {len(regions)} columns")

    # Collect all blocks with their region info
    all_blocks = []
    for region_idx, region in enumerate(regions):
        for block in region.blocks:
            all_blocks.append({
                'block': block,
                'region_idx': region_idx,
                'y': block.y0
            })

    # Detect spanning headers: ONLY in the top portion of the page
    # Find the minimum Y position of any block - this is the top of content
    if not all_blocks:
        return regions

    min_y = min(item['y'] for item in all_blocks)
    max_y = max(item['y'] for item in all_blocks)
    content_height = max_y - min_y

    # Only detect spanning rows in the top 15% of content area
    spanning_zone_threshold = min_y + content_height * 0.15
    y_tolerance = 50  # Blocks within 50px vertical are considered same row

    spanning_blocks = []
    column_blocks_by_region = {i: [] for i in range(len(regions))}

    # Group blocks by Y position
    y_groups = {}
    for item in all_blocks:
        y_rounded = round(item['y'] / y_tolerance) * y_tolerance
        if y_rounded not in y_groups:
            y_groups[y_rounded] = []
        y_groups[y_rounded].append(item)

    # Check each Y group for spanning rows (only in top zone)
    for y_pos, items in y_groups.items():
        avg_y = sum(item['y'] for item in items) / len(items)

        # Only consider spanning in the header zone
        if avg_y > spanning_zone_threshold:
            # Below header zone - all go to column content
            for item in items:
                column_blocks_by_region[item['region_idx']].append(item['block'])
            continue

        regions_in_group = set(item['region_idx'] for item in items)

        # If blocks at this Y exist in multiple regions AND in header zone, it's spanning
        if len(regions_in_group) > 1:
            # Sort by X position (left to right)
            items.sort(key=lambda x: x['block'].x0)
            for item in items:
                spanning_blocks.append(item)
            logger.debug(f"Spanning header at y~{y_pos}: {len(items)} blocks across {len(regions_in_group)} columns")
        else:
            # Single-region blocks go to column content even in header zone
            for item in items:
                column_blocks_by_region[item['region_idx']].append(item['block'])

    # Sort spanning blocks by Y, then X
    spanning_blocks.sort(key=lambda x: (x['y'], x['block'].x0))

    # Assign reading order: spanning headers first, then columns
    reading_order = 0

    # Process spanning headers first
    for item in spanning_blocks:
        block = item['block']
        block.reading_order = reading_order
        block.block_type = "spanning_header"
        reading_order += 1

    # If we have spanning blocks, create a spanning region and prepend it
    output_regions = []
    if spanning_blocks:
        # Create a spanning header region
        spanning_block_objects = [item['block'] for item in spanning_blocks]
        spanning_region = Region(
            blocks=spanning_block_objects,
            x0=min(b.x0 for b in spanning_block_objects),
            x1=max(b.x1 for b in spanning_block_objects),
            region_id=-1,  # Special ID for spanning
            region_type="spanning_header"
        )
        spanning_region.reading_order = -1  # Goes first
        output_regions.append(spanning_region)
        logger.debug(f"Created spanning region with {len(spanning_block_objects)} blocks")

    # Process column content
    for region_idx, region in enumerate(regions):
        region.reading_order = region_idx
        # Get remaining blocks for this region (non-spanning)
        remaining_blocks = column_blocks_by_region[region_idx]
        # Sort by Y position
        remaining_blocks.sort(key=lambda b: b.y0)
        region.blocks = remaining_blocks

        logger.debug(f"Column {region_idx}: {len(remaining_blocks)} blocks after removing spanning")

        for block in remaining_blocks:
            block.reading_order = reading_order
            reading_order += 1

        if remaining_blocks:  # Only add non-empty regions
            output_regions.append(region)

    return output_regions


# ============================================================================
# Strategy 3: Table
# ============================================================================

def assign_table_order(regions: List[Region]) -> List[Region]:
    """
    Assign reading order for table layout.

    Read row by row (top to bottom, left to right within each row).

    Args:
        regions: List of Region instances

    Returns:
        Ordered list of regions
    """
    logger.debug("Using table ordering strategy")

    # Collect all lines from all regions
    all_lines = []
    for region in regions:
        for block in region.blocks:
            all_lines.extend(block.lines)

    if not all_lines:
        return regions

    # Group lines by vertical position (rows)
    from .utils import group_by_vertical_position
    rows = group_by_vertical_position(all_lines, tolerance=10)

    reading_order = 0

    for row in rows:
        # Sort cells in row left to right
        sorted_cells = sorted(row, key=lambda line: line.x0)

        for line in sorted_cells:
            line.reading_order = reading_order
            reading_order += 1

    # Assign order to regions
    for i, region in enumerate(regions):
        region.reading_order = i

    return regions


# ============================================================================
# Strategy 4: Spatial Order (Fallback)
# ============================================================================

def assign_spatial_order(regions: List[Region]) -> List[Region]:
    """
    Assign spatial reading order (fallback for complex layouts).

    Top-left to bottom-right:
    - Group into horizontal bands
    - Within each band, left to right

    Args:
        regions: List of Region instances

    Returns:
        Ordered list of regions
    """
    logger.debug("Using spatial ordering strategy (fallback)")

    # Calculate center Y for each region
    def get_center_y(region):
        block_centers = [block.center_y for block in region.blocks]
        return sum(block_centers) / len(block_centers) if block_centers else 0

    # Sort by: vertical band first (groups of ~100px), then horizontal position
    sorted_regions = sorted(
        regions,
        key=lambda r: (int(get_center_y(r) / 100), r.center_x)
    )

    # Assign reading order
    for i, region in enumerate(sorted_regions):
        region.reading_order = i

        for j, block in enumerate(region.blocks):
            block.reading_order = i * 1000 + j

    return sorted_regions
