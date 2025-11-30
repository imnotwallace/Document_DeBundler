"""
Region Subdivider for Complex Layouts.

Applies projection histogram analysis WITHIN large semantic regions
to detect internal column structures that PP-DocLayout misses.

PP-DocLayout excels at semantic classification (text, image, table, etc.)
but doesn't identify column structure within large text regions.
This class bridges that gap by applying the proven projection histogram
algorithm from column_detection.py to subdivide large regions.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..reading_order.data_structures import WordBox
from ..reading_order.utils import gaussian_smooth
from ..multipass.data_structures import (
    FlorenceRegion,
    BoundingBox,
    RegionType,
)

logger = logging.getLogger(__name__)


# Semantic labels that should be considered for subdivision
TEXT_REGION_LABELS = {
    "text",
    "content",
    "sidebar_text",
}


@dataclass
class SubdivisionConfig:
    """Configuration for region subdivision."""

    # Minimum region dimensions (as ratio of page) to consider subdivision
    min_region_width_ratio: float = 0.4  # 40% of page width
    min_region_height_ratio: float = 0.10  # 10% of page height

    # Minimum words required to attempt subdivision
    min_words_for_subdivision: int = 8

    # Valley detection parameters
    valley_threshold_ratio: float = 0.3  # Threshold as ratio of mean density
    min_valley_width_px: int = 15  # Minimum valley width in pixels
    histogram_smooth_sigma: float = 5.0  # Gaussian smoothing sigma

    # Column constraints
    min_column_width_px: int = 50  # Minimum column width in pixels


class RegionSubdivider:
    """
    Subdivides large semantic regions into columns using projection histograms.

    PP-DocLayout detects "text" regions but doesn't identify column structure
    within them. This class applies the projection histogram algorithm from
    column_detection.py to find internal column boundaries.

    Usage:
        subdivider = RegionSubdivider()
        subdivided_regions = subdivider.subdivide_regions(
            regions=layout_result.regions,
            word_boxes=ocr_word_boxes,
            page_width=1200,
            page_height=1600,
        )
    """

    def __init__(self, config: Optional[SubdivisionConfig] = None):
        """
        Initialize region subdivider.

        Args:
            config: Subdivision configuration. Uses defaults if None.
        """
        self.config = config or SubdivisionConfig()
        logger.info(
            f"RegionSubdivider initialized: "
            f"min_width_ratio={self.config.min_region_width_ratio}, "
            f"valley_threshold={self.config.valley_threshold_ratio}"
        )

    def subdivide_regions(
        self,
        regions: List[FlorenceRegion],
        word_boxes: List[WordBox],
        page_width: int,
        page_height: int,
    ) -> List[FlorenceRegion]:
        """
        Subdivide large text regions that likely contain multiple columns.

        Args:
            regions: List of FlorenceRegion from PP-DocLayout
            word_boxes: List of WordBox from OCR
            page_width: Page width in pixels
            page_height: Page height in pixels

        Returns:
            New list of regions with large regions split as needed.
            Original regions are preserved if no subdivision is needed.
        """
        if not regions or not word_boxes:
            return regions

        result = []
        subdivisions_made = 0

        for region in regions:
            if self._should_subdivide(region, page_width, page_height, word_boxes):
                sub_regions = self._subdivide_region(region, word_boxes)
                if len(sub_regions) > 1:
                    logger.info(
                        f"Subdivided region {region.region_id} ({region.semantic_label}) "
                        f"into {len(sub_regions)} columns"
                    )
                    subdivisions_made += 1
                result.extend(sub_regions)
            else:
                result.append(region)

        # Re-assign reading order hints based on position
        if subdivisions_made > 0:
            self._reassign_reading_order(result)
            logger.info(
                f"Region subdivision complete: {subdivisions_made} regions subdivided, "
                f"{len(result)} total regions"
            )

        return result

    def _should_subdivide(
        self,
        region: FlorenceRegion,
        page_width: int,
        page_height: int,
        word_boxes: List[WordBox],
    ) -> bool:
        """
        Determine if a region should be subdivided.

        Args:
            region: Region to evaluate
            page_width: Page width in pixels
            page_height: Page height in pixels
            word_boxes: All word boxes for counting

        Returns:
            True if region should be subdivided
        """
        # Only subdivide text-type regions
        if region.semantic_label not in TEXT_REGION_LABELS:
            return False

        # Check size thresholds
        width_ratio = region.bbox.width / page_width
        height_ratio = region.bbox.height / page_height

        if width_ratio < self.config.min_region_width_ratio:
            logger.debug(
                f"Region {region.region_id} too narrow for subdivision: "
                f"{width_ratio:.2f} < {self.config.min_region_width_ratio}"
            )
            return False

        if height_ratio < self.config.min_region_height_ratio:
            logger.debug(
                f"Region {region.region_id} too short for subdivision: "
                f"{height_ratio:.2f} < {self.config.min_region_height_ratio}"
            )
            return False

        # Count words in this region
        words_in_region = self._get_words_in_region(region, word_boxes)
        if len(words_in_region) < self.config.min_words_for_subdivision:
            logger.debug(
                f"Region {region.region_id} has too few words: "
                f"{len(words_in_region)} < {self.config.min_words_for_subdivision}"
            )
            return False

        logger.debug(
            f"Region {region.region_id} eligible for subdivision: "
            f"width_ratio={width_ratio:.2f}, words={len(words_in_region)}"
        )
        return True

    def _subdivide_region(
        self,
        region: FlorenceRegion,
        word_boxes: List[WordBox],
    ) -> List[FlorenceRegion]:
        """
        Apply projection histogram to find columns within region.

        Args:
            region: Region to subdivide
            word_boxes: All word boxes

        Returns:
            List of sub-regions (may be single-element if no columns found)
        """
        words_in_region = self._get_words_in_region(region, word_boxes)

        if not words_in_region:
            return [region]

        # Detect column boundaries using projection histogram
        column_boundaries = self._detect_columns_by_projection(
            words_in_region,
            region.bbox.x0,
            region.bbox.x1,
        )

        if len(column_boundaries) <= 1:
            # No subdivision needed
            return [region]

        # Create sub-regions for each column
        sub_regions = []
        for i, (col_x0, col_x1) in enumerate(column_boundaries):
            sub_region = FlorenceRegion(
                region_id=region.region_id * 100 + i,  # Unique ID
                region_type=region.region_type,
                semantic_label=region.semantic_label,
                bbox=BoundingBox(
                    x0=col_x0,
                    y0=region.bbox.y0,
                    x1=col_x1,
                    y1=region.bbox.y1,
                ),
                confidence=region.confidence,
                column_index=i,
                reading_order_hint=region.reading_order_hint,
            )
            sub_regions.append(sub_region)
            logger.debug(
                f"  Column {i}: x=[{col_x0:.0f}, {col_x1:.0f}], "
                f"width={col_x1 - col_x0:.0f}px"
            )

        return sub_regions

    def _detect_columns_by_projection(
        self,
        words: List[WordBox],
        region_x0: float,
        region_x1: float,
    ) -> List[Tuple[float, float]]:
        """
        Detect column boundaries using horizontal projection histogram.

        This is the core algorithm, adapted from column_detection.py
        but operates in local region coordinates.

        Args:
            words: Words within the region
            region_x0: Left edge of region
            region_x1: Right edge of region

        Returns:
            List of (x0, x1) tuples defining column boundaries
        """
        region_width = int(region_x1 - region_x0)
        if region_width <= 0:
            return [(region_x0, region_x1)]

        # Step 1: Build horizontal density histogram
        histogram = self._build_horizontal_histogram(words, region_x0, region_width)

        if histogram.max() == 0:
            return [(region_x0, region_x1)]

        # Step 2: Smooth histogram to reduce noise
        smoothed = gaussian_smooth(histogram, self.config.histogram_smooth_sigma)

        # Step 3: Find valleys (column gutters)
        valleys = self._find_valleys(smoothed)

        if not valleys:
            return [(region_x0, region_x1)]

        logger.debug(
            f"Found {len(valleys)} valleys at local positions: "
            f"{[f'{v:.0f}' for v in valleys]}"
        )

        # Step 4: Convert valleys to column boundaries
        columns = []
        prev_x = region_x0

        for valley_local in valleys:
            col_x1 = region_x0 + valley_local
            if col_x1 - prev_x >= self.config.min_column_width_px:
                columns.append((prev_x, col_x1))
            prev_x = col_x1

        # Add final column
        if region_x1 - prev_x >= self.config.min_column_width_px:
            columns.append((prev_x, region_x1))

        return columns if columns else [(region_x0, region_x1)]

    def _build_horizontal_histogram(
        self,
        words: List[WordBox],
        region_x0: float,
        region_width: int,
    ) -> np.ndarray:
        """
        Build horizontal density histogram for words.

        Each X position gets a count of how many words cover it.

        Args:
            words: Words to analyze
            region_x0: Left edge of region (for coordinate conversion)
            region_width: Width of histogram

        Returns:
            Numpy array of density counts
        """
        histogram = np.zeros(region_width, dtype=np.float32)

        for word in words:
            # Convert to local coordinates
            local_x0 = max(0, int(word.x0 - region_x0))
            local_x1 = min(region_width, int(word.x1 - region_x0))
            if local_x0 < local_x1:
                histogram[local_x0:local_x1] += 1

        return histogram

    def _find_valleys(self, histogram: np.ndarray) -> List[int]:
        """
        Find valley positions in histogram (low-density column gutters).

        Args:
            histogram: Smoothed horizontal density histogram

        Returns:
            List of valley center positions (in local coordinates)
        """
        # Calculate threshold for "low density"
        mean_density = np.mean(histogram)
        if mean_density == 0:
            return []

        threshold = mean_density * self.config.valley_threshold_ratio

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
                    if valley_width >= self.config.min_valley_width_px:
                        valley_center = (valley_start + x) // 2
                        valleys.append(valley_center)

                    in_valley = False

        # Handle valley extending to end of region
        if in_valley:
            valley_width = len(histogram) - valley_start
            if valley_width >= self.config.min_valley_width_px:
                valley_center = (valley_start + len(histogram)) // 2
                valleys.append(valley_center)

        return valleys

    def _get_words_in_region(
        self,
        region: FlorenceRegion,
        word_boxes: List[WordBox],
    ) -> List[WordBox]:
        """
        Get words whose centers fall within the region.

        Args:
            region: Region to check
            word_boxes: All word boxes

        Returns:
            List of words inside the region
        """
        return [
            w
            for w in word_boxes
            if (
                region.bbox.x0 <= w.center_x <= region.bbox.x1
                and region.bbox.y0 <= w.center_y <= region.bbox.y1
            )
        ]

    def _reassign_reading_order(self, regions: List[FlorenceRegion]) -> None:
        """
        Reassign reading order hints based on position after subdivision.

        Uses top-to-bottom, left-to-right ordering.

        Args:
            regions: List of regions to reorder (modified in place)
        """
        # Sort by Y (primary) then X (secondary) for natural reading order
        sorted_regions = sorted(
            regions, key=lambda r: (r.bbox.y0, r.bbox.x0)
        )
        for i, region in enumerate(sorted_regions):
            region.reading_order_hint = i
