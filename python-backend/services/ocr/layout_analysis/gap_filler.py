"""
Gap Filler for Layout Analysis.

Creates synthetic regions for gaps between detected regions where
orphan words accumulate. This handles cases where PP-DocLayout
misses certain areas (like date/tag rows between header and body).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..reading_order.data_structures import WordBox
from ..multipass.data_structures import (
    FlorenceRegion,
    BoundingBox,
    RegionType,
)

logger = logging.getLogger(__name__)


@dataclass
class GapFillerConfig:
    """Configuration for gap filling."""

    # Minimum gap height to consider filling
    min_gap_height_px: int = 50

    # Minimum words in a gap to create a synthetic region
    min_words_for_gap_region: int = 2

    # Horizontal grouping threshold (words within this distance are grouped)
    horizontal_group_threshold_px: int = 100

    # Vertical grouping threshold for lines within a gap
    vertical_line_threshold_px: int = 30

    # Padding around synthetic regions
    region_padding_px: int = 10


class GapFiller:
    """
    Creates synthetic regions for gaps in layout detection.

    PP-DocLayout sometimes misses certain document elements:
    - Date/tag rows between header and body
    - Section headings between regions
    - Scattered labels or annotations

    This class identifies gaps between detected regions, finds orphan
    words in those gaps, and creates synthetic text regions to contain them.

    Usage:
        filler = GapFiller()
        filled_regions = filler.fill_gaps(
            regions=layout_result.regions,
            word_boxes=ocr_word_boxes,
            page_width=1200,
            page_height=1600,
        )
    """

    def __init__(self, config: Optional[GapFillerConfig] = None):
        """
        Initialize gap filler.

        Args:
            config: Gap filling configuration. Uses defaults if None.
        """
        self.config = config or GapFillerConfig()
        logger.info(
            f"GapFiller initialized: "
            f"min_gap_height={self.config.min_gap_height_px}px, "
            f"min_words={self.config.min_words_for_gap_region}"
        )

    def fill_gaps(
        self,
        regions: List[FlorenceRegion],
        word_boxes: List[WordBox],
        page_width: int,
        page_height: int,
    ) -> List[FlorenceRegion]:
        """
        Fill gaps between regions with synthetic regions for orphan words.

        Args:
            regions: List of FlorenceRegion from PP-DocLayout
            word_boxes: List of WordBox from OCR
            page_width: Page width in pixels
            page_height: Page height in pixels

        Returns:
            Extended list of regions including synthetic gap regions.
        """
        if not regions or not word_boxes:
            return regions

        # Step 1: Find orphan words (not contained in any region)
        orphan_words = self._find_orphan_words(word_boxes, regions)

        if not orphan_words:
            logger.debug("No orphan words found - no gap filling needed")
            return regions

        logger.info(f"Found {len(orphan_words)} orphan words for gap analysis")

        # Step 2: Identify gaps between regions
        gaps = self._find_vertical_gaps(regions, page_height)

        if not gaps:
            logger.debug("No significant gaps found between regions")
            return regions

        logger.info(f"Found {len(gaps)} vertical gaps between regions")

        # Step 3: Group orphan words by gap
        gap_word_groups = self._group_words_by_gap(orphan_words, gaps)

        # Step 4: Create synthetic regions for gap word groups
        synthetic_regions = []
        next_region_id = max(r.region_id for r in regions) + 1000  # Avoid ID collision

        for gap_idx, (gap_y0, gap_y1) in enumerate(gaps):
            words_in_gap = gap_word_groups.get(gap_idx, [])

            if len(words_in_gap) < self.config.min_words_for_gap_region:
                continue

            # Create synthetic region(s) for this gap
            new_regions = self._create_gap_regions(
                words_in_gap, gap_y0, gap_y1, next_region_id
            )

            for region in new_regions:
                synthetic_regions.append(region)
                logger.info(
                    f"Created synthetic region {region.region_id} in gap "
                    f"y=[{gap_y0:.0f}, {gap_y1:.0f}] with {len(words_in_gap)} words"
                )
                next_region_id += 1

        # Step 5: Merge original and synthetic regions
        if synthetic_regions:
            all_regions = list(regions) + synthetic_regions

            # Re-assign reading order hints based on position
            self._reassign_reading_order(all_regions)

            logger.info(
                f"Gap filling complete: {len(synthetic_regions)} synthetic regions added, "
                f"{len(all_regions)} total regions"
            )
            return all_regions

        return regions

    def _find_orphan_words(
        self,
        word_boxes: List[WordBox],
        regions: List[FlorenceRegion],
    ) -> List[WordBox]:
        """Find words whose centers don't fall within any region."""
        orphans = []
        for word in word_boxes:
            in_any_region = False
            for region in regions:
                if (
                    region.bbox.x0 <= word.center_x <= region.bbox.x1
                    and region.bbox.y0 <= word.center_y <= region.bbox.y1
                ):
                    in_any_region = True
                    break

            if not in_any_region:
                orphans.append(word)

        return orphans

    def _find_vertical_gaps(
        self,
        regions: List[FlorenceRegion],
        page_height: int,
    ) -> List[Tuple[float, float]]:
        """
        Find vertical gaps between regions.

        Returns list of (y0, y1) tuples representing gaps.
        """
        if not regions:
            return [(0, page_height)]

        # Get Y extents of all regions
        y_ranges = [(r.bbox.y0, r.bbox.y1) for r in regions]
        y_ranges.sort(key=lambda x: x[0])

        gaps = []

        # Check gap at top of page
        if y_ranges[0][0] > self.config.min_gap_height_px:
            gaps.append((0, y_ranges[0][0]))

        # Check gaps between regions
        for i in range(len(y_ranges) - 1):
            current_end = y_ranges[i][1]
            next_start = y_ranges[i + 1][0]

            gap_height = next_start - current_end
            if gap_height >= self.config.min_gap_height_px:
                gaps.append((current_end, next_start))

        # Check gap at bottom of page
        if page_height - y_ranges[-1][1] > self.config.min_gap_height_px:
            gaps.append((y_ranges[-1][1], page_height))

        return gaps

    def _group_words_by_gap(
        self,
        orphan_words: List[WordBox],
        gaps: List[Tuple[float, float]],
    ) -> dict:
        """Group orphan words by which gap they fall into."""
        gap_groups = {i: [] for i in range(len(gaps))}

        for word in orphan_words:
            word_cy = word.center_y

            for gap_idx, (gap_y0, gap_y1) in enumerate(gaps):
                if gap_y0 <= word_cy <= gap_y1:
                    gap_groups[gap_idx].append(word)
                    break

        return gap_groups

    def _create_gap_regions(
        self,
        words: List[WordBox],
        gap_y0: float,
        gap_y1: float,
        start_region_id: int,
    ) -> List[FlorenceRegion]:
        """
        Create synthetic region(s) for words in a gap.

        May create multiple regions if words are horizontally separated.
        """
        if not words:
            return []

        # Group words into horizontal clusters
        word_clusters = self._cluster_words_horizontally(words)

        regions = []
        for cluster_idx, cluster_words in enumerate(word_clusters):
            if len(cluster_words) < self.config.min_words_for_gap_region:
                continue

            # Calculate bounding box for this cluster
            x0 = min(w.x0 for w in cluster_words) - self.config.region_padding_px
            y0 = min(w.y0 for w in cluster_words) - self.config.region_padding_px
            x1 = max(w.x1 for w in cluster_words) + self.config.region_padding_px
            y1 = max(w.y1 for w in cluster_words) + self.config.region_padding_px

            # Clamp to gap bounds
            y0 = max(y0, gap_y0)
            y1 = min(y1, gap_y1)

            # Determine semantic label based on position and content
            label = self._infer_gap_region_label(cluster_words, gap_y0)

            region = FlorenceRegion(
                region_id=start_region_id + cluster_idx,
                region_type=RegionType.TEXT,
                semantic_label=label,
                bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
                confidence=0.7,  # Lower confidence for synthetic regions
                column_index=None,
                reading_order_hint=None,  # Will be assigned later
            )
            regions.append(region)

        return regions

    def _cluster_words_horizontally(
        self,
        words: List[WordBox],
    ) -> List[List[WordBox]]:
        """
        Cluster words into horizontal groups.

        Words that are horizontally close together are grouped.
        """
        if not words:
            return []

        # Sort by x position
        sorted_words = sorted(words, key=lambda w: w.x0)

        clusters = []
        current_cluster = [sorted_words[0]]

        for word in sorted_words[1:]:
            # Check if word is close to current cluster
            cluster_x1 = max(w.x1 for w in current_cluster)

            if word.x0 - cluster_x1 <= self.config.horizontal_group_threshold_px:
                current_cluster.append(word)
            else:
                clusters.append(current_cluster)
                current_cluster = [word]

        clusters.append(current_cluster)
        return clusters

    def _infer_gap_region_label(
        self,
        words: List[WordBox],
        gap_y0: float,
    ) -> str:
        """
        Infer semantic label for a synthetic gap region.

        Heuristics:
        - Large font (tall words) at top of gap -> likely heading/title
        - Date-like patterns -> likely metadata/date
        - Otherwise -> text
        """
        if not words:
            return "text"

        # Check for large text (potential heading)
        avg_height = sum(w.height for w in words) / len(words)
        max_height = max(w.height for w in words)

        # Large text is likely a heading
        if max_height > 100:
            return "paragraph_title"

        # Check for date-like patterns in text
        text_combined = " ".join(w.text for w in words)
        if any(
            pattern in text_combined.lower()
            for pattern in ["20", "19", "/", "-", "jan", "feb", "mar", "apr"]
        ):
            return "number"  # Date/metadata

        return "text"

    def _reassign_reading_order(self, regions: List[FlorenceRegion]) -> None:
        """Reassign reading order hints based on position."""
        sorted_regions = sorted(regions, key=lambda r: (r.bbox.y0, r.bbox.x0))
        for i, region in enumerate(sorted_regions):
            region.reading_order_hint = i
