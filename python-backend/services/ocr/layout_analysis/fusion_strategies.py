"""
Fusion Strategies for OCR + Florence-2 Layout Combination.

SimpleFusion: Use Florence-2 regions as hints for grouping OCR words.
AdvancedFusion: (Future) Crop and process regions separately.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ..reading_order.data_structures import WordBox, Line, Block
from ..reading_order.line_formation import form_lines
from ..reading_order.config import ReadingOrderConfig
from ..multipass.data_structures import FlorenceLayoutResult, FlorenceRegion, RegionType, LayoutType

logger = logging.getLogger(__name__)


class ReadingOrderFusion(ABC):
    """
    Base class for fusion strategies.

    Combines OCR word boxes with Florence-2 layout regions
    to produce properly ordered text blocks.
    """

    @abstractmethod
    def fuse(
        self,
        ocr_words: List[WordBox],
        florence_result: FlorenceLayoutResult,
    ) -> List[Block]:
        """
        Combine OCR and layout data into ordered blocks.

        Args:
            ocr_words: List of WordBox from OCR
            florence_result: Layout analysis from Florence-2

        Returns:
            List of Block objects with reading order assigned
        """
        pass


class SimpleFusion(ReadingOrderFusion):
    """
    Simple fusion: Use Florence-2 regions as hints for grouping.

    Process:
    1. Form lines from OCR word boxes (geometric clustering)
    2. Assign lines to Florence-2 regions (spatial overlap)
    3. Form blocks within regions
    4. Assign reading order based on region types and positions
    """

    def __init__(
        self,
        line_vertical_threshold: float = 0.5,
        line_baseline_tolerance: float = 0.3,
        line_horizontal_gap: float = 3.0,
        block_vertical_gap: float = 2.0,
        min_region_overlap: float = 0.5,
    ):
        """
        Initialize SimpleFusion.

        Args:
            line_vertical_threshold: Max vertical distance (as fraction of height) for same line
            line_baseline_tolerance: Baseline alignment tolerance (as fraction of height)
            line_horizontal_gap: Max horizontal gap (as multiple of avg char width)
            block_vertical_gap: Max vertical gap between lines in block (as multiple of line height)
            min_region_overlap: Minimum overlap ratio to assign line to region
        """
        self.line_vertical_threshold = line_vertical_threshold
        self.line_baseline_tolerance = line_baseline_tolerance
        self.line_horizontal_gap = line_horizontal_gap
        self.block_vertical_gap = block_vertical_gap
        self.min_region_overlap = min_region_overlap

    def fuse(
        self,
        ocr_words: List[WordBox],
        florence_result: FlorenceLayoutResult,
    ) -> List[Block]:
        """Combine OCR and layout data into ordered blocks."""
        if not ocr_words:
            return []

        page_width = florence_result.image_width
        page_height = florence_result.image_height

        # Step 1: Form lines from OCR words
        lines = self._form_lines(ocr_words, page_height)

        # Step 2: Assign lines to Florence regions
        self._assign_lines_to_regions(lines, florence_result.regions)

        # Step 3: Form blocks by region
        blocks = self._form_blocks_by_region(
            lines, florence_result.regions, page_width, page_height
        )

        # Step 4: Assign reading order
        ordered_blocks = self._assign_reading_order(
            blocks, florence_result.layout_type, page_width
        )

        return ordered_blocks

    def _form_lines(self, words: List[WordBox], page_height: float) -> List[Line]:
        """Group words into lines based on vertical proximity and baseline alignment."""
        if not words:
            return []

        # Sort by Y, then X
        sorted_words = sorted(words, key=lambda w: (w.y0, w.x0))

        lines = []
        current_line_words = [sorted_words[0]]
        line_id = 0

        for word in sorted_words[1:]:
            if self._belongs_to_same_line(current_line_words[-1], word):
                current_line_words.append(word)
            else:
                # Create line and start new one
                line = Line(words=current_line_words, line_id=line_id)
                line.sort_words_left_to_right()
                lines.append(line)
                line_id += 1
                current_line_words = [word]

        # Don't forget last line
        if current_line_words:
            line = Line(words=current_line_words, line_id=line_id)
            line.sort_words_left_to_right()
            lines.append(line)

        return lines

    def _belongs_to_same_line(self, prev: WordBox, curr: WordBox) -> bool:
        """Check if current word belongs to same line as previous."""
        avg_height = (prev.height + curr.height) / 2

        # Vertical distance check
        vertical_distance = abs(curr.center_y - prev.center_y)
        vertical_threshold = avg_height * self.line_vertical_threshold

        if vertical_distance >= vertical_threshold:
            return False

        # Baseline alignment check
        baseline_diff = abs(curr.baseline - prev.baseline)
        baseline_threshold = avg_height * self.line_baseline_tolerance

        if baseline_diff >= baseline_threshold:
            return False

        # Horizontal gap check
        horizontal_gap = curr.x0 - prev.x1
        max_gap = avg_height * self.line_horizontal_gap

        if horizontal_gap >= max_gap:
            return False

        return True

    def _assign_lines_to_regions(
        self,
        lines: List[Line],
        regions: List[FlorenceRegion],
    ) -> None:
        """Assign each line to best matching Florence-2 region."""
        for line in lines:
            best_region = None
            best_overlap = 0.0

            for region in regions:
                overlap = self._calculate_overlap(
                    line.x0, line.y0, line.x1, line.y1,
                    region.bbox.x0, region.bbox.y0, region.bbox.x1, region.bbox.y1,
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_region = region

            if best_region and best_overlap >= self.min_region_overlap:
                # Store region info in line
                # Since Line doesn't have these fields, we'll use a workaround
                # by storing them temporarily as attributes
                line._florence_region_id = best_region.region_id
                line._florence_region_type = best_region.region_type
                line._florence_column_index = best_region.column_index
            else:
                line._florence_region_id = None
                line._florence_region_type = None
                line._florence_column_index = None

    def _calculate_overlap(
        self,
        ax0: float, ay0: float, ax1: float, ay1: float,
        bx0: float, by0: float, bx1: float, by1: float,
    ) -> float:
        """Calculate overlap ratio (intersection / box A area)."""
        x_left = max(ax0, bx0)
        y_top = max(ay0, by0)
        x_right = min(ax1, bx1)
        y_bottom = min(ay1, by1)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        box_a_area = (ax1 - ax0) * (ay1 - ay0)

        return intersection / box_a_area if box_a_area > 0 else 0.0

    def _form_blocks_by_region(
        self,
        lines: List[Line],
        regions: List[FlorenceRegion],
        page_width: float,
        page_height: float,
    ) -> List[Block]:
        """Form blocks, grouping by Florence-2 region."""
        blocks = []
        block_id = 0

        # Group lines by region
        for region in regions:
            region_lines = [
                l for l in lines
                if getattr(l, "_florence_region_id", None) == region.region_id
            ]

            if not region_lines:
                continue

            # Form blocks within this region
            region_blocks = self._form_blocks(region_lines, page_width, page_height)

            for block in region_blocks:
                block.block_id = block_id
                block.region_id = region.region_id
                # Map Florence region type to block type
                block.block_type = self._map_region_to_block_type(region.region_type)
                blocks.append(block)
                block_id += 1

        # Handle orphan lines (not in any region)
        orphan_lines = [
            l for l in lines
            if getattr(l, "_florence_region_id", None) is None
        ]

        if orphan_lines:
            orphan_blocks = self._form_blocks(orphan_lines, page_width, page_height)
            for block in orphan_blocks:
                block.block_id = block_id
                block_id += 1
            blocks.extend(orphan_blocks)

        return blocks

    def _form_blocks(
        self,
        lines: List[Line],
        page_width: float,
        page_height: float,
    ) -> List[Block]:
        """Form blocks from lines based on vertical proximity."""
        if not lines:
            return []

        # Sort by Y position
        sorted_lines = sorted(lines, key=lambda l: l.y0)

        blocks = []
        current_block_lines = [sorted_lines[0]]
        block_id = 0

        for line in sorted_lines[1:]:
            if self._belongs_to_same_block(current_block_lines, line):
                current_block_lines.append(line)
            else:
                block = Block(lines=current_block_lines, block_id=block_id)
                block.sort_lines_top_to_bottom()
                blocks.append(block)
                block_id += 1
                current_block_lines = [line]

        # Don't forget last block
        if current_block_lines:
            block = Block(lines=current_block_lines, block_id=block_id)
            block.sort_lines_top_to_bottom()
            blocks.append(block)

        return blocks

    def _belongs_to_same_block(
        self,
        block_lines: List[Line],
        new_line: Line,
    ) -> bool:
        """Check if new line belongs to current block."""
        prev_line = block_lines[-1]

        # Calculate average line height
        avg_height = sum(l.height for l in block_lines) / len(block_lines)

        # Vertical gap check
        vertical_gap = new_line.y0 - prev_line.y1
        max_gap = avg_height * self.block_vertical_gap

        if vertical_gap >= max_gap:
            return False

        # Horizontal overlap check
        overlap = self._calculate_overlap(
            prev_line.x0, prev_line.y0, prev_line.x1, prev_line.y1,
            new_line.x0, new_line.y0, new_line.x1, new_line.y1,
        )

        # Allow some horizontal misalignment
        return True  # For now, just use vertical gap

    def _map_region_to_block_type(self, region_type: RegionType) -> str:
        """Map PP-DocLayout region type to block type."""
        mapping = {
            # PP-DocLayout text region types -> "main"
            RegionType.TEXT: "main",
            RegionType.PARAGRAPH_TITLE: "main",
            RegionType.DOC_TITLE: "title",
            RegionType.ABSTRACT: "main",
            RegionType.CONTENT: "main",
            RegionType.SIDEBAR_TEXT: "sidebar",
            RegionType.REFERENCE: "reference",
            RegionType.ALGORITHM: "main",
            RegionType.FORMULA: "formula",
            RegionType.FORMULA_NUMBER: "formula",
            
            # Structural types
            RegionType.HEADER: "header",
            RegionType.HEADER_IMAGE: "header",
            RegionType.FOOTER: "footer",
            RegionType.FOOTER_IMAGE: "footer",
            RegionType.FOOTNOTE: "footnote",
            
            # Visual elements
            RegionType.IMAGE: "figure",
            RegionType.FIGURE_TITLE: "caption",
            RegionType.CHART: "figure",
            RegionType.CHART_TITLE: "caption",
            RegionType.TABLE: "table",
            RegionType.TABLE_TITLE: "caption",
            RegionType.SEAL: "figure",
            RegionType.NUMBER: "main",
            
            # Legacy Florence-2 types (for backward compatibility)
            RegionType.COLUMN: "main",
            RegionType.TEXT_REGION: "main",
            RegionType.SIDEBAR: "sidebar",
            RegionType.FIGURE: "figure",
            RegionType.SIGNATURE: "signature",
            RegionType.CAPTION: "caption",
            RegionType.LIST: "main",
            RegionType.UNKNOWN: "main",
        }
        return mapping.get(region_type, "main")

    def _assign_reading_order(
        self,
        blocks: List[Block],
        layout_type: LayoutType,
        page_width: float,
    ) -> List[Block]:
        """Assign reading order based on layout type."""
        # Separate by block type
        titles = [b for b in blocks if b.block_type == "title"]
        headers = [b for b in blocks if b.block_type == "header"]
        footers = [b for b in blocks if b.block_type in ("footer", "footnote")]
        captions = [b for b in blocks if b.block_type == "caption"]
        figures = [b for b in blocks if b.block_type == "figure"]
        main_blocks = [b for b in blocks if b.block_type in ("main", "column", "sidebar", "reference", "formula")]
        other = [b for b in blocks if b not in titles and b not in headers and b not in footers 
                 and b not in captions and b not in figures and b not in main_blocks]

        ordered = []
        reading_order = 0

        # Titles first (typically at top)
        for block in sorted(titles, key=lambda b: b.y0):
            block.reading_order = reading_order
            ordered.append(block)
            reading_order += 1

        # Headers next (top to bottom)
        for block in sorted(headers, key=lambda b: b.y0):
            block.reading_order = reading_order
            ordered.append(block)
            reading_order += 1

        # Main content based on layout type
        if layout_type == LayoutType.SINGLE_COLUMN:
            # Top to bottom
            for block in sorted(main_blocks, key=lambda b: b.y0):
                block.reading_order = reading_order
                ordered.append(block)
                reading_order += 1

        elif layout_type == LayoutType.TWO_COLUMN:
            # Left column top to bottom, then right column
            left = [b for b in main_blocks if b.center_x < page_width / 2]
            right = [b for b in main_blocks if b.center_x >= page_width / 2]

            for block in sorted(left, key=lambda b: b.y0):
                block.reading_order = reading_order
                ordered.append(block)
                reading_order += 1

            for block in sorted(right, key=lambda b: b.y0):
                block.reading_order = reading_order
                ordered.append(block)
                reading_order += 1

        elif layout_type == LayoutType.MULTI_COLUMN:
            # Sort by column index (from layout analysis), then Y
            def get_column_index(block: Block) -> int:
                if block.lines:
                    return getattr(block.lines[0], "_florence_column_index", 0) or 0
                return 0

            for block in sorted(main_blocks, key=lambda b: (get_column_index(b), b.y0)):
                block.reading_order = reading_order
                ordered.append(block)
                reading_order += 1

        else:
            # Default: top to bottom, left to right
            for block in sorted(main_blocks, key=lambda b: (b.y0, b.x0)):
                block.reading_order = reading_order
                ordered.append(block)
                reading_order += 1

        # Figures and their captions
        for block in sorted(figures, key=lambda b: b.y0):
            block.reading_order = reading_order
            ordered.append(block)
            reading_order += 1

        for block in sorted(captions, key=lambda b: b.y0):
            block.reading_order = reading_order
            ordered.append(block)
            reading_order += 1

        # Other content
        for block in sorted(other, key=lambda b: b.y0):
            block.reading_order = reading_order
            ordered.append(block)
            reading_order += 1

        # Footers last
        for block in sorted(footers, key=lambda b: b.y0):
            block.reading_order = reading_order
            ordered.append(block)
            reading_order += 1

        return ordered


class RegionFirstFusion(ReadingOrderFusion):
    """
    Region-first fusion: Assign words to regions FIRST, then form lines within each region.

    This approach prevents text from different columns being mixed together.

    Process:
    1. Assign each OCR word to its best-matching layout region
    2. Within each region, form lines from words (geometric clustering)
    3. Within each region, form blocks from lines
    4. Order blocks according to region reading order
    """

    def __init__(
        self,
        line_vertical_threshold: float = 0.6,
        line_horizontal_gap: float = 1.5,
        block_vertical_gap: float = 1.5,
        min_word_region_overlap: float = 0.3,
    ):
        """
        Initialize RegionFirstFusion.

        Args:
            line_vertical_threshold: Max vertical distance (as fraction of height) for same line
            line_horizontal_gap: Max horizontal gap (as multiple of avg char width)
            block_vertical_gap: Max vertical gap between lines (as multiple of line height)
            min_word_region_overlap: Minimum overlap for word-to-region assignment
        """
        self.line_vertical_threshold = line_vertical_threshold
        self.line_horizontal_gap = line_horizontal_gap
        self.block_vertical_gap = block_vertical_gap
        self.min_word_region_overlap = min_word_region_overlap

        # Use lenient line formation config for perspective-distorted documents
        # Perspective distortion causes words on same line to have different Y coords
        self.line_formation_config = ReadingOrderConfig(
            line_vertical_overlap_threshold=0.3,  # More lenient overlap
            line_baseline_tolerance_multiplier=0.8,  # Allow baseline variation
            line_horizontal_gap_multiplier=4.0,  # Allow wider gaps
            line_vertical_threshold_multiplier=0.8,  # More lenient vertical
            line_y_spread_multiplier=1.2,  # Allow more Y variation for perspective
        )

    def fuse(self, ocr_words: List[WordBox], layout_result: FlorenceLayoutResult) -> List[Block]:
        """Combine OCR and layout data using region-first approach."""
        if not ocr_words:
            return []

        page_width = layout_result.image_width
        page_height = layout_result.image_height
        regions = layout_result.regions

        # Step 1: Assign each word to a region
        region_words: Dict[int, List[WordBox]] = {r.region_id: [] for r in regions}
        orphan_words: List[WordBox] = []

        for word in ocr_words:
            best_region = self._find_best_region(word, regions)
            if best_region is not None:
                region_words[best_region.region_id].append(word)
            else:
                orphan_words.append(word)

        # Step 2 & 3: For each region, form lines then blocks
        all_blocks: List[Block] = []
        block_id = 0

        for region in regions:
            words = region_words.get(region.region_id, [])
            if not words:
                continue

            # Form lines within this region using robust line formation
            lines = self._form_lines_in_region(words, page_height)

            # Form blocks from lines
            blocks = self._form_blocks_from_lines(lines)

            # Assign region info to blocks
            for block in blocks:
                block.block_id = block_id
                block.region_id = region.region_id
                block.block_type = self._map_region_to_block_type(region.region_type)
                # Store region reading order for later sorting
                block._region_reading_order = region.reading_order_hint
                all_blocks.append(block)
                block_id += 1

        # Handle orphan words (not in any region)
        if orphan_words:
            orphan_lines = self._form_lines_in_region(orphan_words, page_height)
            orphan_blocks = self._form_blocks_from_lines(orphan_lines)
            for block in orphan_blocks:
                block.block_id = block_id
                block.block_type = "orphan"
                block._region_reading_order = 999  # Put orphans at end
                all_blocks.append(block)
                block_id += 1

        # Step 4: Assign reading order based on region order
        ordered_blocks = self._assign_reading_order(all_blocks, layout_result.layout_type, page_width)

        return ordered_blocks

    def _find_best_region(
        self,
        word: WordBox,
        regions: List[FlorenceRegion],
    ) -> Optional[FlorenceRegion]:
        """Find the best matching region for a word based on containment, overlap, or proximity."""
        word_cx = word.center_x
        word_cy = word.center_y

        best_region = None
        best_score = 0.0

        # First pass: check containment (word center inside region)
        for region in regions:
            if (region.bbox.x0 <= word_cx <= region.bbox.x1 and
                region.bbox.y0 <= word_cy <= region.bbox.y1):
                # Word center is inside region - calculate containment score
                # Prefer smaller regions (more specific)
                region_area = region.bbox.width * region.bbox.height
                if region_area > 0:
                    score = 1.0 / region_area  # Smaller area = higher score
                    if score > best_score:
                        best_score = score
                        best_region = region

        if best_region:
            return best_region

        # Second pass: check overlap
        for region in regions:
            overlap = self._calculate_overlap(
                word.x0, word.y0, word.x1, word.y1,
                region.bbox.x0, region.bbox.y0, region.bbox.x1, region.bbox.y1,
            )
            if overlap >= self.min_word_region_overlap and overlap > best_score:
                best_score = overlap
                best_region = region

        if best_region:
            return best_region

        # Third pass: find nearest text region by distance (for orphan words)
        # Only consider text regions for distance-based assignment
        text_regions = [r for r in regions if r.region_type in (
            RegionType.TEXT, RegionType.PARAGRAPH_TITLE, RegionType.DOC_TITLE,
            RegionType.ABSTRACT, RegionType.CONTENT, RegionType.SIDEBAR_TEXT,
            RegionType.HEADER, RegionType.FOOTER, RegionType.FOOTNOTE,
            RegionType.REFERENCE, RegionType.ALGORITHM,
        )]

        # Separate into regions where word is properly inside Y range vs boundary cases
        # A word is "properly inside" if it's well within the Y range, not just at the edge
        properly_inside_regions = []
        boundary_regions = []

        for region in text_regions:
            y_tolerance = 50  # Allow some slack for book curvature

            # Calculate how deeply the word is inside the region's Y range
            # Positive = inside, negative = outside
            y_margin_top = word_cy - region.bbox.y0
            y_margin_bottom = region.bbox.y1 - word_cy

            if y_margin_top >= 0 and y_margin_bottom >= 0:
                # Word is properly inside the Y range
                properly_inside_regions.append((region, min(y_margin_top, y_margin_bottom)))
            elif y_margin_top >= -y_tolerance and y_margin_bottom >= -y_tolerance:
                # Word is at the boundary (within tolerance)
                boundary_regions.append(region)

        # First, try to find best properly-inside region
        # Prefer regions where word is more deeply inside (larger margin)
        # and horizontally closest
        best_region = None
        best_score = float('-inf')

        for region, y_margin in properly_inside_regions:
            dx = max(region.bbox.x0 - word_cx, 0, word_cx - region.bbox.x1)
            if dx < 500:
                # Score = y_margin - dx (prefer deeper inside Y and smaller X distance)
                # Weight Y margin more heavily to prioritize proper containment
                score = y_margin * 2 - dx
                if score > best_score:
                    best_score = score
                    best_region = region

        if best_region:
            return best_region

        # Fall back to boundary regions, preferring closest
        min_distance = float('inf')
        nearest_region = None

        for region in boundary_regions:
            dx = max(region.bbox.x0 - word_cx, 0, word_cx - region.bbox.x1)
            if dx < min_distance and dx < 500:
                min_distance = dx
                nearest_region = region

        if nearest_region:
            return nearest_region

        # Fall back to finding nearest region by full Euclidean distance
        min_distance = float('inf')

        for region in text_regions:
            dx = max(region.bbox.x0 - word_cx, 0, word_cx - region.bbox.x1)
            dy = max(region.bbox.y0 - word_cy, 0, word_cy - region.bbox.y1)
            distance = (dx * dx + dy * dy) ** 0.5

            if distance < min_distance and distance < 500:
                min_distance = distance
                nearest_region = region

        return nearest_region

    def _calculate_overlap(
        self,
        ax0: float, ay0: float, ax1: float, ay1: float,
        bx0: float, by0: float, bx1: float, by1: float,
    ) -> float:
        """Calculate overlap ratio (intersection / word area)."""
        x_left = max(ax0, bx0)
        y_top = max(ay0, by0)
        x_right = min(ax1, bx1)
        y_bottom = min(ay1, by1)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        word_area = (ax1 - ax0) * (ay1 - ay0)

        return intersection / word_area if word_area > 0 else 0.0

    def _form_lines_in_region(self, words: List[WordBox], page_height: float = 0) -> List[Line]:
        """Form lines using graph-based grouping with adaptive Y tolerance.

        This algorithm handles book spine curvature by using LOCAL comparisons
        between horizontally adjacent words, rather than global curvature estimation.

        Algorithm:
        1. Build adjacency graph: words are connected if horizontally close with similar Y
        2. Y tolerance scales with horizontal distance (handles curvature naturally)
        3. Find connected components - each component is a line
        4. Sort lines by average Y for reading order
        """
        if not words:
            return []

        if len(words) < 2:
            line = Line(words=list(words), line_id=0)
            return [line]

        # Calculate statistics
        avg_height = sum(w.height for w in words) / len(words)
        page_width = max(w.x1 for w in words) - min(w.x0 for w in words)

        # Parameters for line detection
        # Max horizontal gap between words on same line (typical word spacing)
        max_horizontal_gap = avg_height * 3.0

        # Base Y tolerance for adjacent words
        base_y_tolerance = avg_height * 0.6

        # Additional Y tolerance per unit of horizontal distance (curvature compensation)
        # ~0.02 means 20 pixels of Y variation per 1000 pixels of X distance
        curvature_tolerance_rate = 0.025

        # Union-Find data structure for grouping words
        parent = list(range(len(words)))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Sort words by X for efficient neighbor finding
        indexed_words = [(i, w) for i, w in enumerate(words)]
        indexed_words.sort(key=lambda iw: iw[1].x0)

        # Build adjacency graph: connect words that belong on the same line
        for idx in range(len(indexed_words)):
            i, word_i = indexed_words[idx]

            # Look at subsequent words (already sorted by X)
            for jdx in range(idx + 1, len(indexed_words)):
                j, word_j = indexed_words[jdx]

                # Horizontal gap between words
                h_gap = word_j.x0 - word_i.x1

                # If horizontal gap is too large, no need to check further
                # (words are sorted by X, so all subsequent will have larger gaps)
                if h_gap > max_horizontal_gap:
                    break

                # For overlapping or close words, check Y alignment
                # Y tolerance increases with horizontal distance to handle curvature
                x_distance = abs(word_j.center_x - word_i.center_x)
                y_tolerance = base_y_tolerance + curvature_tolerance_rate * x_distance

                y_diff = abs(word_j.center_y - word_i.center_y)

                if y_diff <= y_tolerance:
                    union(i, j)

        # Also connect words that overlap horizontally (covers cases where
        # words are not strictly left-to-right but still on same line)
        for idx in range(len(indexed_words)):
            i, word_i = indexed_words[idx]

            for jdx in range(idx + 1, len(indexed_words)):
                j, word_j = indexed_words[jdx]

                # Check if words overlap horizontally
                overlap_x = min(word_i.x1, word_j.x1) - max(word_i.x0, word_j.x0)
                if overlap_x > 0:
                    # Words overlap horizontally - check Y with tight tolerance
                    y_diff = abs(word_j.center_y - word_i.center_y)
                    if y_diff <= base_y_tolerance:
                        union(i, j)

                # Stop if word_j is completely past word_i
                if word_j.x0 > word_i.x1 + max_horizontal_gap:
                    break

        # Group words by their root parent (connected component)
        groups = {}
        for i, word in enumerate(words):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(word)

        # Convert groups to lines
        lines = []
        for group_words in groups.values():
            # Sort words in each line by X
            sorted_line_words = sorted(group_words, key=lambda w: w.x0)
            line = Line(words=sorted_line_words, line_id=len(lines))
            lines.append(line)

        # Sort lines by average Y (top to bottom reading order)
        lines.sort(key=lambda ln: sum(w.center_y for w in ln.words) / len(ln.words))

        # Renumber line IDs
        for idx, line in enumerate(lines):
            line.line_id = idx

        return lines



    def _form_blocks_from_lines(self, lines: List[Line]) -> List[Block]:
        """Form blocks from lines based on vertical proximity."""
        if not lines:
            return []

        # Sort lines by Y position
        sorted_lines = sorted(lines, key=lambda l: l.y0)

        blocks = []
        current_block_lines = [sorted_lines[0]]
        block_id = 0

        for line in sorted_lines[1:]:
            if self._line_belongs_to_block(current_block_lines, line):
                current_block_lines.append(line)
            else:
                # Finalize current block
                block = Block(lines=current_block_lines, block_id=block_id)
                block.sort_lines_top_to_bottom()
                blocks.append(block)
                block_id += 1
                current_block_lines = [line]

        # Don't forget last block
        if current_block_lines:
            block = Block(lines=current_block_lines, block_id=block_id)
            block.sort_lines_top_to_bottom()
            blocks.append(block)

        return blocks

    def _line_belongs_to_block(self, block_lines: List[Line], line: Line) -> bool:
        """Check if line belongs to the same block."""
        prev_line = block_lines[-1]
        avg_height = sum(l.height for l in block_lines) / len(block_lines)

        # Vertical gap check
        vertical_gap = line.y0 - prev_line.y1
        max_gap = avg_height * self.block_vertical_gap

        return vertical_gap <= max_gap

    def _map_region_to_block_type(self, region_type: RegionType) -> str:
        """Map PP-DocLayout region type to block type."""
        mapping = {
            # Text types
            RegionType.TEXT: "main",
            RegionType.PARAGRAPH_TITLE: "heading",
            RegionType.DOC_TITLE: "title",
            RegionType.ABSTRACT: "main",
            RegionType.CONTENT: "main",
            RegionType.SIDEBAR_TEXT: "sidebar",
            RegionType.REFERENCE: "reference",
            RegionType.ALGORITHM: "main",
            RegionType.FORMULA: "formula",
            RegionType.FORMULA_NUMBER: "formula",
            RegionType.NUMBER: "main",

            # Structural
            RegionType.HEADER: "header",
            RegionType.HEADER_IMAGE: "header",
            RegionType.FOOTER: "footer",
            RegionType.FOOTER_IMAGE: "footer",
            RegionType.FOOTNOTE: "footnote",

            # Visual
            RegionType.IMAGE: "figure",
            RegionType.FIGURE_TITLE: "caption",
            RegionType.CHART: "figure",
            RegionType.CHART_TITLE: "caption",
            RegionType.TABLE: "table",
            RegionType.TABLE_TITLE: "caption",
            RegionType.SEAL: "figure",

            # Legacy
            RegionType.COLUMN: "main",
            RegionType.TEXT_REGION: "main",
            RegionType.SIDEBAR: "sidebar",
            RegionType.FIGURE: "figure",
            RegionType.SIGNATURE: "signature",
            RegionType.CAPTION: "caption",
            RegionType.LIST: "main",
            RegionType.UNKNOWN: "main",
        }
        return mapping.get(region_type, "main")

    def _assign_reading_order(
        self,
        blocks: List[Block],
        layout_type: LayoutType,
        page_width: float,
    ) -> List[Block]:
        """Assign reading order using region reading order hints."""
        # Sort blocks by their region's reading order, then by Y position within region
        def sort_key(block: Block):
            region_order = getattr(block, "_region_reading_order", 999)
            return (region_order, block.y0)

        sorted_blocks = sorted(blocks, key=sort_key)

        # Assign sequential reading order
        for i, block in enumerate(sorted_blocks):
            block.reading_order = i

        return sorted_blocks


class AdvancedFusion(ReadingOrderFusion):
    """
    Advanced fusion: Crop and process Florence-2 regions separately.

    NOTE: Stub for future implementation.
    """

    def fuse(
        self,
        ocr_words: List[WordBox],
        florence_result: FlorenceLayoutResult,
    ) -> List[Block]:
        raise NotImplementedError("Advanced fusion not yet implemented")
