"""
Region Extractor for PP-DocLayout Output.

Converts PP-DocLayout engine output into structured FlorenceLayoutResult
objects for use in the multipass pipeline.
"""

import logging
from typing import List, Dict, Any, Optional

from ..multipass.data_structures import (
    FlorenceRegion,
    FlorenceLayoutResult,
    BoundingBox,
    RegionType,
    LayoutType,
    map_label_to_region_type,
    TEXT_REGION_TYPES,
)

logger = logging.getLogger(__name__)


class RegionExtractor:
    """
    Extracts structured regions from PP-DocLayout output.

    Converts raw DocLayoutEngine.analyze_layout() output into
    FlorenceLayoutResult objects for the multipass pipeline.
    """

    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize region extractor.

        Args:
            min_confidence: Minimum confidence for regions (default 0.5)
        """
        self.min_confidence = min_confidence

    def extract(
        self,
        layout_output: Dict[str, Any],
        page_num: int = 0,
    ) -> FlorenceLayoutResult:
        """
        Extract structured layout result from PP-DocLayout output.

        Args:
            layout_output: Raw output from DocLayoutEngine.analyze_layout()
            page_num: Page number for metadata (overrides output if provided)

        Returns:
            FlorenceLayoutResult with regions and layout type
        """
        # Get dimensions
        width = layout_output.get("image_width", 0)
        height = layout_output.get("image_height", 0)

        # Get page number from output or use provided
        actual_page_num = layout_output.get("page_number", page_num)

        # Extract regions from PP-DocLayout output
        raw_regions = layout_output.get("regions", [])
        regions = []

        for raw_region in raw_regions:
            region = self._convert_region(raw_region, width, height)
            if region and region.confidence >= self.min_confidence:
                regions.append(region)

        # Get layout type (already computed by DocLayoutEngine)
        layout_type_str = layout_output.get("layout_type", "unknown")
        layout_type = self._parse_layout_type(layout_type_str)

        # Assign column indices based on layout
        self._assign_column_indices(regions, width, layout_type)

        # Processing time
        processing_time = layout_output.get("processing_time", 0.0)

        # Model name
        model_name = layout_output.get("model", "PP-DocLayout")

        return FlorenceLayoutResult(
            page_number=actual_page_num,
            layout_type=layout_type,
            overall_caption="",  # PP-DocLayout doesn't provide captions
            regions=regions,
            image_width=width,
            image_height=height,
            processing_time=processing_time,
            model_name=model_name,
        )

    def _convert_region(
        self,
        raw_region: Dict[str, Any],
        page_width: int,
        page_height: int,
    ) -> Optional[FlorenceRegion]:
        """Convert a PP-DocLayout region to FlorenceRegion."""
        # Get bounding box
        bbox_data = raw_region.get("bbox", {})
        if not bbox_data:
            return None

        x0 = float(bbox_data.get("x0", 0))
        y0 = float(bbox_data.get("y0", 0))
        x1 = float(bbox_data.get("x1", 0))
        y1 = float(bbox_data.get("y1", 0))

        # Validate bbox
        if x1 <= x0 or y1 <= y0:
            logger.debug(f"Skipping invalid region bbox: {bbox_data}")
            return None

        # Get label and map to RegionType
        label = raw_region.get("label", "unknown")
        region_type = map_label_to_region_type(label)

        # Get confidence
        confidence = float(raw_region.get("confidence", 0.85))

        # Get reading order hint
        reading_order = raw_region.get("reading_order", raw_region.get("region_id", 0))

        return FlorenceRegion(
            region_id=raw_region.get("region_id", 0),
            region_type=region_type,
            semantic_label=label,
            bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
            confidence=confidence,
            column_index=None,  # Will be assigned later
            reading_order_hint=reading_order,
        )

    def _parse_layout_type(self, layout_str: str) -> LayoutType:
        """Parse layout type string to enum."""
        mapping = {
            "single_column": LayoutType.SINGLE_COLUMN,
            "two_column": LayoutType.TWO_COLUMN,
            "multi_column": LayoutType.MULTI_COLUMN,
            "table": LayoutType.TABLE,
            "complex": LayoutType.COMPLEX,
        }
        return mapping.get(layout_str.lower(), LayoutType.UNKNOWN)

    def _assign_column_indices(
        self,
        regions: List[FlorenceRegion],
        page_width: int,
        layout_type: LayoutType,
    ) -> None:
        """Assign column indices to regions based on layout type."""
        if layout_type == LayoutType.SINGLE_COLUMN:
            for region in regions:
                region.column_index = 0
            return

        # For multi-column layouts, divide page into sections
        if layout_type == LayoutType.TWO_COLUMN:
            num_columns = 2
        elif layout_type == LayoutType.MULTI_COLUMN:
            num_columns = 3
        else:
            num_columns = 1

        if page_width <= 0:
            return

        column_width = page_width / num_columns

        for region in regions:
            center_x = region.bbox.center_x
            region.column_index = min(
                int(center_x / column_width),
                num_columns - 1,
            )

    def extract_text_regions(
        self,
        layout_output: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract only text-containing regions for reading order processing.

        Args:
            layout_output: Raw output from DocLayoutEngine.analyze_layout()

        Returns:
            List of text region dictionaries with bbox and reading order
        """
        raw_regions = layout_output.get("regions", [])
        text_regions = []

        for raw_region in raw_regions:
            # Check if region contains text
            is_text = raw_region.get("is_text", False)
            label = raw_region.get("label", "")

            # Also check by label
            region_type = map_label_to_region_type(label)
            if is_text or region_type in TEXT_REGION_TYPES:
                text_regions.append({
                    "region_id": raw_region.get("region_id", 0),
                    "label": label,
                    "bbox": raw_region.get("bbox", {}),
                    "confidence": raw_region.get("confidence", 0.85),
                    "reading_order": raw_region.get("reading_order", 0),
                    "reading_priority": raw_region.get("reading_priority", 99),
                })

        # Sort by reading order
        text_regions.sort(key=lambda r: r.get("reading_order", 0))

        return text_regions
