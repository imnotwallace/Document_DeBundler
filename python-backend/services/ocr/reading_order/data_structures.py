"""
Core data structures for reading order detection.

Defines the hierarchy: WordBox -> Line -> Block -> Region
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WordBox:
    """
    Represents a single word detected by OCR.

    Coordinates use the same system as the OCR engine (typically pixels).
    """
    # Core OCR data
    text: str
    x0: float          # Left edge
    y0: float          # Top edge
    x1: float          # Right edge
    y1: float          # Bottom edge
    page: int = 0
    confidence: float = 1.0  # OCR confidence (0.0-1.0)

    # Derived properties (calculated once, cached)
    width: float = field(init=False)
    height: float = field(init=False)
    center_x: float = field(init=False)
    center_y: float = field(init=False)
    baseline: float = field(init=False)  # Approximated as y1

    # Assigned during processing
    line_id: Optional[int] = None
    block_id: Optional[int] = None
    region_id: Optional[int] = None
    reading_order: Optional[int] = None
    original_index: Optional[int] = None  # For tracking original position during reordering

    def __post_init__(self):
        """Calculate derived properties after initialization."""
        self.width = self.x1 - self.x0
        self.height = self.y1 - self.y0
        self.center_x = (self.x0 + self.x1) / 2.0
        self.center_y = (self.y0 + self.y1) / 2.0
        self.baseline = self.y1  # Simplified baseline approximation

    @classmethod
    def from_ocr_result(cls, text: str, bbox: list, page: int = 0, confidence: float = 1.0,
                        center_y_override: float = None):
        """
        Create WordBox from OCR bounding box.

        Args:
            text: Detected text
            bbox: Bounding box in format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                  or [x0, y0, x1, y1]
            page: Page number
            confidence: OCR confidence score
            center_y_override: Optional pre-calculated center_y (for curved text accuracy)

        Returns:
            WordBox instance
        """
        calculated_center_y = None

        if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
            # Format: [x0, y0, x1, y1]
            x0, y0, x1, y1 = bbox
        else:
            # Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # For curved text, calculate center_y from polygon midpoints
            # This is more accurate than using min/max bounding box center
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x0 = min(x_coords)
            y0 = min(y_coords)
            x1 = max(x_coords)
            y1 = max(y_coords)

            # Calculate true center_y from polygon edge midpoints
            # bbox format: [top-left, top-right, bottom-right, bottom-left]
            # Left edge mid-y = (top-left.y + bottom-left.y) / 2
            # Right edge mid-y = (top-right.y + bottom-right.y) / 2
            if len(bbox) == 4:
                left_mid_y = (bbox[0][1] + bbox[3][1]) / 2
                right_mid_y = (bbox[1][1] + bbox[2][1]) / 2
                calculated_center_y = (left_mid_y + right_mid_y) / 2

        word_box = cls(
            text=text,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            page=page,
            confidence=confidence
        )

        # Override center_y if provided or calculated from polygon
        if center_y_override is not None:
            word_box.center_y = center_y_override
        elif calculated_center_y is not None:
            word_box.center_y = calculated_center_y

        return word_box


@dataclass
class Line:
    """
    Represents a horizontal line of text (multiple words).
    """
    words: List[WordBox] = field(default_factory=list)
    line_id: int = 0

    # Bounding box (calculated from words)
    x0: float = field(init=False)
    y0: float = field(init=False)
    x1: float = field(init=False)
    y1: float = field(init=False)

    # Derived properties
    width: float = field(init=False)
    height: float = field(init=False)
    center_x: float = field(init=False)
    center_y: float = field(init=False)

    # Assigned during processing
    block_id: Optional[int] = None
    reading_order: Optional[int] = None

    def __post_init__(self):
        """Calculate bounding box and derived properties from words."""
        if self.words:
            self.x0 = min(word.x0 for word in self.words)
            self.y0 = min(word.y0 for word in self.words)
            self.x1 = max(word.x1 for word in self.words)
            self.y1 = max(word.y1 for word in self.words)
        else:
            self.x0 = self.y0 = self.x1 = self.y1 = 0.0

        self.width = self.x1 - self.x0
        self.height = self.y1 - self.y0
        self.center_x = (self.x0 + self.x1) / 2.0
        self.center_y = (self.y0 + self.y1) / 2.0

    def get_text(self) -> str:
        """Get concatenated text from all words."""
        return " ".join(word.text for word in self.words)

    def sort_words_left_to_right(self):
        """Sort words left to right."""
        self.words.sort(key=lambda w: w.x0)
        # Recalculate bounding box
        self.__post_init__()


@dataclass
class Block:
    """
    Represents a block of text (multiple lines).
    Typically a paragraph or section.
    """
    lines: List[Line] = field(default_factory=list)
    block_id: int = 0
    block_type: str = "main"  # "main", "header", "footer", "sidebar", "table"

    # Bounding box (calculated from lines)
    x0: float = field(init=False)
    y0: float = field(init=False)
    x1: float = field(init=False)
    y1: float = field(init=False)

    # Derived properties
    width: float = field(init=False)
    height: float = field(init=False)
    center_x: float = field(init=False)
    center_y: float = field(init=False)

    # Assigned during processing
    region_id: Optional[int] = None
    reading_order: Optional[int] = None

    # Florence-2 layout analysis fields
    florence_region_id: Optional[int] = None
    florence_region_type: Optional[str] = None  # "header", "footer", "column", "table", "figure", "text_region"
    semantic_label: Optional[str] = None  # Florence caption/description

    def __post_init__(self):
        """Calculate bounding box and derived properties from lines."""
        if self.lines:
            self.x0 = min(line.x0 for line in self.lines)
            self.y0 = min(line.y0 for line in self.lines)
            self.x1 = max(line.x1 for line in self.lines)
            self.y1 = max(line.y1 for line in self.lines)
        else:
            self.x0 = self.y0 = self.x1 = self.y1 = 0.0

        self.width = self.x1 - self.x0
        self.height = self.y1 - self.y0
        self.center_x = (self.x0 + self.x1) / 2.0
        self.center_y = (self.y0 + self.y1) / 2.0

    def get_text(self) -> str:
        """Get concatenated text from all lines."""
        return "\n".join(line.get_text() for line in self.lines)

    def sort_lines_top_to_bottom(self):
        """Sort lines top to bottom."""
        self.lines.sort(key=lambda l: l.y0)
        # Recalculate bounding box
        self.__post_init__()


@dataclass
class Region:
    """
    Represents a region (column) containing multiple blocks.
    """
    blocks: List[Block] = field(default_factory=list)
    region_id: int = 0
    region_type: str = "column"  # "column", "header", "footer", "table"

    # Horizontal boundaries
    x0: float = 0.0
    x1: float = 0.0
    center_x: float = field(init=False)

    # Assigned during processing
    reading_order: Optional[int] = None

    def __post_init__(self):
        """Calculate center from boundaries."""
        self.center_x = (self.x0 + self.x1) / 2.0

    def get_text(self) -> str:
        """Get concatenated text from all blocks."""
        return "\n\n".join(block.get_text() for block in self.blocks)

    def sort_blocks_top_to_bottom(self):
        """Sort blocks top to bottom."""
        self.blocks.sort(key=lambda b: b.y0)
