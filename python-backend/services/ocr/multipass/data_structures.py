"""
Data structures for the multi-pass OCR pipeline.

Defines Florence-2 region types, checkpoint structures, and pipeline state.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import json
from pathlib import Path


class PassStatus(Enum):
    """Status of a pipeline pass."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RegionType(Enum):
    """Standardized region types from layout analysis (PP-DocLayout)."""
    # Document structure
    DOC_TITLE = "doc_title"
    PARAGRAPH_TITLE = "paragraph_title"
    HEADER = "header"
    FOOTER = "footer"

    # Text content
    TEXT = "text"
    CONTENT = "content"
    ABSTRACT = "abstract"
    SIDEBAR_TEXT = "sidebar_text"
    FOOTNOTE = "footnote"
    REFERENCE = "reference"

    # Visual elements
    IMAGE = "image"
    FIGURE_TITLE = "figure_title"
    TABLE = "table"
    TABLE_TITLE = "table_title"
    CHART = "chart"
    CHART_TITLE = "chart_title"

    # Technical content
    FORMULA = "formula"
    FORMULA_NUMBER = "formula_number"
    ALGORITHM = "algorithm"

    # Decorative/Other
    HEADER_IMAGE = "header_image"
    FOOTER_IMAGE = "footer_image"
    SEAL = "seal"
    NUMBER = "number"

    # Legacy/fallback
    COLUMN = "column"
    SIDEBAR = "sidebar"
    FIGURE = "figure"
    SIGNATURE = "signature"
    TEXT_REGION = "text_region"
    CAPTION = "caption"
    LIST = "list"
    UNKNOWN = "unknown"


class LayoutType(Enum):
    """Document layout classifications."""
    SINGLE_COLUMN = "single_column"
    TWO_COLUMN = "two_column"
    MULTI_COLUMN = "multi_column"
    TABLE = "table"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Axis-aligned bounding box with utility methods."""
    x0: float  # Left edge
    y0: float  # Top edge
    x1: float  # Right edge
    y1: float  # Bottom edge

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def area(self) -> float:
        return self.width * self.height

    def overlap_ratio(self, other: "BoundingBox") -> float:
        """Calculate overlap ratio (intersection / self area)."""
        x_left = max(self.x0, other.x0)
        y_top = max(self.y0, other.y0)
        x_right = min(self.x1, other.x1)
        y_bottom = min(self.y1, other.y1)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        return intersection / self.area if self.area > 0 else 0.0

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside bounding box."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "BoundingBox":
        """Deserialize from dictionary."""
        return cls(x0=data["x0"], y0=data["y0"], x1=data["x1"], y1=data["y1"])

    @classmethod
    def from_quad(cls, quad: List[List[float]]) -> "BoundingBox":
        """Create from quadrilateral [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]."""
        x_coords = [pt[0] for pt in quad]
        y_coords = [pt[1] for pt in quad]
        return cls(
            x0=min(x_coords),
            y0=min(y_coords),
            x1=max(x_coords),
            y1=max(y_coords)
        )



# PP-DocLayout label to RegionType mapping
DOCLAYOUT_TO_REGION_TYPE = {
    "doc_title": RegionType.DOC_TITLE,
    "paragraph_title": RegionType.PARAGRAPH_TITLE,
    "header": RegionType.HEADER,
    "footer": RegionType.FOOTER,
    "text": RegionType.TEXT,
    "content": RegionType.CONTENT,
    "abstract": RegionType.ABSTRACT,
    "sidebar_text": RegionType.SIDEBAR_TEXT,
    "footnote": RegionType.FOOTNOTE,
    "reference": RegionType.REFERENCE,
    "image": RegionType.IMAGE,
    "figure_title": RegionType.FIGURE_TITLE,
    "table": RegionType.TABLE,
    "table_title": RegionType.TABLE_TITLE,
    "chart": RegionType.CHART,
    "chart_title": RegionType.CHART_TITLE,
    "formula": RegionType.FORMULA,
    "formula_number": RegionType.FORMULA_NUMBER,
    "algorithm": RegionType.ALGORITHM,
    "header_image": RegionType.HEADER_IMAGE,
    "footer_image": RegionType.FOOTER_IMAGE,
    "seal": RegionType.SEAL,
    "number": RegionType.NUMBER,
}


def map_label_to_region_type(label: str) -> RegionType:
    """Map PP-DocLayout label string to RegionType enum."""
    return DOCLAYOUT_TO_REGION_TYPE.get(label.lower(), RegionType.UNKNOWN)


# Categories that contain text content (for reading order)
TEXT_REGION_TYPES = {
    RegionType.DOC_TITLE,
    RegionType.PARAGRAPH_TITLE,
    RegionType.TEXT,
    RegionType.CONTENT,
    RegionType.ABSTRACT,
    RegionType.SIDEBAR_TEXT,
    RegionType.FOOTNOTE,
    RegionType.REFERENCE,
    RegionType.TABLE_TITLE,
    RegionType.FIGURE_TITLE,
    RegionType.CHART_TITLE,
    RegionType.HEADER,
    RegionType.FOOTER,
    RegionType.ALGORITHM,
    RegionType.FORMULA,
}


@dataclass
class FlorenceRegion:
    """A region detected by Florence-2 layout analysis."""
    region_id: int
    region_type: RegionType
    semantic_label: str  # Florence caption/description for the region
    bbox: BoundingBox
    confidence: float = 0.9
    column_index: Optional[int] = None
    reading_order_hint: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "region_id": self.region_id,
            "region_type": self.region_type.value,
            "semantic_label": self.semantic_label,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "column_index": self.column_index,
            "reading_order_hint": self.reading_order_hint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlorenceRegion":
        """Deserialize from dictionary."""
        return cls(
            region_id=data["region_id"],
            region_type=RegionType(data["region_type"]),
            semantic_label=data["semantic_label"],
            bbox=BoundingBox.from_dict(data["bbox"]),
            confidence=data.get("confidence", 0.9),
            column_index=data.get("column_index"),
            reading_order_hint=data.get("reading_order_hint"),
        )


@dataclass
class FlorenceLayoutResult:
    """Complete layout analysis result for a page (supports PP-DocLayout and Florence-2)."""
    page_number: int
    layout_type: LayoutType
    overall_caption: str
    regions: List[FlorenceRegion]
    image_width: int
    image_height: int
    processing_time: float = 0.0
    model_name: str = "PP-DocLayout"  # Model that produced this result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "schema_version": "1.0",
            "page_number": self.page_number,
            "layout_type": self.layout_type.value,
            "overall_caption": self.overall_caption,
            "regions": [r.to_dict() for r in self.regions],
            "image_dimensions": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "metadata": {
                "processing_time_seconds": self.processing_time,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model": getattr(self, 'model_name', 'PP-DocLayout'),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlorenceLayoutResult":
        """Deserialize from dictionary."""
        return cls(
            page_number=data["page_number"],
            layout_type=LayoutType(data["layout_type"]),
            overall_caption=data["overall_caption"],
            regions=[FlorenceRegion.from_dict(r) for r in data["regions"]],
            image_width=data["image_dimensions"]["width"],
            image_height=data["image_dimensions"]["height"],
            processing_time=data.get("metadata", {}).get("processing_time_seconds", 0.0),
        )

    def save_to_file(self, path: Path) -> None:
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, path: Path) -> "FlorenceLayoutResult":
        """Load from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class OCRPageResult:
    """OCR result for a single page (from PaddleOCR)."""
    page_number: int
    image_width: int
    image_height: int
    word_boxes: List[Dict[str, Any]]  # List of {text, bbox, confidence}
    processing_time: float = 0.0
    gpu_used: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "schema_version": "1.0",
            "page_number": self.page_number,
            "image_dimensions": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "word_boxes": self.word_boxes,
            "metadata": {
                "model": "PaddleOCR",
                "processing_time_seconds": self.processing_time,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "gpu_used": self.gpu_used,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRPageResult":
        """Deserialize from dictionary."""
        return cls(
            page_number=data["page_number"],
            image_width=data["image_dimensions"]["width"],
            image_height=data["image_dimensions"]["height"],
            word_boxes=data["word_boxes"],
            processing_time=data.get("metadata", {}).get("processing_time_seconds", 0.0),
            gpu_used=data.get("metadata", {}).get("gpu_used", True),
        )

    def save_to_file(self, path: Path) -> None:
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, path: Path) -> "OCRPageResult":
        """Load from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class PageCheckpoint:
    """Per-page checkpoint for recovery."""
    page_num: int
    pass_statuses: Dict[str, str] = field(default_factory=dict)
    ocr_result_path: Optional[str] = None
    florence_result_path: Optional[str] = None
    reading_order_path: Optional[str] = None
    pdf_page_path: Optional[str] = None
    error: Optional[str] = None
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "page_num": self.page_num,
            "pass_statuses": self.pass_statuses,
            "ocr_result_path": self.ocr_result_path,
            "florence_result_path": self.florence_result_path,
            "reading_order_path": self.reading_order_path,
            "pdf_page_path": self.pdf_page_path,
            "error": self.error,
            "last_updated": self.last_updated or datetime.utcnow().isoformat() + "Z",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageCheckpoint":
        """Deserialize from dictionary."""
        return cls(
            page_num=data["page_num"],
            pass_statuses=data.get("pass_statuses", {}),
            ocr_result_path=data.get("ocr_result_path"),
            florence_result_path=data.get("florence_result_path"),
            reading_order_path=data.get("reading_order_path"),
            pdf_page_path=data.get("pdf_page_path"),
            error=data.get("error"),
            last_updated=data.get("last_updated"),
        )


@dataclass
class PipelineState:
    """State passed between pipeline passes."""
    doc_id: str
    workspace_dir: Path
    pdf_path: Path
    output_path: Path
    total_pages: int
    image_paths: List[Path] = field(default_factory=list)
    current_pass: int = 0
    pass_statuses: Dict[str, PassStatus] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def get_temp_dir(self, subdir: str) -> Path:
        """Get a temp subdirectory path."""
        path = self.workspace_dir / "temp" / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def ocr_results_dir(self) -> Path:
        return self.get_temp_dir("ocr_results")

    @property
    def layout_results_dir(self) -> Path:
        return self.get_temp_dir("layout_results")

    @property
    def reading_order_dir(self) -> Path:
        return self.get_temp_dir("reading_order")

    @property
    def pdf_pages_dir(self) -> Path:
        return self.get_temp_dir("pdf_pages")

    @property
    def checkpoint_path(self) -> Path:
        return self.workspace_dir / "temp" / "checkpoint.json"
