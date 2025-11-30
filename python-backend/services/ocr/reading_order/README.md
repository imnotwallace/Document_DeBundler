# Reading Order Detection System

A 5-layer architecture for detecting and assigning proper reading order to OCR-detected text.

## Quick Start

```python
from services.ocr.reading_order import process_paddleocr_result, ReadingOrderConfig

# For clean scanned documents (default settings)
config = ReadingOrderConfig()

# For photographed/skewed documents
config = ReadingOrderConfig.for_photographed_document()

# Process PaddleOCR results
ordered_text = process_paddleocr_result(
    paddleocr_result,
    page_width,
    page_height,
    config
)
```

## Configuration Presets

### Default (Scanned Documents)
```python
config = ReadingOrderConfig()
# or
config = ReadingOrderConfig.for_scanned_document()
```
Best for: High-quality scans, digital PDFs, well-aligned documents.

### Photographed Documents
```python
config = ReadingOrderConfig.for_photographed_document()
```
Best for: Phone photos of documents, angled captures, documents with skew/perspective distortion.

Features:
- Automatic skew detection and compensation
- Higher tolerance for vertical variation
- More lenient baseline matching

### Multi-Column Documents
```python
config = ReadingOrderConfig.for_multi_column()
```
Best for: Newspapers, magazines, academic papers with columns.

### Dense Text
```python
config = ReadingOrderConfig.for_dense_text()
```
Best for: Legal documents, contracts, fine print with tight line spacing.

## Architecture

### Layer 1: Input Normalization
- Converts OCR results to WordBox objects
- Filters by confidence threshold
- Removes invalid/oversized boxes

### Layer 2: Structural Grouping
- **Line Formation**: Groups words into horizontal lines
  - Uses vertical overlap and baseline alignment
  - Skew-aware for photographed documents
  - Slope detection for tilted text
- **Block Formation**: Groups lines into paragraphs

### Layer 3: Region Detection
- Detects overall layout type (single-column, multi-column, table)
- Column detection using projection histograms
- Identifies headers and footers

### Layer 4: Reading Order Assignment
- Assigns sequential order based on layout type
- Handles multi-column reading patterns
- Respects document structure

### Layer 5: Text Output
- Generates final ordered text
- Normalizes whitespace
- Preserves paragraph structure

## Skew Handling

The system automatically detects and compensates for document skew in photographed documents:

1. **Skew Estimation**: Uses linear regression on word positions to estimate tilt angle
2. **Coordinate Compensation**: Adjusts y-coordinates to simulate deskewed document
3. **Slope-Aware Line Detection**: Predicts expected y-position based on line slope

This is particularly important for:
- Phone photos of documents
- Documents photographed at an angle
- Pages with perspective distortion

## Key Parameters

### Line Formation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `line_vertical_overlap_threshold` | 0.5 | Minimum vertical overlap for same line (0-1) |
| `line_baseline_tolerance_multiplier` | 0.3 | Baseline tolerance as multiple of word height |
| `line_horizontal_gap_multiplier` | 3.0 | Max gap between words as multiple of height |
| `line_vertical_threshold_multiplier` | 0.5 | Vertical distance tolerance multiplier |
| `line_y_spread_multiplier` | 0.6 | Max Y-spread within line (for skewed docs) |

### Column Detection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `column_separation_threshold` | 0.15 | Gap fraction of page width for column boundary |
| `valley_density_threshold` | 0.3 | Density threshold for gutter detection |

### Special Zones
| Parameter | Default | Description |
|-----------|---------|-------------|
| `header_zone_ratio` | 0.12 | Top N% of page is header |
| `footer_zone_ratio` | 0.88 | Bottom (1-N)% of page is footer |

## Troubleshooting

### Lines are fragmented (too many short lines)
- Increase `line_y_spread_multiplier` (try 1.5-2.5)
- Increase `line_baseline_tolerance_multiplier`
- Use `ReadingOrderConfig.for_photographed_document()`

### Lines are merged (text from different lines combined)
- Decrease `line_y_spread_multiplier` (try 0.4-0.6)
- Decrease `line_baseline_tolerance_multiplier`
- Use `ReadingOrderConfig.for_dense_text()`

### Column text is interleaved
- Decrease `line_horizontal_gap_multiplier` (try 2.0)
- Decrease `column_separation_threshold` (try 0.10)
- Use `ReadingOrderConfig.for_multi_column()`

### Text order is wrong for skewed documents
The system includes automatic skew detection and compensation. For best results:
- Use `ReadingOrderConfig.for_photographed_document()` preset (uses clustering + deskewing)
- For extreme skew (>3 degrees), consider preprocessing the image to deskew before OCR
- The adaptive clustering analyzes y-coordinate gaps to find optimal line boundaries

**Known limitations**: Documents with both significant skew AND tight line spacing may still have some word order issues, as words at line boundaries can have nearly identical y-coordinates after deskewing

## API Reference

### Main Functions

```python
def process_paddleocr_result(
    paddleocr_result: List[List],
    page_width: float,
    page_height: float,
    config: Optional[ReadingOrderConfig] = None
) -> str:
    """Process PaddleOCR format results into ordered text."""

def process_reading_order(
    ocr_results: List[Dict[str, Any]],
    page_width: float,
    page_height: float,
    config: Optional[ReadingOrderConfig] = None,
    return_structured: bool = False
) -> str | List[Dict[str, Any]]:
    """Main pipeline for reading order detection."""

def process_reading_order_safe(
    ocr_results: List[Dict[str, Any]],
    page_width: float,
    page_height: float,
    config: Optional[ReadingOrderConfig] = None,
    return_structured: bool = False
) -> str | List[Dict[str, Any]]:
    """Safe wrapper with fallback to lexical ordering on error."""
```

### Data Structures

```python
from services.ocr.reading_order import WordBox, Line, Block, Region

# WordBox: Single detected word with bounding box
# Line: Horizontal sequence of words
# Block: Paragraph (group of lines)
# Region: Column or content area
```

## Performance Notes

- Skew estimation adds minimal overhead (~10-20ms for 100+ words)
- Coordinate deskewing is O(n) where n = number of words
- Slope-aware line detection has negligible additional cost
- Overall pipeline typically completes in <100ms for standard documents
