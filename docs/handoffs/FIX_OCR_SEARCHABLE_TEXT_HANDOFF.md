# FIX OCR SEARCHABLE TEXT HANDOFF

**Created**: 2025-11-15
**Status**: Ready for Implementation
**Priority**: CRITICAL
**Estimated Effort**: 2-3 hours

---

## EXECUTIVE SUMMARY

The OCR pipeline successfully extracts text using PaddleOCR but **produces completely unsearchable PDFs**. All four text insertion points use `insert_textbox(..., render_mode=3)`, which **does not create searchable text in PyMuPDF**.

**Impact**: 100% of OCR'd PDFs are unsearchable, defeating the primary purpose of OCR.

**Root Cause**: Wrong PyMuPDF API - must use `insert_text()` or `TextWriter`, not `insert_textbox()`.

**Solution**: Replace all `insert_textbox()` calls with proper `insert_text()` or `TextWriter.write_text()` with `render_mode=3`.

---

## TABLE OF CONTENTS

1. [Problem Statement](#1-problem-statement)
2. [Root Cause Analysis](#2-root-cause-analysis)
3. [PyMuPDF API Research](#3-pymupdf-api-research)
4. [Proposed Solution Architecture](#4-proposed-solution-architecture)
5. [Implementation Plan](#5-implementation-plan)
6. [Code Changes - All Four Insertion Points](#6-code-changes---all-four-insertion-points)
7. [Testing & Verification](#7-testing--verification)
8. [Edge Cases & Error Handling](#8-edge-cases--error-handling)
9. [Rollback Plan](#9-rollback-plan)

---

## 1. PROBLEM STATEMENT

### Current Behavior

**File**: `python-backend/services/ocr_batch_service.py`

**Symptom**: Output PDFs contain no searchable text despite successful OCR processing.

**Diagnostic Evidence**:
```
Test shows: "Text extracted: 0 characters" from output PDF
OCR runs successfully (text is extracted from source images)
Output PDF is created but completely unsearchable
```

### Affected Code Locations

All four text insertion points use the same broken pattern:

1. **Phase 4** (Line ~1920): Text overlay for pages with existing text layer
2. **`_add_text_overlay()`** (Line ~1010): Overlay strategy for OCR'd pages
3. **`_full_rebuild_with_compression()`** (Line ~1125): Full rebuild strategy
4. **`_insert_positioned_text()`** (Line ~1230): Coordinate-mapped text with bounding boxes

### Current Broken Code Pattern

```python
page.insert_textbox(
    page_rect,
    text,
    fontsize=8,
    color=(1, 1, 1),  # White (invisible on white background)
    overlay=True,
    render_mode=3  # DOES NOT WORK - creates no searchable text!
)
```

---

## 2. ROOT CAUSE ANALYSIS

### Why `insert_textbox()` Doesn't Work

**PyMuPDF Maintainer Statement** (GitHub Discussion #2464):
> "In any case do not use `insert_textbox()`, but `insert_text()`"

**Technical Explanation**:

1. **`insert_textbox()`** is designed for **visible text layout** in a rectangular area
   - Supports text wrapping, alignment, and formatting
   - `render_mode` parameter is **not reliably honored** for searchability
   - Primary use case: visible text boxes, forms, annotations

2. **`insert_text()`** is designed for **precise text positioning**
   - Places text at exact coordinates with exact font size
   - `render_mode=3` creates **invisible but searchable text**
   - Primary use case: OCR text overlay, text replacement

3. **`TextWriter`** is the **high-level API** for batch text insertion
   - Accumulates text with positions, then writes all at once
   - Supports `render_mode=3` in `write_text()` method
   - Best for complex multi-line text insertion

### Why Current Code Appears to Work

The code doesn't fail or raise errors because:
- PyMuPDF accepts the parameters without complaint
- Text insertion "succeeds" (returns positive values)
- PDF is created without errors

But the text is **not searchable** because `insert_textbox()` doesn't properly implement searchable text with `render_mode=3`.

---

## 3. PYMUPDF API RESEARCH

### Correct APIs for Searchable Text

#### Option 1: `insert_text()` - Simple, Direct

**Signature**:
```python
page.insert_text(
    point,           # fitz.Point - insertion point (baseline-left)
    text,            # str - text to insert
    fontsize=11,     # float - font size in points
    fontname="helv", # str - font name
    fontfile=None,   # str - path to font file
    set_simple=0,    # int - font embedding mode
    encoding=0,      # int - font encoding
    color=None,      # tuple - (r, g, b) in range 0-1
    fill=None,       # tuple - fill color
    render_mode=0,   # int - 0=fill, 1=stroke, 2=fill+stroke, 3=INVISIBLE
    border_width=1,  # float - stroke width
    rotate=0,        # int - rotation angle
    morph=None,      # tuple - transformation
    overlay=True     # bool - True=foreground, False=background
)
```

**Key Parameters for OCR**:
- `render_mode=3`: **INVISIBLE TEXT** (searchable but not visible)
- `overlay=True`: Place in foreground (standard for OCR overlay)
- `fontsize`: Should match OCR bounding box height for accuracy
- `point`: Baseline-left corner of text (NOT top-left!)

**Insertion Point Formula**:
```python
# For text without descenders (most text)
insertion_point = bbox.bl  # Bottom-left

# For text with descenders (g, y, j, p, q)
font = fitz.Font("helv")
insertion_point = bbox.bl + (0, font.descender * fontsize)
```

#### Option 2: `TextWriter` - Batch Processing

**Workflow**:
```python
# 1. Create TextWriter for page
tw = fitz.TextWriter(page.rect, opacity=1, color=None)

# 2. Accumulate text at positions
tw.append(
    pos,         # fitz.Point or tuple (x, y) - insertion point
    text,        # str - text to insert
    font=None,   # fitz.Font object
    fontsize=11, # float - font size
    language=None # str - language code
)

# 3. Write all text at once
tw.write_text(
    page,
    opacity=None,   # float - override opacity
    color=None,     # tuple - override color
    morph=None,     # tuple - transformation
    overlay=True,   # bool - foreground/background
    oc=0,           # int - optional content
    render_mode=3   # int - 3=INVISIBLE
)
```

**Advantages**:
- Batch text insertion (more efficient for multiple lines)
- Single `write_text()` call applies `render_mode=3` to all text
- Better performance for pages with many text regions

**Disadvantages**:
- More complex API (two-step process)
- Must manage font objects explicitly

---

## 4. PROPOSED SOLUTION ARCHITECTURE

### Strategy Selection by Use Case

| Use Case | Method | Rationale |
|----------|--------|-----------|
| **WITH bounding boxes** (coordinate mapping) | `insert_text()` | Precise positioning, per-line font sizing |
| **WITHOUT bounding boxes** (fallback) | `TextWriter` | Better multi-line handling, single write operation |
| **Phase 4 text layer pages** | `TextWriter` | Text-only overlay, no coordinates needed |

### High-Level Architecture

```
OCR Result
    |
    v
Has Bounding Boxes?
    |
    +-- YES --> Use insert_text() with coordinate mapping
    |           - Calculate font size from bbox height
    |           - Position at bbox.bl (bottom-left)
    |           - Set render_mode=3 (invisible)
    |
    +-- NO  --> Use TextWriter for page-wide text
                - Split text into lines
                - Distribute across page height
                - Single write_text(render_mode=3)
```

### Font Size Calculation

**For Bounding Box Text**:
```python
# Calculate font size to fit bbox height
font = fitz.Font("helv")
bbox_height = pdf_bbox.height
fontsize = bbox_height * 0.9  # 90% of bbox height (prevents overflow)
```

**For Full-Page Text** (no bounding boxes):
```python
# Use small, uniform font size
fontsize = 8  # Small enough to be unobtrusive
```

---

## 5. IMPLEMENTATION PLAN

### Implementation Checklist

- [ ] **Step 1**: Add helper function `_insert_text_with_bbox()` for coordinate-mapped text
- [ ] **Step 2**: Add helper function `_insert_text_fallback()` for text without coordinates
- [ ] **Step 3**: Update `_insert_positioned_text()` to use `insert_text()`
- [ ] **Step 4**: Update `_add_text_overlay()` to use new helper functions
- [ ] **Step 5**: Update `_full_rebuild_with_compression()` to use new helper functions
- [ ] **Step 6**: Update Phase 4 (text layer pages) to use `TextWriter`
- [ ] **Step 7**: Add logging to track which method is used per page
- [ ] **Step 8**: Test with PDFs containing various text types
- [ ] **Step 9**: Verify searchability with PDF reader text search
- [ ] **Step 10**: Update documentation and tests

### Estimated Time Breakdown

| Task | Time | Risk |
|------|------|------|
| Write helper functions | 30 min | Low |
| Update 4 insertion points | 45 min | Low |
| Add comprehensive logging | 15 min | Low |
| Test with real PDFs | 30 min | Medium |
| Fix edge cases | 30 min | Medium |
| **TOTAL** | **2.5 hours** | **Low-Medium** |

---

## 6. CODE CHANGES - ALL FOUR INSERTION POINTS

### Helper Function 1: Coordinate-Mapped Text Insertion

**Location**: Add to `ocr_batch_service.py` after `_clean_rebuild_page_with_ocr()`

```python
def _insert_text_with_bbox(
    self,
    page: fitz.Page,
    text_line: str,
    pdf_bbox: 'PDFBBox',  # Custom bbox class from coordinate_mapper
    page_rect: fitz.Rect
) -> bool:
    """
    Insert text at precise position using OCR bounding box.

    Args:
        page: PyMuPDF page
        text_line: Text to insert
        pdf_bbox: PDF coordinates from coordinate mapper
        page_rect: Page rectangle (for bounds checking)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Calculate font size from bbox height
        bbox_height = pdf_bbox.height
        fontsize = max(1.0, bbox_height * 0.9)  # 90% of height, minimum 1pt

        # Get font for descender calculation
        font = fitz.Font("helv")

        # Calculate insertion point (baseline-left)
        # Start at bottom-left, adjust for descenders
        insertion_point = fitz.Point(pdf_bbox.x0, pdf_bbox.y1)

        # Adjust for descenders (descender is negative)
        insertion_point = insertion_point + (0, font.descender * fontsize)

        # Bounds check - ensure point is within page
        if not page_rect.contains(insertion_point):
            logger.warning(
                f"Insertion point {insertion_point} outside page rect {page_rect}, "
                f"clamping to page bounds"
            )
            insertion_point.x = max(page_rect.x0, min(insertion_point.x, page_rect.x1))
            insertion_point.y = max(page_rect.y0, min(insertion_point.y, page_rect.y1))

        # Insert text with invisible render mode
        page.insert_text(
            insertion_point,
            text_line.strip(),
            fontsize=fontsize,
            fontname="helv",
            color=(1, 1, 1),  # White (invisible on white background)
            render_mode=3,    # INVISIBLE - searchable but not visible
            overlay=True
        )

        logger.debug(
            f"Inserted text at ({insertion_point.x:.1f}, {insertion_point.y:.1f}), "
            f"fontsize={fontsize:.1f}, bbox_height={bbox_height:.1f}"
        )
        return True

    except Exception as e:
        logger.warning(f"Failed to insert text with bbox: {e}")
        return False


def _insert_text_fallback(
    self,
    page: fitz.Page,
    text: str,
    page_rect: fitz.Rect
) -> bool:
    """
    Insert text without coordinate mapping using TextWriter.

    Distributes text lines across page height for better searchability.

    Args:
        page: PyMuPDF page
        text: Full text to insert
        page_rect: Page rectangle

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create TextWriter for this page
        tw = fitz.TextWriter(page_rect)

        # Split text into lines
        lines = text.split('\n')
        if not lines:
            return True  # Empty text, nothing to do

        # Calculate line spacing
        margin = 10  # pixels from edge
        available_height = page_rect.height - (2 * margin)
        line_spacing = available_height / max(len(lines), 1)

        # Font size
        fontsize = 8  # Small, unobtrusive
        font = fitz.Font("helv")

        # Insert each line
        for i, line in enumerate(lines):
            if not line.strip():
                continue  # Skip empty lines

            # Calculate position (left margin, distributed vertically)
            x = page_rect.x0 + margin
            y = page_rect.y0 + margin + (i * line_spacing)

            # Adjust for baseline (descender)
            y = y + (fontsize * 0.8)  # Approximate baseline adjustment

            # Add to TextWriter
            tw.append(
                (x, y),
                line.strip(),
                font=font,
                fontsize=fontsize
            )

        # Write all text at once with invisible render mode
        tw.write_text(
            page,
            render_mode=3,  # INVISIBLE
            overlay=True
        )

        logger.debug(f"Inserted {len(lines)} lines using TextWriter fallback")
        return True

    except Exception as e:
        logger.error(f"TextWriter fallback failed: {e}", exc_info=True)
        return False
```

---

### Point 1: `_insert_positioned_text()` (Line ~1160)

**CURRENT CODE** (BROKEN):
```python
def _insert_positioned_text(
    self,
    page: fitz.Page,
    page_rect: fitz.Rect,
    ocr_text: str,
    ocr_result: object
) -> None:
    """Insert text at precise locations using OCR bounding boxes."""
    try:
        from .ocr.coordinate_mapper import CoordinateMapper

        mapper = CoordinateMapper()

        # Extract text lines and bounding boxes
        if hasattr(ocr_result, 'raw_result') and ocr_result.raw_result:
            text_lines, bboxes = self._extract_text_and_boxes_from_ocr(ocr_result.raw_result)
        elif isinstance(ocr_result.bbox, list) and ocr_result.bbox:
            text_lines = [ocr_text]
            bboxes = [ocr_result.bbox]
        else:
            # Fallback
            logger.warning(f"No bounding boxes available, falling back to single textbox")
            page.insert_textbox(
                page_rect,
                ocr_text,
                fontsize=8,
                color=(1, 1, 1),
                overlay=True,
                render_mode=3  # DOES NOT WORK!
            )
            return

        # Get image dimensions
        dpi = 300
        if hasattr(ocr_result, 'image_width') and ocr_result.image_width:
            image_width = ocr_result.image_width
            image_height = ocr_result.image_height
        else:
            image_width = int(page_rect.width * dpi / 72)
            image_height = int(page_rect.height * dpi / 72)

        # Insert each text line
        for text_line, bbox in zip(text_lines, bboxes):
            if not text_line or not text_line.strip():
                continue

            # Convert bbox to PDF coordinates
            pdf_bbox = mapper.image_to_pdf_coords(
                bbox=bbox,
                image_width=image_width,
                image_height=image_height,
                page_rect=page_rect,
                image_dpi=dpi
            )

            # Calculate font size
            font_size = mapper.calculate_font_size_from_pdf_bbox(pdf_bbox)

            # BROKEN: insert_textbox doesn't create searchable text!
            try:
                page.insert_textbox(
                    pdf_bbox.to_fitz_rect(),
                    text_line.strip(),
                    fontsize=font_size,
                    color=(1, 1, 1),
                    overlay=True,
                    render_mode=3,  # DOES NOT WORK!
                    align=0
                )
            except Exception as e:
                logger.debug(f"Failed to insert text at position {pdf_bbox}: {e}")

    except Exception as e:
        logger.warning(f"Coordinate mapping failed: {e}, falling back")
        # BROKEN fallback
        page.insert_textbox(page_rect, ocr_text, fontsize=8,
                          color=(1, 1, 1), overlay=True, render_mode=3)
```

**NEW CODE** (FIXED):
```python
def _insert_positioned_text(
    self,
    page: fitz.Page,
    page_rect: fitz.Rect,
    ocr_text: str,
    ocr_result: object
) -> None:
    """Insert text at precise locations using OCR bounding boxes."""
    try:
        from .ocr.coordinate_mapper import CoordinateMapper

        mapper = CoordinateMapper()

        # Extract text lines and bounding boxes
        if hasattr(ocr_result, 'raw_result') and ocr_result.raw_result:
            text_lines, bboxes = self._extract_text_and_boxes_from_ocr(ocr_result.raw_result)
        elif isinstance(ocr_result.bbox, list) and ocr_result.bbox:
            text_lines = [ocr_text]
            bboxes = [ocr_result.bbox]
        else:
            # FIXED: Use TextWriter fallback
            logger.warning(f"No bounding boxes available, using TextWriter fallback")
            success = self._insert_text_fallback(page, ocr_text, page_rect)
            if not success:
                logger.error("TextWriter fallback failed!")
            return

        # Get image dimensions
        dpi = 300
        if hasattr(ocr_result, 'image_width') and ocr_result.image_width:
            image_width = ocr_result.image_width
            image_height = ocr_result.image_height
        else:
            image_width = int(page_rect.width * dpi / 72)
            image_height = int(page_rect.height * dpi / 72)

        # Insert each text line with insert_text()
        inserted_count = 0
        failed_count = 0

        for text_line, bbox in zip(text_lines, bboxes):
            if not text_line or not text_line.strip():
                continue

            # Convert bbox to PDF coordinates
            pdf_bbox = mapper.image_to_pdf_coords(
                bbox=bbox,
                image_width=image_width,
                image_height=image_height,
                page_rect=page_rect,
                image_dpi=dpi
            )

            # FIXED: Use insert_text() with bbox
            success = self._insert_text_with_bbox(page, text_line, pdf_bbox, page_rect)
            if success:
                inserted_count += 1
            else:
                failed_count += 1

        logger.info(
            f"Coordinate mapping: {inserted_count} positioned text elements inserted, "
            f"{failed_count} failed"
        )

    except Exception as e:
        logger.warning(f"Coordinate mapping failed: {e}, using TextWriter fallback")
        # FIXED: Use TextWriter fallback
        success = self._insert_text_fallback(page, ocr_text, page_rect)
        if not success:
            logger.error("TextWriter fallback ALSO failed!")
```

**CHANGES EXPLAINED**:
1. Replaced `insert_textbox()` with `_insert_text_with_bbox()` helper
2. Uses `insert_text()` with proper baseline positioning
3. Fallback uses `TextWriter` instead of broken `insert_textbox()`
4. Added success tracking and better logging

---

### Point 2: `_add_text_overlay()` (Line ~990)

**CURRENT CODE** (BROKEN):
```python
def _add_text_overlay(
    self,
    page: fitz.Page,
    page_rect: fitz.Rect,
    ocr_text: str,
    ocr_result: Optional[object] = None
) -> None:
    """Add invisible text overlay to existing page without re-rendering."""

    # Check if we can use coordinate mapping
    if self.config.use_coordinate_mapping and ocr_result and hasattr(ocr_result, 'bbox') and ocr_result.bbox:
        # Use positioned text with coordinate mapping
        logger.info(f"Using coordinate mapping to insert {len(ocr_result.bbox)} bounding boxes")
        self._insert_positioned_text(page, page_rect, ocr_text, ocr_result)
    else:
        # Fallback: Single textbox covering entire page
        logger.warning("Using fallback single textbox")

        try:
            result = page.insert_textbox(
                page_rect,
                ocr_text,
                fontsize=8,
                color=(1, 1, 1),  # White (invisible)
                overlay=True,
                render_mode=3  # DOES NOT WORK!
            )

            if result > 0:
                logger.info(f"SUCCESS - Fallback textbox inserted ({result} chars fit)")
            else:
                logger.warning(f"WARNING - insert_textbox returned {result}")

        except Exception as e:
            logger.error(f"FAILED to insert fallback textbox: {e}", exc_info=True)
            raise
```

**NEW CODE** (FIXED):
```python
def _add_text_overlay(
    self,
    page: fitz.Page,
    page_rect: fitz.Rect,
    ocr_text: str,
    ocr_result: Optional[object] = None
) -> None:
    """Add invisible text overlay to existing page without re-rendering."""

    # Check if we can use coordinate mapping
    if self.config.use_coordinate_mapping and ocr_result and hasattr(ocr_result, 'bbox') and ocr_result.bbox:
        # Use positioned text with coordinate mapping
        logger.info(f"Using coordinate mapping for text overlay")
        self._insert_positioned_text(page, page_rect, ocr_text, ocr_result)
    else:
        # FIXED: Use TextWriter fallback
        logger.warning("Using TextWriter fallback for text overlay")

        success = self._insert_text_fallback(page, ocr_text, page_rect)

        if success:
            logger.info(f"TextWriter fallback succeeded ({len(ocr_text)} chars)")
        else:
            logger.error("TextWriter fallback failed!")
            raise RuntimeError("Failed to add text overlay - TextWriter failed")
```

**CHANGES EXPLAINED**:
1. Removed broken `insert_textbox()` call
2. Calls `_insert_text_fallback()` helper which uses `TextWriter`
3. Simplified error handling
4. `_insert_positioned_text()` already fixed (Point 1)

---

### Point 3: `_full_rebuild_with_compression()` (Line ~1125)

**CURRENT CODE** (BROKEN):
```python
# ... (image insertion code) ...

# Step 4: Add OCR text with coordinate mapping
logger.info(f"Step 4: Adding OCR text ({len(ocr_text)} chars) to rebuilt page")

if self.config.use_coordinate_mapping and ocr_result and hasattr(ocr_result, 'bbox') and ocr_result.bbox:
    self._insert_positioned_text(new_page, page_rect, ocr_text, ocr_result)
else:
    # Fallback: Single textbox
    logger.warning("Using fallback textbox in full rebuild")

    try:
        result = new_page.insert_textbox(
            page_rect,
            ocr_text,
            fontsize=8,
            color=(1, 1, 1),
            overlay=True,
            render_mode=3  # DOES NOT WORK!
        )

        if result > 0:
            logger.info(f"SUCCESS - Text inserted in rebuilt page")
        else:
            logger.warning(f"WARNING - insert_textbox in rebuild returned {result}")

    except Exception as e:
        logger.error(f"FAILED to insert text in rebuilt page: {e}", exc_info=True)
        raise

# Step 5: Replace original page
# ...
```

**NEW CODE** (FIXED):
```python
# ... (image insertion code) ...

# Step 4: Add OCR text with coordinate mapping
logger.info(f"Step 4: Adding OCR text ({len(ocr_text)} chars) to rebuilt page")

if self.config.use_coordinate_mapping and ocr_result and hasattr(ocr_result, 'bbox') and ocr_result.bbox:
    self._insert_positioned_text(new_page, page_rect, ocr_text, ocr_result)
else:
    # FIXED: Use TextWriter fallback
    logger.warning("Using TextWriter fallback in full rebuild")

    success = self._insert_text_fallback(new_page, ocr_text, page_rect)

    if success:
        logger.info(f"TextWriter succeeded in rebuilt page ({len(ocr_text)} chars)")
    else:
        logger.error("TextWriter failed in rebuilt page!")
        raise RuntimeError("Failed to add text to rebuilt page - TextWriter failed")

# Step 5: Replace original page
# ...
```

**CHANGES EXPLAINED**:
1. Replaced `insert_textbox()` with `_insert_text_fallback()` helper
2. Uses `TextWriter` for batch text insertion
3. Consistent error handling with other methods

---

### Point 4: Phase 4 - Text Layer Pages (Line ~1920)

**CURRENT CODE** (BROKEN):
```python
# PHASE 4: Add text overlays to text-layer-only pages
if text_layer_pages:
    logger.info(f"Adding text overlays to {len(text_layer_pages)} text layer pages")

    for page_num, text in text_layer_pages.items():
        try:
            page = doc[page_num]
            page_rect = page.rect

            logger.info(f"Page {page_num+1}: Adding text overlay ({len(text)} chars)...")

            # ATTEMPT 1: Try insert_textbox with render_mode=3
            try:
                result = page.insert_textbox(
                    page_rect,
                    text,
                    fontsize=8,
                    color=(1, 1, 1),  # White (invisible)
                    overlay=True,
                    render_mode=3  # DOES NOT WORK!
                )

                if result > 0:
                    logger.info(f"Page {page_num+1}: SUCCESS - Text overlay added")
                else:
                    logger.warning(f"Page {page_num+1}: WARNING - returned {result}")

            except Exception as e1:
                logger.error(f"Page {page_num+1}: insert_textbox FAILED: {e1}")

                # ATTEMPT 2: Fallback - still broken!
                try:
                    logger.info(f"Page {page_num+1}: Trying fallback method...")
                    lines = text.split('\n')
                    y_pos = page_rect.y0 + 10
                    line_height = (page_rect.height - 20) / max(len(lines), 1)

                    for i, line in enumerate(lines):
                        if line.strip():
                            point = fitz.Point(page_rect.x0 + 5, y_pos + (i * line_height))
                            page.insert_text(
                                point,
                                line,
                                fontsize=1,  # Tiny font
                                color=(1, 1, 1),  # White
                                render_mode=3  # This DOES work, but font too small!
                            )
                    logger.info(f"Page {page_num+1}: SUCCESS - Fallback completed")
                except Exception as e2:
                    logger.error(f"Page {page_num+1}: Fallback ALSO FAILED: {e2}")
                    raise

        except Exception as e:
            logger.error(f"Page {page_num+1}: CRITICAL - Failed to add overlay - {e}", exc_info=True)
```

**NEW CODE** (FIXED):
```python
# PHASE 4: Add text overlays to text-layer-only pages
if text_layer_pages:
    logger.info(f"Adding text overlays to {len(text_layer_pages)} text layer pages")

    for page_num, text in text_layer_pages.items():
        try:
            page = doc[page_num]
            page_rect = page.rect

            logger.info(f"Page {page_num+1}: Adding text overlay ({len(text)} chars)...")

            # FIXED: Use TextWriter for text-layer-only pages
            success = self._insert_text_fallback(page, text, page_rect)

            if success:
                logger.info(f"Page {page_num+1}: Text overlay added with TextWriter")
            else:
                logger.error(f"Page {page_num+1}: TextWriter failed!")
                raise RuntimeError(f"Failed to add text overlay to page {page_num+1}")

        except Exception as e:
            logger.error(
                f"Page {page_num+1}: CRITICAL - Failed to add text overlay - {e}",
                exc_info=True
            )
            # Continue with other pages (don't fail entire batch)
```

**CHANGES EXPLAINED**:
1. Removed **both broken attempts** (primary `insert_textbox()` and fallback with `fontsize=1`)
2. Single call to `_insert_text_fallback()` using `TextWriter`
3. Cleaner code, better logging
4. Continues processing other pages if one fails

---

## 7. TESTING & VERIFICATION

### Test Plan

#### Test 1: Basic Searchability Test

**File**: Create new test `test_ocr_searchability.py`

```python
import fitz
import pytest
from pathlib import Path

def test_ocr_output_is_searchable(tmp_path):
    """Verify OCR'd PDF contains searchable text."""

    # Process test PDF with OCR
    from services.ocr_batch_service import OCRBatchService

    service = OCRBatchService(use_gpu=False)
    input_pdf = "tests/fixtures/scanned_page.pdf"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = service._process_single_file(
        file_path=input_pdf,
        output_dir=str(output_dir),
        file_idx=1,
        total_files=1
    )

    assert result['status'] == 'success'
    output_pdf = Path(result['output_path'])
    assert output_pdf.exists()

    # VERIFICATION: Extract text from output PDF
    doc = fitz.open(str(output_pdf))
    page = doc[0]
    extracted_text = page.get_text()
    doc.close()

    # CRITICAL: Text should be extractable
    assert len(extracted_text) > 0, "Output PDF has NO searchable text!"
    assert len(extracted_text) > 50, f"Only {len(extracted_text)} chars extracted - too few!"

    print(f"SUCCESS: Extracted {len(extracted_text)} characters from OCR'd PDF")
    print(f"Sample: {extracted_text[:200]}")


def test_coordinate_mapped_text_searchability(tmp_path):
    """Verify coordinate-mapped text is searchable."""

    # Test with PDF that has good OCR bboxes
    from services.ocr_batch_service import OCRBatchService

    service = OCRBatchService(use_gpu=False)
    service.config.use_coordinate_mapping = True  # Force coordinate mapping

    input_pdf = "tests/fixtures/scanned_page.pdf"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = service._process_single_file(
        file_path=input_pdf,
        output_dir=str(output_dir),
        file_idx=1,
        total_files=1
    )

    # Verify searchability
    output_pdf = Path(result['output_path'])
    doc = fitz.open(str(output_pdf))
    page = doc[0]
    extracted_text = page.get_text()
    doc.close()

    assert len(extracted_text) > 0, "Coordinate-mapped text is NOT searchable!"
    print(f"Coordinate mapping: {len(extracted_text)} chars searchable")
```

#### Test 2: Manual Verification

**Process**:
1. Run OCR on test PDF: `npm run tauri:dev` → Load scanned PDF → Run OCR
2. Open output PDF in Adobe Acrobat or PDF reader
3. Press Ctrl+F and search for known text from the scan
4. **Expected**: Text is found and highlighted
5. **Previous Behavior**: "No matches found"

#### Test 3: Regression Test

**Verify No Visual Changes**:
```python
def test_text_is_invisible(tmp_path):
    """Verify inserted text is invisible (render_mode=3)."""

    from services.ocr_batch_service import OCRBatchService

    service = OCRBatchService(use_gpu=False)

    # Process PDF
    result = service._process_single_file(...)
    output_pdf = Path(result['output_path'])

    # Render page to image
    doc = fitz.open(str(output_pdf))
    page = doc[0]
    pix = page.get_pixmap(dpi=150)

    # Compare to original (should look identical)
    # Text should be invisible, only searchable

    # Simple check: No white text artifacts visible
    # (More sophisticated: compare pixel-by-pixel with original)
```

### Verification Checklist

- [ ] Test 1: Basic searchability test passes
- [ ] Test 2: Manual search in PDF reader finds text
- [ ] Test 3: Text is invisible (no visual artifacts)
- [ ] Phase 4 pages (text layer only) are searchable
- [ ] Coordinate-mapped pages are searchable
- [ ] Fallback pages (no bboxes) are searchable
- [ ] Full rebuild pages are searchable
- [ ] Large PDFs process without errors
- [ ] Multi-page PDFs have all pages searchable
- [ ] Special characters (accents, symbols) are searchable

---

## 8. EDGE CASES & ERROR HANDLING

### Edge Case 1: Empty Text

**Scenario**: OCR returns empty string or whitespace-only text

**Handling**:
```python
# In _insert_text_fallback()
if not text or not text.strip():
    logger.debug("Empty text, skipping insertion")
    return True  # Not an error, just nothing to insert
```

### Edge Case 2: Very Long Text

**Scenario**: OCR returns >100KB of text for single page

**Handling**:
```python
# In _insert_text_fallback()
MAX_TEXT_LENGTH = 100_000  # 100KB

if len(text) > MAX_TEXT_LENGTH:
    logger.warning(f"Text too long ({len(text)} chars), truncating to {MAX_TEXT_LENGTH}")
    text = text[:MAX_TEXT_LENGTH] + "... [truncated]"
```

### Edge Case 3: Special Characters

**Scenario**: Text contains Unicode, emojis, control characters

**Handling**:
```python
# Clean text before insertion
import unicodedata

def _clean_text_for_pdf(text: str) -> str:
    """Remove problematic characters from text."""
    # Remove control characters except newline/tab
    text = ''.join(ch for ch in text if ch in '\n\t' or not unicodedata.category(ch).startswith('C'))
    # Normalize Unicode
    text = unicodedata.normalize('NFKC', text)
    return text

# Use in insertion functions
text = self._clean_text_for_pdf(text)
```

### Edge Case 4: Coordinate Mapping Failure

**Scenario**: Bounding boxes are malformed or out of bounds

**Handling**:
Already handled - `_insert_text_with_bbox()` returns `False` on failure, code falls back to `_insert_text_fallback()`.

### Edge Case 5: Font Not Available

**Scenario**: "helv" font not available in PyMuPDF

**Handling**:
```python
# In helper functions, catch font errors
try:
    font = fitz.Font("helv")
except Exception as e:
    logger.warning(f"Font 'helv' not available: {e}, using default")
    font = fitz.Font()  # Use default font
```

### Error Recovery Strategy

**Principle**: **Never fail silently** - always log errors but try to continue

**Hierarchy**:
1. **Try**: Coordinate mapping with `insert_text()` + bboxes
2. **Catch**: Fall back to `TextWriter` without coordinates
3. **Catch**: Log error, continue processing other pages
4. **Only fail**: If entire file processing fails (not just one page)

---

## 9. ROLLBACK PLAN

### If Implementation Fails

**Symptoms**:
- New code causes crashes
- Text still not searchable
- PDFs corrupted

**Rollback Steps**:

1. **Immediate**: Revert `ocr_batch_service.py` to previous commit
   ```bash
   cd F:\Document-De-Bundler
   git checkout HEAD~1 -- python-backend/services/ocr_batch_service.py
   ```

2. **Verify**: Run existing tests to confirm rollback works
   ```bash
   cd python-backend
   pytest tests/test_ocr_batch_service.py -v
   ```

3. **Communicate**: Update documentation that searchability is still broken

### If Partial Success

**Scenario**: Some insertion points work, others don't

**Strategy**:
- Keep working methods
- Revert broken methods to old code (with clear TODO comments)
- File bug report with specific failure details

---

## APPENDIX A: PYMUPDF API REFERENCE

### `insert_text()` Full Signature

```python
page.insert_text(
    point: Point,              # Insertion point (baseline-left)
    text: str,                 # Text to insert
    fontsize: float = 11,      # Font size in points
    fontname: str = "helv",    # Font name
    fontfile: str = None,      # Custom font file path
    set_simple: int = 0,       # Font embedding: 0=full, 1=simple
    encoding: int = 0,         # Font encoding
    color: tuple = None,       # RGB color (0-1 range)
    fill: tuple = None,        # Fill color (alternative to color)
    render_mode: int = 0,      # 0=fill, 1=stroke, 2=both, 3=INVISIBLE
    border_width: float = 1,   # Stroke width (for render_mode 1, 2)
    rotate: int = 0,           # Rotation angle (0, 90, 180, 270)
    morph: tuple = None,       # (pivot, matrix) for transformation
    stroke_opacity: float = 1, # Stroke opacity (0-1)
    fill_opacity: float = 1,   # Fill opacity (0-1)
    overlay: bool = True,      # True=foreground, False=background
    oc: int = 0                # Optional content reference
) -> int
```

**Returns**: Number of successfully inserted characters

### `TextWriter` Full API

```python
# Create TextWriter
tw = fitz.TextWriter(
    page_rect: Rect,      # Page rectangle
    opacity: float = 1,   # Default opacity
    color: tuple = None   # Default color
)

# Accumulate text
tw.append(
    pos: Point | tuple,      # (x, y) position
    text: str,               # Text to insert
    font: Font = None,       # Font object (default: Helvetica)
    fontsize: float = 11,    # Font size
    language: str = None,    # Language code (e.g., "en")
    right_to_left: int = 0,  # 0=LTR, 1=RTL
    small_caps: bool = False # Small caps rendering
) -> None

# Write to page
tw.write_text(
    page: Page,              # Target page
    opacity: float = None,   # Override default opacity
    color: tuple = None,     # Override default color
    morph: tuple = None,     # (pivot, matrix) transformation
    overlay: bool = True,    # Foreground/background
    oc: int = 0,             # Optional content
    render_mode: int = 0     # 0=fill, 1=stroke, 2=both, 3=INVISIBLE
) -> None

# Get fill opacity
tw.fill_opacity() -> float

# Get text bounding rect
tw.text_rect -> Rect
```

---

## APPENDIX B: COORDINATE MAPPING DETAILS

### Image to PDF Coordinate Conversion

**OCR Image Coordinates** (pixels, top-left origin):
```
(0, 0) ─────────────────────→ (image_width, 0)
  │
  │  bbox: (x, y, width, height)
  │
  ↓
(0, image_height)
```

**PDF Coordinates** (points, bottom-left origin):
```
(0, page_height) ─────────────→ (page_width, page_height)
  ↑
  │  bbox: (x0, y0, x1, y1)
  │
  │
(0, 0) ───────────────────────→ (page_width, 0)
```

**Conversion Formula**:
```python
# X coordinate (same direction)
pdf_x = (image_x / image_width) * page_width

# Y coordinate (flip vertical axis)
pdf_y = page_height - ((image_y / image_height) * page_height)

# Bounding box
pdf_bbox = fitz.Rect(
    pdf_x0,              # Left
    pdf_y1,              # Bottom (note: y1 in PDF is bottom!)
    pdf_x0 + pdf_width,  # Right
    pdf_y0               # Top
)
```

### Font Size from Bbox Height

```python
# Calculate font size to fit text in bbox
font = fitz.Font("helv")

# Simple approach (use bbox height)
fontsize = bbox_height * 0.9  # 90% to prevent overflow

# Advanced approach (use text length)
text_length = font.text_length(text, fontsize=1)  # Length at fontsize=1
fontsize = bbox_width / text_length  # Scale to fit width

# Take minimum of height-based and width-based
fontsize = min(
    bbox_height * 0.9,
    bbox_width / font.text_length(text, fontsize=1)
)
```

---

## APPENDIX C: DIAGNOSTIC LOGGING

### Add Diagnostic Helper

**Location**: Add to `ocr_batch_service.py`

```python
def _verify_page_searchability(
    self,
    page: fitz.Page,
    page_num: int,
    expected_text_length: int
) -> None:
    """
    Verify that page has searchable text (diagnostic only).

    Args:
        page: Page to check
        page_num: Page number (for logging)
        expected_text_length: Expected text length from OCR
    """
    try:
        extracted = page.get_text()
        extracted_len = len(extracted.strip())

        if extracted_len == 0:
            logger.error(
                f"Page {page_num+1}: VERIFICATION FAILED - No searchable text! "
                f"(Expected ~{expected_text_length} chars)"
            )
        elif extracted_len < expected_text_length * 0.5:
            logger.warning(
                f"Page {page_num+1}: VERIFICATION WARNING - Only {extracted_len} chars "
                f"extracted (expected ~{expected_text_length})"
            )
        else:
            logger.debug(
                f"Page {page_num+1}: VERIFICATION OK - {extracted_len} chars searchable"
            )

    except Exception as e:
        logger.warning(f"Page {page_num+1}: Verification failed: {e}")
```

**Usage**: Call after each text insertion
```python
# After inserting text
self._verify_page_searchability(page, page_num, len(ocr_text))
```

---

## CONCLUSION

This handoff provides:

1. **Complete problem analysis** - Why `insert_textbox()` doesn't work
2. **PyMuPDF API research** - Correct methods (`insert_text()`, `TextWriter`)
3. **Comprehensive solution** - Two helper functions + 4 insertion point fixes
4. **Code changes** - Before/after for all 4 locations with explanations
5. **Testing plan** - Automated and manual verification
6. **Edge cases** - Handling for all common failure modes
7. **Rollback plan** - If implementation fails

**Next Steps**:
1. Implement helper functions
2. Update 4 insertion points
3. Run tests
4. Verify searchability manually
5. Update documentation

**Estimated completion time**: 2.5 hours

**Risk level**: Low-Medium (well-researched solution, clear rollback path)
