# Test Scenarios: Partial Text Layer Coverage Detection

## Overview

This document provides comprehensive test scenarios for validating the partial text layer coverage detection feature implemented in `python-backend/services/ocr/text_quality.py`.

### Purpose

The coverage detection feature solves the problem of **partial text layers** - pages where only some of the visible text has an embedded text layer (e.g., header has text layer but scanned body doesn't). Without coverage detection, these pages would skip OCR, leaving content unsearchable.

### How It Works

1. Analyzes bounding boxes of all text blocks on each page
2. Calculates what percentage of the page is covered by text
3. Detects presence of images and vector graphics
4. Computes confidence score based on coverage and content type
5. Triggers OCR if coverage is inadequate

### Integration

- Integrated into `TextLayerValidator.has_valid_text_layer()`
- Runs automatically during OCR batch processing
- Configurable via `TextQualityThresholds` parameters
- Can be disabled for backward compatibility

---

## Test Scenario Categories

### 1. Basic Scenarios

These cover the fundamental cases that the system should handle correctly.

#### 1.1: Pure Digital PDF (Full Text Layer)

**Description**: PDF created digitally with complete embedded text layer on all pages.

**Sample**: Business report, academic paper, digital invoice

**Expected Behavior**:
- Text extraction: Fast (from text layer)
- Coverage ratio: 75-90%
- Has images: False
- Coverage confidence: 0.9-1.0
- **Result**: PASS validation, NO OCR needed

**Expected Metrics**:
```
Total Characters: 2000+
Text Coverage: 85%
Image Coverage: 0%
Uncovered Area: 15%
Coverage Confidence: 95%+
Status: VALID
```

**Validation Steps**:
1. Process PDF through OCR batch service
2. Check logs for "Using text layer" messages
3. Verify no OCR processing occurred
4. Confirm output PDF copied as-is without modification

---

#### 1.2: Pure Scanned PDF (No Text Layer)

**Description**: Scanned document with no embedded text layer.

**Sample**: Scanned book pages, scanned contracts, photocopied documents

**Expected Behavior**:
- Text extraction: None available
- Coverage ratio: 0%
- Has images: True (page is image)
- Coverage confidence: 0.0
- **Result**: FAIL validation, FULL OCR needed

**Expected Metrics**:
```
Total Characters: 0
Text Coverage: 0%
Image Coverage: 90%+
Uncovered Area: 100%
Coverage Confidence: 0%
Status: INVALID - OCR RECOMMENDED
```

**Validation Steps**:
1. Process PDF through OCR batch service
2. Check logs for "Needs OCR" messages
3. Verify OCR processing on all pages
4. Confirm output PDF has searchable text layer

---

#### 1.3: Blank Pages

**Description**: Pages with no visible content.

**Sample**: Separator pages, blank backs of single-sided printing

**Expected Behavior**:
- Text extraction: Empty or minimal
- Coverage ratio: 0%
- Has images: False
- Coverage confidence: 0.0
- **Result**: FAIL validation (minimum chars not met)

**Expected Metrics**:
```
Total Characters: 0-50
Text Coverage: 0%
Image Coverage: 0%
Uncovered Area: 100%
Coverage Confidence: 0%
Status: INVALID - OCR RECOMMENDED
```

**Validation Steps**:
1. Process PDF with blank pages
2. Verify blank pages trigger OCR (will find no text)
3. Check that blank pages don't crash processing
4. Confirm output includes blank pages

---

#### 1.4: Sparse Text (Intentional)

**Description**: Pages with intentionally minimal text (cover pages, title pages, section dividers).

**Sample**: Book covers, presentation title slides, chapter dividers

**Expected Behavior** (with default settings):
- Text extraction: 100-500 characters
- Coverage ratio: 15-30%
- Has images: Variable
- Coverage confidence: 0.4-0.7
- **Result**: May PASS or FAIL depending on exact coverage

**Expected Metrics**:
```
Total Characters: 150
Text Coverage: 25%
Image Coverage: 10%
Uncovered Area: 65%
Coverage Confidence: 50-70%
Status: Variable (depends on configuration)
```

**Configuration Adjustment**:
For documents with intentionally sparse text, use permissive thresholds:
```python
permissive = TextQualityThresholds(
    min_text_coverage_ratio=0.30,  # Allow 30% coverage
    max_uncovered_area_ratio=0.70   # Allow 70% uncovered
)
```

---

### 2. Partial Coverage Scenarios

These are the PRIMARY USE CASES for coverage detection.

#### 2.1: Header with Text, Scanned Body (PRIMARY)

**Description**: Page has text layer for header/title but body content is scanned image.

**Sample**: Scanned forms with digital overlays, hybrid documents, partially OCR'd scans

**Expected Behavior**:
- Text extraction: 150-300 characters (header only)
- Coverage ratio: 10-20% (header area only)
- Has images: True (scanned body)
- Coverage confidence: 0.1-0.3 (LOW - triggers OCR)
- **Result**: FAIL validation, FULL OCR needed

**Expected Metrics**:
```
Total Characters: 200
Text Coverage: 15%
Image Coverage: 80%
Uncovered Area: 85%
Text Blocks: 1-3
Has Images: True
Coverage Confidence: 20% (LOW)
Status: INVALID - OCR RECOMMENDED
```

**Why This Works**:
- Low text coverage (15% < 70% threshold)
- High uncovered area (85% > 30% threshold)
- Images present + low coverage → confidence penalty
- Few text blocks + low coverage → additional penalty
- **Result**: Coverage confidence drops below 50% threshold → FAIL

**Validation Steps**:
1. Create PDF with header text layer + scanned body
2. Process through OCR batch
3. Verify logs show coverage metrics: `coverage=0.15, coverage_conf=0.20`
4. Confirm "Text layer failed validation" logged
5. Verify entire page sent to OCR (not just body)
6. Check output PDF has searchable text for full page

---

#### 2.2: Footer/Margin Text Only

**Description**: Page has text layer only in footer or margins, main content is scanned.

**Sample**: Scanned documents with page numbers added digitally

**Expected Behavior**:
- Text extraction: 50-150 characters (footer/margins)
- Coverage ratio: 5-15%
- Has images: True
- Coverage confidence: 0.1-0.2
- **Result**: FAIL validation, FULL OCR needed

**Expected Metrics**:
```
Total Characters: 80
Text Coverage: 8%
Image Coverage: 85%
Uncovered Area: 92%
Text Blocks: 1-2
Coverage Confidence: 15%
Status: INVALID - OCR RECOMMENDED
```

---

#### 2.3: Partial OCR (First Few Paragraphs)

**Description**: OCR was started but stopped partway through the page.

**Sample**: Interrupted OCR processing, corrupted OCR data

**Expected Behavior**:
- Text extraction: 500-1000 characters
- Coverage ratio: 30-40%
- Has images: True (unprocessed portion)
- Coverage confidence: 0.3-0.5
- **Result**: FAIL validation (borderline but fails)

**Expected Metrics**:
```
Total Characters: 750
Text Coverage: 35%
Image Coverage: 60%
Uncovered Area: 65%
Text Blocks: 5-10
Coverage Confidence: 40%
Status: INVALID - OCR RECOMMENDED
```

**Why This Works**:
- Coverage confidence (40%) < threshold (50%)
- Even though has enough characters, spatial coverage is inadequate

---

#### 2.4: Mixed Regions (Text + Scanned Table)

**Description**: Page has digital text paragraphs but scanned table/image in middle.

**Sample**: Reports with embedded scanned charts/tables

**Expected Behavior**:
- Text extraction: 1000-2000 characters
- Coverage ratio: 40-60% (depends on table size)
- Has images: True
- Coverage confidence: 0.5-0.8
- **Result**: Variable (depends on exact proportions)

**Expected Metrics**:
```
Total Characters: 1500
Text Coverage: 50%
Image Coverage: 40%
Uncovered Area: 50%
Text Blocks: 15-20
Coverage Confidence: 60-70%
Status: May PASS or FAIL (borderline case)
```

**Note**: This is a borderline case. Behavior depends on:
- Exact coverage ratio (≥70% passes)
- Size of scanned region
- Can tune thresholds if needed for specific document types

---

### 3. Mixed Content Scenarios

#### 3.1: Image-Heavy PDF with Text Overlays

**Description**: PDF with large images but all text has proper text layer.

**Sample**: Illustrated documents, marketing materials, photo albums with captions

**Expected Behavior**:
- Text extraction: Variable (500-3000 characters)
- Coverage ratio: 20-60% (depends on text amount)
- Has images: True
- Coverage confidence: Variable
- **Result**: Should PASS if text coverage adequate

**Expected Metrics**:
```
Total Characters: 1200
Text Coverage: 25%
Image Coverage: 70%
Uncovered Area: 5% (images have text overlay)
Text Blocks: 20-30
Coverage Confidence: 70-80%
Status: VALID (text properly overlaid on images)
```

**Why This Works**:
- Uncovered area is LOW (5%) even though text coverage is medium
- This indicates images have text layers
- Coverage confidence remains high

---

#### 3.2: Charts/Diagrams with Embedded Text

**Description**: Pages with vector graphics (charts, diagrams) containing text.

**Sample**: Business presentations, scientific papers with figures

**Expected Behavior**:
- Text extraction: Variable
- Coverage ratio: Variable
- Has drawings: True
- Coverage confidence: Variable
- **Result**: Depends on whether chart text is in text layer

**Expected Metrics (Good Case - Text Layer Present)**:
```
Total Characters: 800
Text Coverage: 60%
Has Drawings: True
Coverage Confidence: 85%
Status: VALID
```

**Expected Metrics (Bad Case - Chart Text Not Extracted)**:
```
Total Characters: 200
Text Coverage: 15%
Has Drawings: True
Coverage Confidence: 30%
Status: INVALID - OCR RECOMMENDED
```

---

#### 3.3: Multi-Column Layouts

**Description**: Pages with multiple columns (newspapers, academic journals).

**Sample**: Newspaper articles, conference proceedings, multi-column documents

**Expected Behavior**:
- Text extraction: High character count
- Coverage ratio: 60-80% (whitespace between columns acceptable)
- Has images: Variable
- Coverage confidence: 0.8-1.0
- **Result**: Should PASS

**Expected Metrics**:
```
Total Characters: 3000+
Text Coverage: 70%
Uncovered Area: 30% (column gutters, margins)
Text Blocks: 30-50
Coverage Confidence: 90%
Status: VALID
```

**Why This Works**:
- 70% threshold accounts for multi-column whitespace
- High text block count indicates proper layout
- Coverage confidence remains high

---

#### 3.4: Rotated/Skewed Pages

**Description**: Pages that are rotated or slightly skewed.

**Sample**: Scanned documents with rotation, user-rotated PDFs

**Expected Behavior**:
- PyMuPDF normalizes coordinates automatically
- Bounding boxes still calculated correctly
- Coverage detection should work normally
- **Result**: Same as non-rotated version

**Note**: PyMuPDF handles coordinate normalization, so rotation should not affect coverage detection accuracy.

---

### 4. Edge Cases

#### 4.1: Overlapping Text Blocks

**Description**: Text blocks that overlap (e.g., watermarks, annotations).

**Sample**: Watermarked documents, annotated PDFs

**Expected Behavior**:
- Coverage calculation sums overlapping areas (conservative)
- May slightly overestimate coverage
- **Result**: Better to overestimate and skip OCR than underestimate

**Impact**: Minimal - conservative approach is acceptable

---

#### 4.2: Corrupted Partial Text Layer

**Description**: Text layer exists but has corruption/encoding errors.

**Sample**: Damaged PDFs, incorrect encoding

**Expected Behavior**:
- Existing quality validation catches corruption (unicode errors, low printable ratio)
- Coverage may be adequate but quality fails
- **Result**: FAIL validation, OCR triggered

**This is CORRECT behavior**: Both quality AND coverage must pass

---

#### 4.3: Very Large Pages (Posters, Engineering Drawings)

**Description**: Large format PDFs with sparse text.

**Sample**: Architectural drawings, posters, banners

**Expected Behavior**:
- Text coverage naturally low due to large page area
- May need permissive thresholds
- **Result**: Configure for sparse text

**Configuration**:
```python
large_format = TextQualityThresholds(
    min_text_coverage_ratio=0.20,  # 20% for large format
    max_uncovered_area_ratio=0.80   # 80% uncovered OK
)
```

---

#### 4.4: Pages with Only Image-Embedded Text

**Description**: Text is part of an image (screenshot, photo of document).

**Sample**: Screenshots, photos of text, image-embedded documents

**Expected Behavior**:
- No text layer (text is pixels, not selectable)
- Coverage ratio: 0%
- Has images: True
- **Result**: FAIL validation, OCR needed

**This is CORRECT behavior**: OCR will extract text from image

---

### 5. Configuration Test Matrix

Test the same document with different configurations to verify threshold behavior.

#### Test Document: Partial Coverage Sample
- Header: 200 characters, 15% coverage
- Body: Scanned image, 80% of page
- Uncovered: 85%

#### Configuration Tests:

**5.1: Default Configuration**
```python
default = TextQualityThresholds()  # Uses defaults
```
Expected: FAIL (coverage 15% < 70%, confidence ~20%)

**5.2: Strict Configuration**
```python
strict = TextQualityThresholds(
    min_text_coverage_ratio=0.85,
    min_coverage_confidence=0.70
)
```
Expected: FAIL (even stricter, definitely fails)

**5.3: Permissive Configuration**
```python
permissive = TextQualityThresholds(
    min_text_coverage_ratio=0.30,
    max_uncovered_area_ratio=0.70,
    min_coverage_confidence=0.30
)
```
Expected: May PASS or FAIL (borderline at 15% coverage)

**5.4: Coverage Detection Disabled**
```python
disabled = TextQualityThresholds(
    enable_coverage_detection=False
)
```
Expected: Depends on quality-only validation (likely PASS if header text quality is good)

**This is the OLD behavior**: Would incorrectly skip OCR

---

## Test Data Requirements

### Creating Test PDFs

#### Method 1: Using PyMuPDF (Programmatic)

Create synthetic test PDFs with specific coverage scenarios:

```python
import fitz

# Create PDF with header text + blank body (simulates partial coverage)
doc = fitz.open()
page = doc.new_page(width=612, height=792)  # US Letter

# Add text header
header_rect = fitz.Rect(50, 50, 562, 100)
page.insert_textbox(header_rect, "This is a header with text layer",
                   fontsize=14, fontname="helv")

# Add image to body (simulates scanned content)
# Leave body area empty or insert image

doc.save("test_partial_coverage.pdf")
```

#### Method 2: Real-World Samples

Collect sample PDFs:
1. **Pure digital**: Export from Word/Google Docs
2. **Pure scanned**: Scan physical documents without OCR
3. **Partial coverage**:
   - Scan document
   - Add header/footer in PDF editor
   - Save without re-OCR'ing body
4. **Mixed content**: Marketing materials, illustrated documents

### Required Test Files

Create test suite with:
- `test_pure_digital.pdf` - 100% text layer
- `test_pure_scanned.pdf` - 0% text layer
- `test_header_only.pdf` - Header text, scanned body
- `test_footer_only.pdf` - Footer text, scanned body
- `test_partial_ocr.pdf` - First paragraph OCR'd, rest scanned
- `test_blank_pages.pdf` - Mix of content and blank pages
- `test_sparse_text.pdf` - Cover page, title page
- `test_multi_column.pdf` - 2-3 column layout
- `test_image_heavy.pdf` - Images with text overlays
- `test_charts.pdf` - Vector graphics with text

---

## Validation Checklist

### Pre-Test Setup

- [ ] PyTorch and PaddleOCR dependencies installed
- [ ] GPU drivers up to date (if using GPU)
- [ ] Test PDF samples prepared
- [ ] Logging level set to DEBUG for detailed output
- [ ] Clean output directory

### During Testing

For each test scenario:

- [ ] **Process PDF** through OCR batch service
- [ ] **Check logs** for coverage metrics
- [ ] **Verify decision** (OCR vs text layer extraction)
- [ ] **Inspect output** PDF for searchable text
- [ ] **Measure performance** (processing time per page)

### Log Validation

Look for these log entries:

**Coverage Detection Success**:
```
Coverage metrics: text=0.15, images=True, drawings=False,
uncovered=0.85, confidence=0.20
```

**Validation Decision**:
```
Text layer failed validation: chars=200, confidence=0.65,
coverage=0.15, coverage_conf=0.20
```

**OCR Triggered**:
```
Page X: Needs OCR
```

### Output Validation

- [ ] **Open output PDF** in PDF reader (Adobe, Foxit, etc.)
- [ ] **Test text search** - search for text that was in scanned regions
- [ ] **Verify searchability** - all expected content is searchable
- [ ] **Check file size** - output should be similar or slightly larger
- [ ] **Visual inspection** - pages should look identical to original

### Performance Benchmarks

Expected performance (with coverage detection enabled):

| Operation | Time per Page | Notes |
|-----------|---------------|-------|
| Pure digital PDF | 10-20ms | Fast text extraction |
| Pure scanned PDF | 150-350ms | Full OCR (GPU) |
| Partial coverage detection | +5-10ms | Minimal overhead |
| Coverage detection (total) | <50ms | Bounding box analysis |

**Total Overhead**: <50ms per page (negligible compared to OCR time)

### Configuration Testing

Test with each configuration mode:

- [ ] **Default mode** - Coverage detection enabled with 70% threshold
- [ ] **Strict mode** - 85% coverage required
- [ ] **Permissive mode** - 30% coverage acceptable
- [ ] **Disabled mode** - Coverage detection off (backward compatibility)

---

## Known Limitations

### 1. Cannot Detect Text Quality Within Covered Areas

**Issue**: If text layer covers the right area but has poor OCR quality (wrong words), coverage detection won't catch it.

**Example**: Scanned page with low-quality OCR that covers 80% of page but text is garbled.

**Workaround**: Existing quality validation catches this via unicode errors, printable ratio, etc.

**Future Enhancement**: Add OCR quality scoring for existing text layers.

---

### 2. Conservative Area Calculation

**Issue**: Overlapping text blocks are summed (may slightly overestimate coverage).

**Impact**: In rare cases, might skip OCR when it would be beneficial.

**Workaround**: Acceptable tradeoff - better to be conservative.

**Future Enhancement**: Implement union algorithm for precise area calculation.

---

### 3. Cannot Detect Positional Accuracy

**Issue**: Text layer might cover correct area but text is in wrong position.

**Example**: OCR placed text blocks in wrong locations on page.

**Workaround**: User would notice during review (text doesn't align visually).

**Future Enhancement**: Add position validation by comparing rendered page to text layer.

---

### 4. Vector Graphics Text Detection

**Issue**: Cannot determine if vector drawings contain text or are pure graphics.

**Example**: Chart with axis labels (needs OCR) vs. pure decorative graphics (doesn't need OCR).

**Current Behavior**: Detects drawings but treats conservatively.

**Future Enhancement**: Analyze drawing complexity to estimate text likelihood.

---

### 5. Very Large Page Formats

**Issue**: Architectural drawings, posters may have naturally low text coverage.

**Workaround**: Use permissive configuration for these document types.

**Example**:
```python
large_format = TextQualityThresholds(
    min_text_coverage_ratio=0.20
)
```

---

### 6. Performance on Extremely Large PDFs

**Issue**: For PDFs with thousands of pages, coverage analysis adds overhead.

**Impact**: ~5-50ms per page (negligible compared to OCR but accumulates).

**Workaround**: Can disable coverage detection if performance critical.

**Benchmark**: 5000 pages × 20ms = ~100 seconds total overhead

---

## Future Improvements

### High Priority

1. **Automated Test Suite**
   - Unit tests for coverage calculation
   - Integration tests with sample PDFs
   - Regression tests for edge cases

2. **OCR Quality Scoring**
   - Detect when existing text layer has poor OCR quality
   - Compare to re-OCR to determine if improvement needed

3. **Position Validation**
   - Verify text layer positions match visual content
   - Detect misaligned OCR text

### Medium Priority

4. **Union Algorithm for Area Calculation**
   - Precisely calculate covered area accounting for overlaps
   - More accurate coverage ratios

5. **Adaptive Thresholds**
   - Automatically adjust based on document type detection
   - Machine learning to optimize thresholds

6. **Visual Content Analysis**
   - Detect type of images (photos vs scanned text)
   - Different handling for different content types

### Low Priority

7. **Performance Optimization**
   - Cache bounding box calculations
   - Parallel processing for large batches
   - Skip coverage detection for obviously valid/invalid pages

8. **User Feedback Loop**
   - Allow users to mark false positives/negatives
   - Use feedback to tune thresholds

---

## Troubleshooting Guide

### Issue: Coverage Detection Not Running

**Symptoms**: No coverage metrics in logs

**Possible Causes**:
1. Coverage detection disabled in configuration
2. PyMuPDF import failing
3. Exception in coverage calculation

**Debug Steps**:
```python
# Check if enabled
print(validator.thresholds.enable_coverage_detection)  # Should be True

# Check for import errors
import fitz
print(fitz.__version__)

# Check logs for warnings
grep "Coverage detection failed" logs.txt
```

---

### Issue: All Pages Failing Validation

**Symptoms**: Every page sent to OCR even digital PDFs

**Possible Causes**:
1. Thresholds too strict
2. Coverage calculation error

**Debug Steps**:
```python
# Check threshold values
print(validator.thresholds.min_text_coverage_ratio)  # Should be 0.70
print(validator.thresholds.min_coverage_confidence)  # Should be 0.50

# Enable DEBUG logging to see coverage values
# Look for: "Coverage metrics: text=X.XX, ..."
```

---

### Issue: Partial Coverage Pages Not Detected

**Symptoms**: Pages with partial text layers skip OCR

**Possible Causes**:
1. Coverage detection disabled
2. Partial coverage passes thresholds (edge case)
3. Images not detected properly

**Debug Steps**:
```python
# Check coverage metrics in logs
# For partial page, should see:
# text_coverage_ratio < 0.70
# has_images = True
# coverage_confidence < 0.50

# If not, may need to adjust thresholds or investigate
```

---

### Issue: Poor Performance

**Symptoms**: Slow processing even for digital PDFs

**Possible Causes**:
1. Coverage detection overhead
2. Large page sizes
3. Many text blocks

**Debug Steps**:
```python
# Measure coverage detection time
import time
start = time.time()
metrics = validator._calculate_coverage_metrics(page)
print(f"Coverage detection: {(time.time() - start) * 1000}ms")

# Should be <50ms. If higher, consider:
# - Disabling for known digital PDFs
# - Optimizing bounding box calculations
```

---

## Summary

This test plan provides comprehensive coverage of:
- ✅ **10+ test scenarios** covering all major use cases
- ✅ **Expected behaviors** and metrics for each scenario
- ✅ **Configuration testing** across multiple modes
- ✅ **Validation checklists** for thorough testing
- ✅ **Performance benchmarks** for monitoring
- ✅ **Known limitations** and workarounds
- ✅ **Troubleshooting guide** for common issues

Use this document to:
1. **Validate** the coverage detection feature works correctly
2. **Document** expected behaviors for future reference
3. **Create** automated tests based on these scenarios
4. **Troubleshoot** issues when they arise
5. **Tune** configuration for specific document types

The coverage detection feature significantly improves OCR accuracy by catching partial text layer cases that would otherwise be missed.
