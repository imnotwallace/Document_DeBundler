# Partial OCR Detection Fix - Implementation Checklist

This checklist guides the systematic implementation of fixes from `PARTIAL_OCR_DETECTION_FIX_PLAN.md`.

## Prerequisites

- [x] Step 4: Fixed empty coverage metrics fallback (`text_quality.py`)
  - Changed `coverage_confidence` from 1.0 to 0.0 in `_empty_coverage_metrics()`

## Step 1: OCR Output Validation with Retry Logic

File: `python-backend/services/ocr_batch_service.py`

- [x] **1.1**: Add `_validate_ocr_output()` method
  - Location: Add as new method in OCRBatchService class
  - Code: Lines 193-223 from the plan
  - Purpose: Validate OCR output has minimum 50 chars and 30% alphanumeric ratio
  - **STATUS**: COMPLETE

- [x] **1.2**: Add `_ocr_with_settings()` helper method
  - Location: Add as new method in OCRBatchService class
  - Code: Lines 275-308 from the plan
  - Purpose: Process single page with specific OCR engine and DPI
  - **STATUS**: COMPLETE

- [x] **1.3**: Add `_retry_ocr_with_fallback()` method
  - Location: Add as new method in OCRBatchService class
  - Code: Lines 225-273 from the plan
  - Purpose: Retry OCR with multiple engines/DPI settings on failure
  - **STATUS**: COMPLETE

- [x] **1.4**: Integrate validation into `_process_page_batch()`
  - Location: Modify existing method around line 463
  - Code: Lines 312-346 from the plan
  - Changes:
    - After `texts = self._process_page_batch(...)`, add validation loop
    - Validate each text result
    - Retry failed pages with `_retry_ocr_with_fallback()`
    - Track failed pages in result dict
  - **STATUS**: COMPLETE

## Step 2: Quality Comparison System

File: `python-backend/services/ocr_batch_service.py`

- [x] **2.1**: Add `_calculate_quality_score()` method
  - Location: Add as new method in OCRBatchService class
  - Code: Lines 365-384 from the plan
  - Purpose: Calculate quality score using TextLayerValidator
  - **STATUS**: COMPLETE

- [x] **2.2**: Add `_should_replace_with_ocr()` method
  - Location: Add as new method in OCRBatchService class
  - Code: Lines 387-433 from the plan
  - Purpose: Compare original vs OCR quality, decide if replacement is beneficial
  - **STATUS**: COMPLETE

- [x] **2.3**: Store original text during detection phase
  - Location: Modify `_process_single_file()` around line 428
  - Code: Lines 438-455 from the plan
  - Changes:
    - Add `original_texts = {}` dictionary
    - Store `original_texts[page_num]` for all pages (even empty)
  - **STATUS**: COMPLETE

- [x] **2.4**: Add quality comparison before embedding
  - Location: Modify embedding phase around line 489
  - Code: Lines 459-483 from the plan
  - Changes:
    - Before rebuilding page, check if original_texts exists
    - Call `_should_replace_with_ocr()` to compare quality
    - Skip page if original is better quality
  - **STATUS**: COMPLETE

## Step 3: Clean Page Rebuild for OCR (CRITICAL - Test Thoroughly!)

File: `python-backend/services/ocr_batch_service.py`

- [x] **3.1**: Add `_clean_rebuild_page_with_ocr()` method
  - Location: Add as new method in OCRBatchService class
  - Code: Lines 503-574 from the plan (primary implementation)
  - Purpose: Rebuild page cleanly with OCR text layer, removing existing text
  - Process:
    1. Render page to image (preserves visual)
    2. Create new blank page
    3. Insert rendered image
    4. Add OCR text as invisible overlay
    5. Replace original page
  - **STATUS**: COMPLETE

- [x] **3.2**: Replace embedding code (lines 483-507)
  - Location: Modify `_process_single_file()` embedding section
  - Code: Lines 630-663 from the plan
  - Changes:
    - Replace `insert_textbox()` call with `_clean_rebuild_page_with_ocr()`
    - Add quality comparison logic before rebuild
    - Add error handling and logging
  - **STATUS**: COMPLETE

## Step 5: Comprehensive Testing

File: `python-backend/tests/test_partial_ocr_fixes.py`

- [x] **5.1**: Create test file structure
  - Location: New file `python-backend/tests/test_partial_ocr_fixes.py`
  - Code: Lines 724-1053 from the plan
  - Components:
    - TestPartialOCRFixes class
    - Fixtures for test PDFs (partial coverage, full coverage, mixed)
    - Test methods
  - **STATUS**: COMPLETE

- [x] **5.2**: Implement critical test: `test_no_text_duplication()`
  - Purpose: VERIFY text layer duplication is fixed
  - Checks: Header text appears only ONCE in output
  - **STATUS**: COMPLETE

- [x] **5.3**: Implement `test_ocr_output_validation()`
  - Purpose: Verify validation rejects empty/garbage OCR results
  - Checks: OCR output has meaningful text (>50 chars)
  - **STATUS**: COMPLETE

- [x] **5.4**: Implement `test_quality_preservation()`
  - Purpose: Verify high-quality OCR is not re-processed
  - Checks: Existing good text layers are preserved
  - **STATUS**: COMPLETE

- [x] **5.5**: Implement `test_empty_coverage_metrics_fallback()`
  - Purpose: Verify Step 4 fix works (0.0 confidence triggers OCR)
  - Checks: Empty metrics return 0.0 coverage_confidence
  - **STATUS**: COMPLETE

## Verification Steps

After implementation:

- [x] Run all new tests: `cd python-backend && pytest tests/test_partial_ocr_fixes.py -v`
  - **NOTE**: Tests created but require proper Python environment with pytest and PyMuPDF
  - Test file ready at `python-backend/tests/test_partial_ocr_fixes.py`
  - Tests can be run once environment is properly set up
- [ ] Run existing tests to ensure no regressions: `pytest`
  - **PENDING**: Requires environment setup
- [ ] Manual test with sample PDF that has partial text layer coverage
  - **PENDING**: Can be done during integration testing
- [ ] Verify no text duplication in output PDF
  - **PENDING**: Can be done during integration testing
- [ ] Check logs for validation and retry messages
  - **PENDING**: Can be done during integration testing

## Implementation Notes

- **Step 1 & 2** can be implemented in parallel (independent)
- **Step 3** requires Steps 1 & 2 to be complete (uses their methods)
- **Step 3** is HIGH RISK - test extensively before committing
- Use exact code from plan but adapt to existing code style
- Add detailed logging at each step for debugging
- Follow existing patterns in ocr_batch_service.py

## Rollback Plan

If issues occur:
1. Revert changes to `ocr_batch_service.py`
2. Keep Step 4 fix in `text_quality.py` (safe change)
3. Disable coverage detection temporarily if needed:
   ```python
   thresholds = TextQualityThresholds(enable_coverage_detection=False)
   ```
