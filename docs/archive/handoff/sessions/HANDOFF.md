# Project Handoff Document
**Date**: 2025-10-15
**Project**: Document De-Bundler - Advanced OCR Processing Implementation
**Phase**: Phase 1 - Enhanced Text Layer Detection (COMPLETED)

---

## Executive Summary

We are implementing advanced OCR processing features for the Document De-Bundler application, a Tauri-based desktop app for processing large PDFs (up to 5GB). This handoff covers the completion of **Phase 1: Enhanced Text Layer Detection** and outlines the next phases.

**Critical Project Constraint**: This project MUST be completely self-contained with no global dependencies (see project memory in serena MCP).

---

## What Was Just Completed

### Phase 1: Enhanced Text Layer Detection ✅

Successfully implemented robust text quality validation to prevent false positives when PDFs have garbage/artifact text instead of valid OCR.

#### Files Created/Modified:

1. **`python-backend/services/ocr/text_quality.py`** (NEW - 387 lines)
   - **TextQualityMetrics**: Dataclass with 11 quality assessment fields
   - **TextQualityThresholds**: Configurable validation thresholds
   - **TextLayerValidator**: Main validation class with 9-factor scoring system
   - **OCRDecisionEngine**: Intelligent decision engine for OCR vs text layer extraction

   **Key Features**:
   - 9-factor quality scoring (printable ratio, alphanumeric ratio, whitespace, word length, common words, unicode errors, text density, sentence markers, word count)
   - Weighted confidence scoring (0-1 scale)
   - Common English words dictionary (60+ words) for language sanity
   - Unicode error detection for corrupted encodings
   - Configurable thresholds: normal (65% confidence) vs strict (80% confidence)
   - Human-readable quality reports
   - User preference system (auto OCR, force OCR, preview mode)

2. **`python-backend/services/pdf_processor.py`** (UPDATED)
   - Added imports for text quality validation components
   - Updated `__init__()` to initialize TextLayerValidator and OCRDecisionEngine
   - Replaced naive `has_text_layer()` with `has_valid_text_layer()` using comprehensive validation
   - Enhanced `analyze_structure()` with quality metrics and distributed sampling (10 pages)
   - Completely rewrote `process_pages_with_ocr()` to use OCRDecisionEngine for intelligent decisions

   **Key Changes**:
   ```python
   # Old naive approach (line 44-51):
   def has_text_layer(self, page_num: int) -> bool:
       text = page.get_text().strip()
       return len(text) > 50  # Arbitrary threshold

   # New robust approach:
   def has_valid_text_layer(self, page_num: int, return_metrics: bool = False) -> Tuple[bool, Optional[TextQualityMetrics]]:
       is_valid, metrics = self.text_validator.has_valid_text_layer(
           page, require_high_confidence=self.use_strict_validation
       )
       # Returns comprehensive quality metrics with 9-factor validation
   ```

   **Return Format Changed**:
   - `process_pages_with_ocr()` now returns `List[Dict[str, Any]]` instead of `List[str]`
   - Each result contains: `page_num`, `text`, `method` (text_layer/ocr), `quality_metrics`, `reason`
   - Added `force_ocr` parameter for user override

---

## Current Architecture Overview

### Technology Stack
- **Frontend**: Svelte + Vite + TypeScript + TailwindCSS
- **Desktop Framework**: Tauri (Rust)
- **Backend**: Python 3.8+
- **PDF Processing**: PyMuPDF (fitz)
- **OCR**: PaddleOCR (primary, GPU auto-detect) + Tesseract (fallback)
- **IPC**: JSON over stdin/stdout between Rust and Python

### OCR Engine Architecture (Implemented)

```
┌─────────────────────────────────────────────────────────┐
│                    OCR Manager                          │
│  (Facade with engine selection & fallback logic)       │
└─────────────────────────────────────────────────────────┘
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
┌──────────────────────┐    ┌──────────────────────┐
│  PaddleOCR Engine    │    │  Tesseract Engine    │
│  (Primary)           │    │  (Fallback)          │
│  - GPU auto-detect   │    │  - CPU only          │
│  - 95-98% accuracy   │    │  - System-installed  │
│  - 2-3x faster       │    │  - Lightweight       │
└──────────────────────┘    └──────────────────────┘
```

**Key Files**:
- `python-backend/services/ocr/base.py` - Abstract OCR interface
- `python-backend/services/ocr/config.py` - Hardware detection
- `python-backend/services/ocr/manager.py` - Engine factory with fallback
- `python-backend/services/ocr/engines/paddleocr_engine.py` - PaddleOCR implementation
- `python-backend/services/ocr/engines/tesseract_engine.py` - Tesseract implementation

### Text Quality Validation Flow

```
PDF Page → TextLayerValidator.has_valid_text_layer()
              │
              ├─ Extract text from page
              ├─ Calculate 9 quality metrics
              ├─ Compute weighted confidence score (0-1)
              └─ Compare against thresholds
                    │
                    ├─ Valid (≥65% confidence) → Use text layer
                    └─ Invalid (<65%) → Perform OCR
```

**9 Quality Metrics**:
1. **Printable ratio** (≥85% expected)
2. **Alphanumeric ratio** (≥60% expected)
3. **Whitespace ratio** (≤40% expected, ideal 10-25%)
4. **Average word length** (2-20 chars, ideal 4-7)
5. **Word count** (≥10 expected)
6. **Common words found** (≥10% of words should be common English words)
7. **Sentence markers** (periods, question marks, exclamation points)
8. **Unicode errors** (replacement chars, null chars, excessive control chars)
9. **Text density** (chars per page area, ≥0.001 expected)

**Weighted Confidence Calculation**:
- Printable ratio: 15%
- Alphanumeric ratio: 15%
- Whitespace ratio: 10%
- Word length: 15%
- Word count: 10%
- Common words: 15%
- Unicode errors: 10%
- Text density: 10%

---

## Remaining Tasks (7 of 10 total)

### Phase 2: Memory Management & Checkpointing (NEXT UP)

**Task #4**: Create `memory_manager.py` with adaptive batch sizing
- **Location**: `python-backend/services/ocr/memory_manager.py`
- **Purpose**: Monitor system memory and dynamically adjust batch size for 5GB PDFs
- **Requirements**:
  - Use `psutil` to monitor available RAM
  - Adaptive batch sizing: 1-50 pages based on available memory
  - Emergency fallbacks: reduce DPI, reduce batch size
  - Disk caching for batches >1GB (use temp directory)
  - Memory pressure detection (warn at 80%, emergency at 90%)

**Design Guidance from Architect**:
```python
class MemoryManager:
    """Adaptive memory management for large PDF processing"""

    def get_optimal_batch_size(
        self,
        page_size_estimate: int,  # bytes per page
        dpi: int = 300
    ) -> int:
        """
        Calculate optimal batch size based on available memory.

        Formula:
        - Get available RAM (psutil.virtual_memory().available)
        - Reserve 2GB for system
        - Calculate: batch_size = (available - 2GB) / (page_size * DPI_factor)
        - Clamp between 1 and 50
        """

    def should_use_disk_cache(self, batch_size_bytes: int) -> bool:
        """Return True if batch exceeds 1GB"""

    def check_memory_pressure(self) -> Tuple[bool, str]:
        """
        Check if system is under memory pressure.
        Returns (is_critical, message)
        """
```

**Task #5**: Create `checkpoint.py` for resume capability
- **Location**: `python-backend/services/ocr/checkpoint.py`
- **Purpose**: Save processing state every 100 pages for crash recovery
- **Requirements**:
  - JSON-based checkpoint format
  - Store: current page, processed results, configuration
  - Auto-resume detection on startup
  - Checkpoint validation (detect corrupted checkpoints)

**Design Guidance**:
```python
@dataclass
class ProcessingCheckpoint:
    file_path: str
    total_pages: int
    last_processed_page: int
    results: List[Dict[str, Any]]  # Matches new pdf_processor format
    config: Dict[str, Any]
    timestamp: str

class CheckpointManager:
    def save_checkpoint(self, checkpoint: ProcessingCheckpoint) -> None
    def load_checkpoint(self, file_path: str) -> Optional[ProcessingCheckpoint]
    def validate_checkpoint(self, checkpoint: ProcessingCheckpoint) -> bool
    def clear_checkpoint(self, file_path: str) -> None
```

**Task #6**: Implement `AdaptiveOCRProcessor` with memory monitoring
- **Location**: `python-backend/services/ocr/adaptive_processor.py`
- **Purpose**: Orchestrate OCR processing with memory management and checkpointing
- **Requirements**:
  - Integrate MemoryManager and CheckpointManager
  - Progress reporting via callbacks
  - Automatic batch size adjustment during processing
  - Graceful degradation (reduce DPI if memory pressure detected)

### Phase 3: Searchable PDF Creation

**Task #7**: Create `searchable_pdf.py` for invisible text layer creation
- **Location**: `python-backend/services/ocr/searchable_pdf.py`
- **Purpose**: Create searchable PDFs with invisible OCR text overlay
- **Requirements**:
  - Use PyMuPDF render mode 3 (invisible text)
  - Transform coordinates from image space to PDF space
  - Position text at exact OCR bounding box locations
  - Preserve original PDF appearance

**Design Guidance from Architect**:
```python
class SearchablePDFCreator:
    """Create searchable PDFs with invisible OCR text layer"""

    def add_invisible_text_layer(
        self,
        page: fitz.Page,
        ocr_results: List[Dict[str, Any]]  # text, bbox coordinates
    ) -> None:
        """
        Add invisible text overlay to page.

        Coordinate transformation:
        - OCR returns pixel coordinates (x, y, w, h)
        - PDF uses points (1 inch = 72 points)
        - Transform: pdf_x = (pixel_x / image_width) * page_width
        """

    def create_searchable_pdf(
        self,
        input_pdf: str,
        output_pdf: str,
        ocr_results: List[Dict]  # per-page OCR with bounding boxes
    ) -> None:
        """Process entire PDF and create searchable version"""
```

**Task #8**: Update OCR pipeline to return bounding boxes with text
- **Files to Modify**:
  - `python-backend/services/ocr/base.py` - OCRResult already has `bbox` field
  - `python-backend/services/ocr/engines/paddleocr_engine.py` - Extract bbox from PaddleOCR results
  - `python-backend/services/ocr/engines/tesseract_engine.py` - Use pytesseract.image_to_data() for bbox
  - `python-backend/services/pdf_processor.py` - Store bbox in results

**Task #9**: Add output format configuration (searchable/text/both)
- **Files to Modify**:
  - `python-backend/services/pdf_processor.py` - Add `output_format` parameter
  - `python-backend/services/ocr_service.py` - Handle different output formats
- **Output Options**:
  - `searchable_pdf_only`: Create searchable PDF with invisible text layer
  - `text_files_only`: Extract text to .txt files (current behavior)
  - `both`: Create searchable PDF + text files

### Phase 4: Documentation

**Task #10**: Update `CLAUDE.md` with new OCR processing architecture
- **Sections to Add**:
  - Text Quality Validation system
  - Memory Management architecture
  - Checkpoint/Resume capability
  - Searchable PDF creation
  - Usage examples for all new features

---

## Important Code Patterns & Decisions

### 1. Text Quality Validation Usage

**Before** (naive):
```python
if len(page.get_text().strip()) > 50:
    text = page.get_text()  # Use text layer
else:
    text = ocr_service.process_page(page)  # OCR
```

**After** (robust):
```python
validator = TextLayerValidator()
is_valid, metrics = validator.has_valid_text_layer(page)

if is_valid:
    text = page.get_text()
    logger.info(f"Valid text layer (confidence: {metrics.confidence_score:.2%})")
else:
    text = ocr_service.process_page(page)
    logger.info(f"OCR needed: {metrics.confidence_score:.2%}")
```

### 2. OCRDecisionEngine Usage

```python
engine = OCRDecisionEngine()
should_ocr, reason, metrics = engine.should_perform_ocr(page)

if should_ocr:
    text = ocr_service.process_page(page)
    logger.debug(f"Page {page_num}: {reason}")
else:
    text = page.get_text()
    logger.debug(f"Page {page_num}: {reason}")
```

### 3. New process_pages_with_ocr() Return Format

**Old** (simple list):
```python
results: List[str] = ["text from page 1", "text from page 2", ...]
```

**New** (rich metadata):
```python
results: List[Dict[str, Any]] = [
    {
        'page_num': 0,
        'text': "extracted text",
        'method': 'text_layer',  # or 'ocr'
        'quality_metrics': TextQualityMetrics(...),
        'reason': "Valid text layer (confidence: 87.3%)"
    },
    ...
]
```

**IMPORTANT**: Any code calling `process_pages_with_ocr()` needs to be updated to handle the new format!

---

## Key Files Reference

### Completed Files (Phase 1)
- ✅ `python-backend/services/ocr/text_quality.py` - Text quality validation (NEW)
- ✅ `python-backend/services/pdf_processor.py` - Updated with validation (MODIFIED)

### Files to Create (Phase 2 & 3)
- ⏳ `python-backend/services/ocr/memory_manager.py` - Adaptive memory management
- ⏳ `python-backend/services/ocr/checkpoint.py` - Checkpoint/resume system
- ⏳ `python-backend/services/ocr/adaptive_processor.py` - Orchestration layer
- ⏳ `python-backend/services/ocr/searchable_pdf.py` - Searchable PDF creation

### Files to Modify (Phase 2 & 3)
- ⏳ `python-backend/services/ocr/base.py` - Ensure bbox support
- ⏳ `python-backend/services/ocr/engines/paddleocr_engine.py` - Extract bounding boxes
- ⏳ `python-backend/services/ocr/engines/tesseract_engine.py` - Extract bounding boxes
- ⏳ `python-backend/services/ocr_service.py` - Output format handling
- ⏳ `CLAUDE.md` - Documentation updates

### Existing Architecture Files (Do Not Modify)
- `python-backend/services/ocr/base.py` - OCR interface (already has bbox support)
- `python-backend/services/ocr/config.py` - Hardware detection
- `python-backend/services/ocr/manager.py` - Engine factory
- `python-backend/services/ocr/engines/paddleocr_engine.py` - PaddleOCR implementation
- `python-backend/services/ocr/engines/tesseract_engine.py` - Tesseract implementation

---

## Testing Considerations

### Testing Text Quality Validation

Create test cases for:
1. **Valid text**: Normal PDF with good embedded text
2. **Garbage text**: PDF with artifact/corrupted text (confidence <65%)
3. **Scanned pages**: No text layer (confidence = 0%)
4. **Partially valid**: Mixed quality pages
5. **Edge cases**: Empty pages, single words, very dense text

### Testing Memory Management

Test scenarios:
1. **Small PDF** (<100 pages): Should use large batches (20-50 pages)
2. **Large PDF** (1000+ pages): Should use adaptive batching
3. **Low memory** (simulated): Should reduce batch size and DPI
4. **Memory pressure**: Should trigger disk caching

### Testing Checkpointing

Test scenarios:
1. **Normal completion**: Checkpoint should be deleted
2. **Interrupted processing**: Should resume from last checkpoint
3. **Corrupted checkpoint**: Should start fresh with warning
4. **Different file version**: Should detect mismatch and start fresh

---

## Known Issues & Gotchas

### 1. Breaking Change in process_pages_with_ocr()

The return type changed from `List[str]` to `List[Dict[str, Any]]`. Any calling code needs updates:

```python
# Old code (WILL BREAK):
texts = processor.process_pages_with_ocr(ocr_service)
for text in texts:
    print(text)

# New code (CORRECT):
results = processor.process_pages_with_ocr(ocr_service)
for result in results:
    print(result['text'])
    print(f"Method: {result['method']}, Confidence: {result.get('quality_metrics')}")
```

### 2. TextLayerValidator Requires PyMuPDF Page Object

The validator expects a `fitz.Page` object, not just text:

```python
# WRONG:
text = page.get_text()
is_valid, metrics = validator.has_valid_text_layer(text)  # ERROR

# CORRECT:
is_valid, metrics = validator.has_valid_text_layer(page)  # Pass page object
```

### 3. Common English Words Dictionary

The current implementation uses a 60-word English dictionary. For multi-language support:
- Consider making `COMMON_WORDS` configurable per language
- Or use language detection libraries (langdetect, polyglot)

### 4. Memory Estimates

For 5GB PDFs at 300 DPI:
- **Assumptions**: 5000 pages, each page ~300KB when rendered
- **Memory per page**: ~1MB (300KB image + overhead)
- **Batch of 50 pages**: ~50MB
- **Safe batch size**: Available_RAM / 10 (leave 90% free)

---

## Architecture Decisions Log

### Decision 1: PaddleOCR over EasyOCR
- **Reason**: 2-3x faster, 50% less memory, better accuracy
- **Date**: Early in project
- **Impact**: Affects all OCR processing

### Decision 2: Hybrid Text Layer + OCR
- **Reason**: Many PDFs have valid embedded text; OCR only when needed
- **Date**: Early in project
- **Impact**: Massive performance improvement for documents with text layers

### Decision 3: 9-Factor Quality Validation
- **Reason**: Simple character count was causing false positives
- **Date**: Recent (this session)
- **Impact**: Solves garbage text problem, more accurate OCR decisions

### Decision 4: Weighted Confidence Scoring
- **Reason**: Not all quality factors are equally important
- **Date**: Recent (this session)
- **Impact**: More nuanced quality assessment

### Decision 5: 65% vs 80% Confidence Thresholds
- **Reason**: 65% for normal mode (faster), 80% for strict mode (more accurate)
- **Date**: Recent (this session)
- **Impact**: Gives users flexibility between speed and accuracy

---

## Next Immediate Actions

**Start Here** when continuing:

1. **Read this handoff document** completely
2. **Review the project memory** in serena MCP about self-contained dependencies
3. **Read the updated files**:
   - `python-backend/services/ocr/text_quality.py`
   - `python-backend/services/pdf_processor.py`
4. **Start Task #4**: Create `memory_manager.py`
   - Use the design guidance above
   - Follow the psutil-based memory monitoring pattern
   - Implement adaptive batch sizing (1-50 pages)
   - Test with memory pressure scenarios

---

## Questions for User (If Clarification Needed)

- **Memory limits**: Should we enforce a hard memory limit (e.g., 4GB max for OCR)?
- **Checkpoint location**: Where to store checkpoints? Temp directory or alongside PDF?
- **Searchable PDF preference**: Default output format (searchable_pdf, text_files, or both)?
- **Language support**: Should text quality validation support multiple languages?

---

## Resources & References

### Documentation Links
- PyMuPDF (fitz): https://pymupdf.readthedocs.io/
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- psutil: https://psutil.readthedocs.io/
- Tauri: https://tauri.app/

### Project Files
- Main README: `F:\Document-De-Bundler\README.md`
- Development Guidelines: `F:\Document-De-Bundler\CLAUDE.md`
- Project Structure: See README.md, section "Project Structure"

### Key Conversations
- Architect discussion on memory management (archived)
- Architect discussion on text quality validation (archived)
- Architect discussion on searchable PDF creation (archived)

---

## Contact & Continuity

This handoff document should provide complete context for continuing the implementation. If you have questions about design decisions or need clarification, refer to:

1. The architect discussions (summarized in this document)
2. The project memory in serena MCP (self-contained dependencies)
3. The existing implementation in text_quality.py (comprehensive example)

**Good luck with Phase 2!** 🚀

---

_Last Updated: 2025-10-15_
_Phase 1 Status: COMPLETED ✅_
_Next Phase: Memory Management & Checkpointing_
