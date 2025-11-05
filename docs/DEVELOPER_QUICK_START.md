# Developer Quick Start Guide

Get up and running with Document De-Bundler development in minutes.

## Table of Contents

- [Setup](#setup)
- [OCR System](#ocr-system)
- [Document De-Bundling](#document-de-bundling)
- [Testing](#testing)
- [Common Tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)

---

## Setup

### Prerequisites

```bash
# Activate virtual environment
cd python-backend
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# For de-bundling features:
pip install sentence-transformers>=2.2.2
pip install scikit-learn>=1.3.0

# For LLM features (optional):
pip install llama-cpp-python>=0.2.0
pip install huggingface-hub>=0.19.0
```

### Quick Verification

```bash
# Test OCR system
python test_problem_document.py sample.pdf

# Test embedding service
python test_embedding_service.py

# Test split detection
python test_split_detection_standalone.py

# Test LLM configuration
python test_llm_simple.py
```

---

## OCR System

### Quick Start

```python
from services.ocr_batch_service import OCRBatchService, OCRProcessingConfig

# Configure with all improvements enabled
config = OCRProcessingConfig(
    min_alphanumeric_ratio=0.15,        # Accept receipts/forms
    min_quality_improvement_margin=0.02, # Sensitive comparison
    processing_mode="hybrid",            # Smart strategy per page
    use_coordinate_mapping=True,         # Positioned text
    enable_compression=True,             # Compress images
    image_compression_quality=85         # Quality vs size balance
)

# Process PDFs
service = OCRBatchService(config=config, use_gpu=True)
results = service.process_batch(
    file_paths=["path/to/document.pdf"],
    output_dir="output/"
)
service.cleanup()
```

### OCR Improvements Overview

| Feature | Before | After | Impact |
|---------|--------|-------|--------|
| **Alphanumeric threshold** | 30% | 15% (configurable) | Accepts receipts/forms |
| **Coverage detection** | Strict | Lenient for sparse docs | Fewer false negatives |
| **Quality margin** | 5% | 2% (configurable) | More sensitive |
| **Processing mode** | Always full rebuild | Hybrid (overlay/full/skip) | 2-5x vs 10-60x size |
| **Text positioning** | Single textbox | Per-line with bboxes | Accurate search |
| **Compression** | None | JPEG 85% quality | Smaller output |

### Configuration Presets

**Balanced (Recommended)**:
```python
config = OCRProcessingConfig(
    min_alphanumeric_ratio=0.15,
    min_quality_improvement_margin=0.02,
    processing_mode="hybrid",
    use_coordinate_mapping=True,
    enable_compression=True,
    image_compression_quality=85
)
```

**Aggressive (Maximum Acceptance)**:
```python
config = OCRProcessingConfig(
    min_alphanumeric_ratio=0.10,
    min_quality_improvement_margin=0.01,
    processing_mode="full",
    use_coordinate_mapping=True,
    enable_compression=True,
    image_compression_quality=75
)
```

**Conservative (Strict Quality)**:
```python
config = OCRProcessingConfig(
    min_alphanumeric_ratio=0.20,
    min_quality_improvement_margin=0.05,
    processing_mode="selective",
    use_coordinate_mapping=False,
    enable_compression=True,
    image_compression_quality=90
)
```

### GPU vs CPU

```python
# Auto-detect hardware
from services.ocr.config import detect_hardware_capabilities

caps = detect_hardware_capabilities()
use_gpu = caps['gpu_available'] and caps['gpu_memory_gb'] >= 4

# Initialize with appropriate backend
service = OCRBatchService(config=config, use_gpu=use_gpu)
```

**Performance (5000-page PDF)**:
- 4GB VRAM + 16GB RAM: 8-20 min (hybrid mode)
- 8GB+ VRAM: 8-15 min (GPU only)
- CPU only (16GB RAM): 40-80 min

---

## Document De-Bundling

### Architecture Overview

```
┌─────────────┐
│ Phase 0     │  OCR (if needed)
│ OCR         │  → Extract text from all pages
└──────┬──────┘  → Save to cache database
       │
       ↓ Clean memory
┌─────────────┐
│ Phase 1     │  Embedding Generation
│ EMBEDDING   │  → Load Nomic Embed v1.5 on CPU
└──────┬──────┘  → Generate 768-dim embeddings
       │          → Save to cache
       ↓ Clean memory
┌─────────────┐
│ Phase 2     │  Split Detection
│ DETECTION   │  → Run heuristic rules
└──────┬──────┘  → Run clustering (DBSCAN)
       │          → Combine signals
       ↓          → Save candidates to cache
┌─────────────┐
│ Phase 3     │  LLM Refinement (optional)
└──────┬──────┘  → Analyze ambiguous splits
       ↓
┌─────────────┐
│ Phase 4     │  Document Naming (optional)
└──────┬──────┘  → Generate intelligent names
       ↓
┌─────────────┐
│ Phase 5     │  User Review
└──────┬──────┘  → Confirm splits and names
       ↓
┌─────────────┐
│ Phase 6     │  Split Execution
└─────────────┘  → Create output PDFs
```

### Memory Budget (4GB VRAM)

| Phase | VRAM | RAM | Duration (5000 pages) |
|-------|------|-----|----------------------|
| OCR | 2.5-3GB | 500MB | 8-20 min |
| Embedding | 0GB | 800MB | 2-8 min |
| Detection | 0GB | 500MB | <1 min |
| LLM Refine | 2.3GB | 1GB | 5-15 min |
| LLM Naming | 2.3GB | 1GB | 3-10 min |
| Execute | 0GB | <200MB | 2-5 min |

**Key Point**: Sequential operation prevents resource conflicts.

### Key Components

#### 1. CacheManager

SQLite database for persistent storage.

```python
from services.cache_manager import get_cache_manager

cache = get_cache_manager()

# Create document
doc_id = cache.create_document(
    file_path="/path/to/bundle.pdf",
    total_pages=100,
    file_size_bytes=5000000
)

# Save page text
cache.save_page_text(
    doc_id=doc_id,
    page_num=0,
    text="Page content...",
    has_text_layer=True,
    ocr_method="text_layer"
)

# Save embedding
cache.save_page_embedding(doc_id, page_num=0, embedding=np.array([...]))

# Get cache stats
stats = cache.get_cache_stats()
print(f"Cache size: {stats['total_size_mb']} MB")
```

**Database Location**:
- Windows: `%APPDATA%\DocumentDeBundler\cache.db`
- macOS: `~/Library/Application Support/DocumentDeBundler/cache.db`
- Linux: `~/.local/share/DocumentDeBundler/cache.db`

#### 2. EmbeddingService

Generate semantic embeddings with Nomic Embed v1.5.

```python
from services.embedding_service import generate_embeddings_for_document

# High-level function (recommended)
success = generate_embeddings_for_document(
    doc_id="abc123",
    progress_callback=lambda curr, total, msg: print(f"{curr}/{total}: {msg}")
)

# Or use class directly
from services.embedding_service import EmbeddingService

service = EmbeddingService(device='cpu')
service.initialize()

texts = ["Page 1 text", "Page 2 text"]
embeddings = service.generate_embeddings(texts, batch_size=32)
# Returns: numpy array of shape (2, 768)

similarity = service.compute_similarity(embeddings[0], embeddings[1])
# Returns: float between 0 and 1

service.cleanup()
```

**Notes**:
- Always use CPU device (avoid GPU conflict with OCR)
- First run downloads model (~768MB)
- Embeddings are normalized for efficient similarity

#### 3. SplitDetector

Detect document boundaries using heuristics and clustering.

```python
from services.split_detection import detect_splits_for_document

num_splits = detect_splits_for_document(
    doc_id="abc123",
    progress_callback=lambda curr, total, msg: print(msg)
)

# Results saved to split_candidates table
from services.cache_manager import get_cache_manager
cache = get_cache_manager()

candidates = cache.get_split_candidates("abc123")
for split in candidates:
    print(f"Page {split['split_page']}: "
          f"confidence={split['confidence']:.2f}, "
          f"method={split['detection_method']}")
```

**Detection Methods**:
- Page number resets (confidence: 0.7-0.9)
- Header/footer changes (confidence: 0.4-0.6)
- Blank separator pages (confidence: 0.85)
- Semantic discontinuity (confidence: 0.5-0.7)
- DBSCAN clustering (confidence: 0.6)

**Confidence Tiers**:
- **High (≥0.8)**: Auto-accept, no LLM review needed
- **Medium (0.5-0.8)**: Should review with LLM
- **Low (<0.5)**: Filtered out

#### 4. LLM Configuration

Select optimal LLM based on available VRAM.

```python
from services.llm.config import select_optimal_llm_config

config = select_optimal_llm_config()
# Auto-detects VRAM and returns optimal config

print(f"Model: {config['model_name']}")
print(f"VRAM: {config['expected_vram_gb']} GB")
print(f"Strategy: {config['offload_strategy']}")
```

For 4GB VRAM:
```python
{
    'model_name': 'Phi-3 Mini (4GB Optimized)',
    'model_id': 'microsoft/Phi-3-mini-4k-instruct-gguf',
    'model_file': 'Phi-3-mini-4k-instruct-q4_k_m.gguf',
    'n_gpu_layers': 28,  # 28 GPU + 4 CPU layers
    'expected_vram_gb': 2.3,
    'offload_strategy': 'hybrid'
}
```

#### 5. LLM Prompts

Prompt templates for split refinement and naming.

```python
from services.llm.prompts import (
    format_split_prompt,
    format_naming_prompt,
    parse_split_decision,
    parse_filename
)

# Format split refinement prompt
prompt = format_split_prompt(
    split_page=12,
    before_pages=[{...}, {...}, {...}],  # Last 3 pages before
    after_pages=[{...}, {...}, {...}],   # First 3 pages after
    heuristic_signals=["Page number reset", "Header changed"]
)

# Parse LLM response
decision, reasoning = parse_split_decision(
    "YES - Page numbering resets and content topic changes"
)
# Returns: (True, "Page numbering resets and content topic changes")

# Format naming prompt
prompt = format_naming_prompt(
    start_page=1,
    end_page=5,
    first_page_text="Invoice from Acme Corp...",
    second_page_text="Line items..."
)

# Parse filename
filename = parse_filename("2024-06-15_Invoice_Acme Corp Services.pdf")
# Returns: "2024-06-15_Invoice_Acme Corp Services"
```

**Filename Format**: `{DATE}_{DOCTYPE}_{DESCRIPTION}`
- DATE: YYYY-MM-DD or "UNDATED"
- DOCTYPE: Invoice, Contract, Letter, Report, etc.
- DESCRIPTION: 2-5 words

---

## Testing

### OCR Tests

```bash
cd python-backend

# Test single document
python test_problem_document.py your_receipt.pdf

# Run OCR improvement tests
python test_ocr_improvements.py

# Check for specific issues
python check_alphanumeric_ratio.py output/result.pdf
```

**What to Look For**:
```
✓ Alphanumeric ratio: 15-25% accepted (was rejected)
✓ Strategy: "overlay" or "full" (not "full" for all)
✓ Size: 2-5x increase (was 10-60x)
✓ Searchable: All pages have text layer
```

### De-Bundling Tests

```bash
# Test embedding service
python test_embedding_service.py

# Test split detection
python test_split_detection_standalone.py

# Test LLM config and prompts
python test_llm_simple.py
```

### Create Test Database

```python
from services.cache_manager import CacheManager
from pathlib import Path

# Create test database
cache = CacheManager(db_path=Path("test_cache.db"))

# Create document
doc_id = cache.create_document(
    file_path="/test/sample.pdf",
    total_pages=10,
    file_size_bytes=1000000
)

# Add test pages
for i in range(10):
    cache.save_page_text(
        doc_id=doc_id,
        page_num=i,
        text=f"This is page {i+1} content...",
        has_text_layer=True,
        ocr_method="text_layer"
    )

print(f"Created test document: {doc_id}")
```

### Memory Testing

```python
import psutil
import gc

process = psutil.Process()

# Before operation
before_mb = process.memory_info().rss / (1024**2)

# Run operation
service.initialize()
embeddings = service.generate_embeddings(texts)
service.cleanup()

# After cleanup
gc.collect()
after_mb = process.memory_info().rss / (1024**2)

print(f"Memory used: {after_mb - before_mb:.1f} MB")
```

---

## Common Tasks

### Add New Heuristic Detector

1. Open `split_detection.py`
2. Add method to `SplitDetector` class:

```python
def detect_my_pattern(self, pages: List[Dict]) -> List[Tuple[int, float, str]]:
    """
    Detect my custom pattern.

    Returns:
        List of (page_num, confidence, reason)
    """
    splits = []

    for i in range(1, len(pages)):
        if my_condition(pages[i], pages[i-1]):
            splits.append((
                i,
                0.7,  # Confidence
                "My pattern detected"
            ))

    return splits
```

3. Call it in `detect_splits_for_document()`:

```python
all_splits.append(detector.detect_my_pattern(pages))
```

### Adjust OCR Sensitivity

```python
# More lenient (accept more documents)
config.min_alphanumeric_ratio = 0.10

# More strict (higher quality bar)
config.min_alphanumeric_ratio = 0.25

# More sensitive quality detection
config.min_quality_improvement_margin = 0.01

# Less sensitive quality detection
config.min_quality_improvement_margin = 0.05
```

### Add New LLM Configuration Tier

1. Open `llm/config.py`
2. Add to `select_optimal_llm_config()`:

```python
elif gpu_memory_gb >= 3:
    return {
        'model_name': 'My Model',
        'model_id': 'org/model-gguf',
        'model_file': 'model-q4.gguf',
        'n_gpu_layers': 24,
        'expected_vram_gb': 2.0,
        # ...
    }
```

### Extend Database Schema

1. Open `cache_manager.py`
2. Add column in `_create_schema()`:

```python
cursor.execute("""
    CREATE TABLE my_table (
        id INTEGER PRIMARY KEY,
        doc_id TEXT,
        my_data TEXT,
        FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
    )
""")
```

3. Add CRUD methods:

```python
def save_my_data(self, doc_id, data):
    with self.get_connection() as conn:
        conn.execute(
            "INSERT INTO my_table (doc_id, my_data) VALUES (?, ?)",
            (doc_id, data)
        )

def get_my_data(self, doc_id):
    with self.get_connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM my_table WHERE doc_id = ?",
            (doc_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
```

4. Increment `DB_VERSION` and add migration logic

---

## Troubleshooting

### OCR Issues

**"Low alphanumeric ratio" still failing**
```python
# Solution: Lower threshold
config.min_alphanumeric_ratio = 0.10  # or even 0.05
```

**PDF still too large**
```python
# Check:
# 1. Strategy distribution in logs
# 2. Compression enabled
config.enable_compression = True
config.image_compression_quality = 70  # Lower quality = smaller size
```

**Search not highlighting correctly**
```python
# Check:
config.use_coordinate_mapping = True  # Must be enabled
# Check logs for "No bounding boxes available"
```

### Database Issues

```bash
# Inspect database
sqlite3 %APPDATA%\DocumentDeBundler\cache.db

# List tables
.tables

# Check document status
SELECT doc_id, original_filename, processing_status
FROM documents;

# Check splits
SELECT split_page, confidence, detection_method
FROM split_candidates
WHERE doc_id = 'your-doc-id';

# Check processing log
SELECT phase, status, timestamp
FROM processing_log
WHERE doc_id = 'your-doc-id'
ORDER BY timestamp DESC;
```

### Memory Issues

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor VRAM
from services.ocr.vram_monitor import VRAMMonitor
monitor = VRAMMonitor()
info = monitor.get_info()
print(f"VRAM: {info['used_gb']:.1f}/{info['total_gb']:.1f} GB")

# Force cleanup
import gc
gc.collect()

import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Embedding Issues

```python
# Check embedding shape
embedding = cache.get_page_embedding(doc_id, page_num=0)
print(f"Shape: {embedding.shape}")  # Should be (768,)
print(f"Type: {embedding.dtype}")   # Should be float32

# Verify similarity range
sim = service.compute_similarity(emb1, emb2)
assert 0 <= sim <= 1, f"Invalid similarity: {sim}"
```

---

## Key Files

### OCR System
- `services/ocr_batch_service.py` - Main OCR processing logic
- `services/ocr_service.py` - OCR facade and engine management
- `services/ocr/` - OCR abstraction layer
  - `base.py` - Abstract engine interface
  - `config.py` - Hardware detection
  - `manager.py` - Engine factory
  - `engines/paddleocr_engine.py` - PaddleOCR implementation
  - `engines/tesseract_engine.py` - Tesseract implementation
  - `text_quality.py` - Quality analysis
  - `coordinate_mapper.py` - Coordinate transforms
- `services/pdf_text_layer_analyzer.py` - Page classification

### De-Bundling System
- `services/cache_manager.py` - SQLite database layer
- `services/embedding_service.py` - Nomic Embed v1.5
- `services/split_detection.py` - Heuristic + clustering
- `services/llm/` - LLM integration
  - `config.py` - VRAM-optimized configs
  - `prompts.py` - Prompt templates

---

## Pro Tips

1. **Monitor strategy usage**: Check logs for "overlay" vs "full" counts
2. **Adjust per document type**: Use different configs for receipts vs reports
3. **Benchmark before/after**: Run same PDF through old and new code
4. **Test searchability**: Open PDF and try searching for known text
5. **Check file sizes**: `ls -lh output/` to see actual sizes
6. **Follow the checklist**: Check off items as you complete them
7. **Update documentation**: Keep docs current with code changes
8. **Write tests**: Unit tests for each component
9. **Memory conscious**: Always cleanup resources (use context managers)
10. **Log appropriately**: Use logging module, not print()
11. **Type hints**: Use Python type hints
12. **Docstrings**: Document all public functions

---

## See Also

- [Architecture Overview](ARCHITECTURE.md) - System design
- [Implementation Status](IMPLEMENTATION_STATUS.md) - Feature completion
- [OCR System](features/LLM_INTEGRATION.md) - Detailed OCR documentation
- [De-Bundling Guide](features/DEBUNDLING_QUICK_START.md) - In-depth de-bundling guide
- [Test Results](testing/TEST_RESULTS_2025-11-01.md) - Latest test status

---

**Last Updated**: 2025-11-03
**Status**: All major features implemented and tested
