# Document De-bundling Quick Start Guide

**For developers working on the de-bundling feature**

---

## 📁 Project Structure

```
Document-De-Bundler/
├── python-backend/
│   └── services/
│       ├── cache_manager.py           ✅ SQLite database layer
│       ├── embedding_service.py       ✅ Nomic Embed v1.5
│       ├── split_detection.py         ✅ Heuristic + clustering
│       ├── phase_coordinator.py       ⏳ Pipeline orchestration
│       └── llm/
│           ├── __init__.py            ✅ Module exports
│           ├── config.py              ✅ VRAM-optimized configs
│           ├── prompts.py             ✅ Prompt templates
│           ├── loader.py              ⏳ LLM loading
│           ├── split_analyzer.py      ⏳ Split refinement
│           └── name_generator.py      ⏳ Filename generation
├── docs/
│   ├── IMPLEMENTATION_CHECKLIST.md    📋 This checklist
│   └── DEBUNDLING_QUICK_START.md      📖 This guide
└── IMPLEMENTATION_SPEC_DEBUNDLING.md  📚 Complete specification
```

**Legend**: ✅ Complete | ⏳ In Progress / Not Started

---

## 🚀 Getting Started

### Prerequisites

```bash
# Activate virtual environment
cd python-backend
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install sentence-transformers>=2.2.2
pip install scikit-learn>=1.3.0

# For LLM work (when ready):
pip install llama-cpp-python>=0.2.0
pip install huggingface-hub>=0.19.0
```

### Quick Test

```bash
# Test embedding service
python test_embedding_service.py

# Test split detection
python test_split_detection_standalone.py

# Test LLM configuration
python test_llm_simple.py
```

---

## 🔄 De-bundling Pipeline

### Phase Overview

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
┌─────────────┐  Clean memory
│ Phase 3     │
│ LLM REFINE  │  Optional: LLM Split Refinement
│ (optional)  │  → Load Phi-3 Mini (GPU: 2.3GB)
└──────┬──────┘  → Analyze ambiguous splits
       │          → Update confidence
       ↓          → Unload LLM
┌─────────────┐  Clean memory
│ Phase 4     │
│ LLM NAMING  │  Document Naming
└──────┬──────┘  → Load Phi-3 Mini (same as Phase 3)
       │          → Generate filenames
       ↓          → Unload LLM
┌─────────────┐  Clean memory
│ Phase 5     │
│ REVIEW      │  User Confirmation (UI)
└──────┬──────┘  → User edits splits/names
       │          → Confirm final plan
       ↓
┌─────────────┐
│ Phase 6     │  Split Execution
│ EXECUTE     │  → Create output PDFs
└─────────────┘  → Save to disk
```

### Memory Budget (4GB VRAM)

| Phase | VRAM | RAM | Duration (5000 pages) |
|-------|------|-----|----------------------|
| OCR | 2.5-3GB | 500MB | 8-20 min |
| Embedding | 0GB | 800MB | 2-8 min |
| Detection | 0GB | 500MB | <1 min |
| LLM Refine | 2.3GB | 1GB | 5-15 min |
| LLM Naming | 2.3GB | 1GB | 3-10 min |
| Review | 0GB | <100MB | User time |
| Execute | 0GB | <200MB | 2-5 min |

**Key Point**: No phase overlap - sequential operation ensures no resource competition.

---

## 📦 Key Components

### 1. CacheManager

**Purpose**: SQLite database for persistent storage

**Usage**:
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

### 2. EmbeddingService

**Purpose**: Generate semantic embeddings with Nomic Embed v1.5

**Usage**:
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

### 3. SplitDetector

**Purpose**: Detect document boundaries using heuristics and clustering

**Usage**:
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

### 4. LLM Configuration

**Purpose**: Select optimal LLM based on available VRAM

**Usage**:
```python
from services.llm.config import select_optimal_llm_config

config = select_optimal_llm_config()
# Auto-detects VRAM and returns optimal config

print(f"Model: {config['model_name']}")
print(f"VRAM: {config['expected_vram_gb']} GB")
print(f"Strategy: {config['offload_strategy']}")

# For 4GB VRAM system:
# {
#     'model_name': 'Phi-3 Mini (4GB Optimized)',
#     'model_id': 'microsoft/Phi-3-mini-4k-instruct-gguf',
#     'model_file': 'Phi-3-mini-4k-instruct-q4_k_m.gguf',
#     'n_gpu_layers': 28,  # 28 GPU + 4 CPU layers
#     'expected_vram_gb': 2.3,
#     'offload_strategy': 'hybrid'
# }
```

### 5. LLM Prompts

**Purpose**: Prompt templates for split refinement and naming

**Usage**:
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

## 🧪 Testing

### Run All Tests

```bash
cd python-backend

# Embedding service
python test_embedding_service.py

# Split detection
python test_split_detection_standalone.py

# LLM config and prompts
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

## 🔧 Common Tasks

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

## 🐛 Debugging Tips

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

## 📚 Additional Resources

- **Full Specification**: `IMPLEMENTATION_SPEC_DEBUNDLING.md`
- **Implementation Checklist**: `docs/IMPLEMENTATION_CHECKLIST.md`
- **OCR Documentation**: `CLAUDE.md` (section: OCR Architecture)
- **Project README**: `README.md`

---

## 🤝 Contributing

When working on this feature:

1. **Follow the checklist**: Check off items as you complete them
2. **Update documentation**: Keep this guide current
3. **Write tests**: Unit tests for each component
4. **Memory conscious**: Always cleanup resources
5. **Log appropriately**: Use logging module, not print()
6. **No emojis in code**: Only in documentation
7. **Type hints**: Use Python type hints
8. **Docstrings**: Document all public functions

---

**Questions?** Refer to `IMPLEMENTATION_SPEC_DEBUNDLING.md` for detailed specifications.

**Last Updated**: 2025-10-31