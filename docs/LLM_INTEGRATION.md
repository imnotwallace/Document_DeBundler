# LLM Integration for Document De-Bundler

## Overview

Document De-Bundler integrates **llama.cpp** to provide AI-powered features for intelligent document processing. The integration enables:

- **LLM-Powered Split Refinement**: Validates DBSCAN split candidates using natural language understanding
- **Intelligent Document Naming**: Generates structured filenames (`{DATE}_{DOCTYPE}_{DESCRIPTION}`) based on content analysis
- **Local-First AI**: All LLM processing runs entirely locally without cloud dependencies
- **GPU Acceleration**: Optimized for 4GB VRAM systems with CPU fallback

### Model Selection

Two lightweight models are supported:

1. **Phi-3 Mini 4K Instruct Q4_K_M** (~2.3GB)
   - Microsoft's compact LLM, optimized for instruction-following
   - Best balance of quality and speed
   - **Recommended for 4GB VRAM systems**

2. **Gemma 2 2B Instruct Q4_K_M** (~1.5GB)
   - Google's ultra-compact LLM
   - Faster inference, lower memory footprint
   - Good for CPU-only or <4GB VRAM systems

Both models use **Q4_K_M quantization** (4-bit) for optimal performance on consumer hardware.

---

## Installation

### Prerequisites

**System Requirements**:
- **Minimum**: 4GB VRAM + 8GB RAM (GPU mode) OR 16GB RAM (CPU mode)
- **Recommended**: 4GB VRAM + 16GB RAM (hybrid mode with CPU fallback)
- **Python**: 3.10+ with virtual environment
- **GPU** (optional): CUDA 11.8+ (NVIDIA) or DirectML (AMD/Intel on Windows)

**Dependencies** (installed via requirements.txt):
- `llama-cpp-python==0.3.4` - Python bindings for llama.cpp
- `huggingface-hub` - Model downloading from HuggingFace

### Model Download

**Option 1: Interactive Downloader (Recommended)**
```bash
cd python-backend
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Linux/macOS

python download_llm_models.py
```

Follow the interactive prompts:
1. Select model (1 = Phi-3 Mini, 2 = Gemma 2 2B, 3 = Both)
2. Confirm download size (~2.3GB or ~1.5GB)
3. Wait for download with progress tracking
4. Models saved to `python-backend/models/llm/`

**Option 2: Manual Download**
```bash
# From HuggingFace manually, then place in:
# python-backend/models/llm/phi-3-mini-4k-instruct-q4.gguf
# python-backend/models/llm/gemma-2-2b-instruct-q4.gguf
```

**Option 3: Auto-Download (First Run)**
- Models auto-download on first use if not found locally
- Slower first-run experience but no pre-setup needed

### Verify Installation

```bash
cd python-backend
.venv\Scripts\activate

# Check model availability
python -c "from services.resource_path import verify_llm_models; print(verify_llm_models())"
# Expected: {'phi3_mini': True, 'gemma2_2b': True}

# Test model loading
python -c "from services.llm.loader import test_loader; test_loader()"
# Expected: "✓ Model loaded: phi-3-mini"
```

### First-Time Setup

1. **Install llama-cpp-python**:
   ```bash
   cd python-backend
   .venv\Scripts\activate
   pip install llama-cpp-python==0.3.4
   ```

   For GPU support (NVIDIA CUDA):
   ```bash
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.3.4
   ```

   For GPU support (AMD/Intel DirectML on Windows):
   ```bash
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/directml
   ```

2. **Download Models** (see above)

3. **Configure Settings** (optional):
   ```python
   from services.llm.settings import get_settings_manager
   
   manager = get_settings_manager()
   settings = manager.get()
   
   # Customize settings
   settings.enabled = True
   settings.split_refinement_enabled = True
   settings.naming_enabled = True
   settings.use_gpu = True
   settings.model_preference = "phi3_mini"  # or "gemma2_2b"
   
   manager.save(settings)
   ```

---

## Usage

### Basic Usage

**Split Refinement** (validate DBSCAN candidates):
```python
from services.split_detection import detect_splits_for_document

# Automatically uses LLM refinement if enabled
split_count = detect_splits_for_document(
    doc_id="abc123",
    use_llm_refinement=True,  # Enable LLM validation
    progress_callback=lambda c, t, m: print(f"{m} ({c}/{t})")
)
```

**Intelligent Naming** (generate structured filenames):
```python
from services.naming_service import NamingService

naming_service = NamingService()

# LLM-powered naming
filename = naming_service.suggest_name(
    text_content=first_page_text,
    page_num=1,
    fallback_prefix="document",
    second_page_text=second_page_text,  # Optional context
    use_llm=True  # Enable LLM naming
)

# Example output: "2024-01-15_Invoice_Acme Corp Consulting"
```

**Manual LLM Control**:
```python
from services.llm.manager import get_llm_manager, cleanup_llm

# Initialize LLM
manager = get_llm_manager()
manager.initialize()

# Generate text
response = manager.generate(
    prompt="Analyze this document...",
    task_type="split_analysis",  # or "naming"
    max_tokens=100
)

# Cleanup when done
cleanup_llm()
```

### Advanced Configuration

**Customize LLM Settings**:
```python
from services.llm.settings import get_settings_manager, LLMSettings

manager = get_settings_manager()
settings = manager.get()

# Performance tuning
settings.use_gpu = True
settings.n_gpu_layers = 28  # For 4GB VRAM (28 GPU + 4 CPU layers)
settings.n_ctx = 2048  # Context window size
settings.n_batch = 512  # Batch size for prompt processing

# Quality thresholds
settings.split_confidence_threshold = 0.7  # Minimum confidence for LLM splits
settings.naming_fallback_on_failure = True  # Use heuristics if LLM fails

# Memory management
settings.auto_cleanup_enabled = True  # Auto-cleanup after processing
settings.max_concurrent_requests = 1  # Process one at a time

manager.save(settings)
```

**Batch Processing**:
```python
from services.llm.split_analyzer import SplitAnalyzer

analyzer = SplitAnalyzer()

# Analyze multiple splits in batch
splits_to_check = [
    {"page": 15, "texts": page_texts_15, "signals": [...]},
    {"page": 32, "texts": page_texts_32, "signals": [...]},
    {"page": 67, "texts": page_texts_67, "signals": [...]},
]

results = analyzer.analyze_batch(
    splits_to_check,
    progress_callback=lambda c, t, m: print(f"{m}")
)

for result in results:
    print(f"Page {result['page']}: {result['decision']} (confidence: {result['confidence']})")
```

---

## Architecture

### Components

```
services/llm/
├── config.py           # VRAM-based model selection
├── prompts.py          # Prompt templates
├── loader.py           # Model loading (llama-cpp-python + binary fallback)
├── manager.py          # Singleton lifecycle manager
├── split_analyzer.py   # LLM split refinement
├── name_generator.py   # Intelligent document naming
└── settings.py         # User configuration management
```

**Loader** (`loader.py`):
- Dual integration: llama-cpp-python (primary) + standalone binary (fallback)
- GPU layer optimization (28 GPU + 4 CPU for 4GB VRAM)
- Memory monitoring and cleanup

**Manager** (`manager.py`):
- Lazy loading (initialize only when needed)
- Thread-safe generation with queuing
- Statistics tracking
- Auto cleanup and memory management

**Split Analyzer** (`split_analyzer.py`):
- Analyzes DBSCAN candidates using LLM
- Provides YES/NO decisions with reasoning and confidence scoring
- Caching support for LLM decisions

**Name Generator** (`name_generator.py`):
- Generates `{DATE}_{DOCTYPE}_{DESCRIPTION}` format
- Extracts dates, document types, and descriptions
- Validation and auto-fixing of malformed names

**Settings Manager** (`settings.py`):
- Feature toggles (enable/disable LLM, split refinement, naming)
- Performance tuning (GPU layers, context size, batch size)
- Quality thresholds (confidence, fallback behavior)

### Data Flow

```
PDF Processing Pipeline:
    ↓
1. OCR Phase (GPU)
    → Extract text from all pages
    → Cleanup GPU memory
    ↓
2. Embedding Phase (GPU)
    → Generate semantic embeddings
    → Cleanup GPU memory
    ↓
3. Split Detection (DBSCAN)
    → Cluster pages by similarity
    → Identify split candidates
    ↓
3a. LLM Split Refinement (GPU) [OPTIONAL]
    → Load LLM model
    → Analyze each split candidate
    → Filter by LLM decision + confidence
    → Cleanup LLM from GPU
    ↓
4. Document Extraction
    → Split PDF at validated points
    ↓
4a. LLM Naming (GPU) [OPTIONAL]
    → Load LLM model
    → Generate intelligent names
    → Cleanup LLM from GPU
    ↓
5. Output
```

**Sequential GPU Processing** (prevents VRAM conflicts):
- Each GPU-heavy phase completes fully before the next
- Explicit cleanup (`cleanup()`) between phases
- Garbage collection to free memory
- No concurrent GPU operations

---

## Performance

### Benchmarks

**Processing Time** (5000-page bundled PDF, 4GB VRAM + 16GB RAM):
- **Without LLM**: ~8-15 minutes
- **With LLM Refinement**: +20-40 seconds (5-10 split candidates)
- **With LLM Naming**: +30-60 seconds (6-12 documents)
- **Total with Both**: ~10-18 minutes

**Split Refinement** (~20s overhead for typical document):
- DBSCAN detects 5-10 split candidates
- LLM analyzes each in ~2-4 seconds
- 70-90% of candidates confirmed as valid splits
- Reduces false positives by 40-60%

**Intelligent Naming** (~5s per document):
- LLM analyzes first 1-2 pages
- Extracts date, document type, description
- Generates structured filename
- 85-95% success rate (falls back to heuristics on failure)

### Memory Requirements

**GPU Mode (4GB VRAM)**:
- Model loading: 2.3GB (Phi-3 Mini) or 1.5GB (Gemma 2 2B)
- Prompt processing: ~200-500MB overhead
- **Total peak**: ~2.5-3GB VRAM usage
- **Safe for 4GB VRAM systems**

**CPU Mode (16GB RAM)**:
- Model loading: 2.3GB (Phi-3 Mini) or 1.5GB (Gemma 2 2B)
- Prompt processing: ~500MB-1GB overhead
- **Total peak**: ~3-4GB RAM usage

**Hybrid Mode (4GB VRAM + 16GB RAM)**:
- Automatically offloads to CPU if GPU memory pressure detected
- Seamless switching between GPU and CPU
- Best performance on constrained hardware

### Memory Management

**Automatic Cleanup**:
```python
# Cleanup triggered automatically at end of processing
# if settings.auto_cleanup_enabled = True

# Or manual cleanup:
from services.llm.manager import cleanup_llm
cleanup_llm()

# Aggressive memory cleanup:
import gc
import torch
gc.collect()
torch.cuda.empty_cache()  # If GPU
```

**Cache Management**:
- LLM decisions cached by content hash (MD5)
- Cached decisions reused for identical splits
- Cache stored in memory during session
- Cleared on cleanup

---

## Troubleshooting

### Common Issues

**Models Won't Download**:
- Check internet connection
- Try manual download from HuggingFace:
  - https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
  - https://huggingface.co/google/gemma-2-2b-it-gguf
- Verify `huggingface-hub` is installed: `pip install huggingface-hub`

**llama-cpp-python Won't Install**:
- May need Visual Studio Build Tools on Windows
- Try pre-built wheels:
  ```bash
  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
  ```
- For CPU-only:
  ```bash
  pip install llama-cpp-python
  ```

**GPU Not Being Used**:
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify VRAM >= 2GB: `nvidia-smi` (NVIDIA) or Task Manager (Windows)
- Check logs for "GPU: enabled" message
- Ensure llama-cpp-python installed with CUDA support

**Generation is Slow**:
- Verify GPU is being used (check logs for "GPU: enabled")
- Check `n_gpu_layers` in settings (should be 28 for 4GB VRAM)
- Try CPU mode to confirm it's slower (validates GPU is working)
- Reduce `n_ctx` (context size) if too large

**Poor Quality Results**:
- Check prompt templates in `services/llm/prompts.py`
- Increase `max_tokens` for more detailed responses
- Try different model: Phi-3 Mini (better quality) vs Gemma 2 2B (faster)
- Verify input text quality (OCR accuracy affects LLM performance)

**Out of Memory Errors**:
- Reduce `n_gpu_layers` (e.g., 20 instead of 28)
- Use Gemma 2 2B instead of Phi-3 Mini (smaller footprint)
- Enable CPU mode: `settings.use_gpu = False`
- Ensure sequential processing (cleanup between phases)

---

## API Reference

### Key Functions

**Model Loading**:
```python
from services.llm.loader import LlamaLoader

with LlamaLoader(use_gpu=True) as loader:
    if loader.load_model():
        response = loader.generate("prompt", max_tokens=50)
```

**Manager**:
```python
from services.llm.manager import get_llm_manager, cleanup_llm

manager = get_llm_manager()
manager.initialize()
response = manager.generate("prompt", task_type="general")
cleanup_llm()
```

**Split Analysis**:
```python
from services.llm.split_analyzer import SplitAnalyzer

analyzer = SplitAnalyzer()
should_split, confidence, reasoning = analyzer.analyze_split(
    split_page=15,
    page_texts=[...],
    heuristic_signals=["Low similarity (0.15)"]
)
```

**Naming**:
```python
from services.llm.name_generator import NameGenerator

generator = NameGenerator()
filename = generator.generate_name(
    first_page_text="...",
    start_page=0,
    end_page=10,
    second_page_text="..."  # Optional
)
```

**Settings**:
```python
from services.llm.settings import get_settings_manager

manager = get_settings_manager()
settings = manager.get()
settings.enabled = True
manager.save(settings)
```

### Key Classes

**LLMSettings** (dataclass):
- `enabled: bool` - Master LLM toggle
- `split_refinement_enabled: bool` - Enable split refinement
- `naming_enabled: bool` - Enable intelligent naming
- `model_preference: str` - "phi3_mini" or "gemma2_2b"
- `use_gpu: bool` - GPU acceleration
- `n_gpu_layers: int` - GPU layers (28 for 4GB VRAM)
- `n_ctx: int` - Context window size
- `split_confidence_threshold: float` - Minimum confidence for splits
- `auto_cleanup_enabled: bool` - Auto-cleanup after processing

**LlamaLoader**:
- `load_model() -> bool` - Load LLM model
- `generate(prompt, max_tokens) -> str` - Generate text
- `get_model_info() -> dict` - Get model information
- `cleanup()` - Free resources

**LLMManager**:
- `initialize() -> bool` - Initialize LLM (lazy loading)
- `generate(prompt, task_type, max_tokens) -> str` - Generate text
- `cleanup()` - Cleanup resources
- `stats: dict` - Usage statistics

---

## Integration Examples

### Example 1: Full Pipeline with LLM

```python
from services.pdf_processor import PDFProcessor
from services.ocr_service import OCRService
from services.embedding_service import generate_embeddings_for_document
from services.split_detection import detect_splits_for_document
from services.naming_service import NamingService
from services.cache_manager import get_cache_manager
import gc

cache = get_cache_manager()
pdf_path = "bundle.pdf"

# Phase 1: OCR
with PDFProcessor(pdf_path) as pdf:
    ocr = OCRService(gpu=True)
    # ... OCR processing ...
    ocr.cleanup()
    gc.collect()

# Phase 2: Embeddings
doc_id = cache.create_document(pdf_path, total_pages, file_size)
generate_embeddings_for_document(doc_id)

# Phase 3: Split Detection with LLM Refinement
split_count = detect_splits_for_document(
    doc_id,
    use_llm_refinement=True  # Enable LLM validation
)

# Phase 4: Document Extraction with LLM Naming
naming_service = NamingService()
with PDFProcessor(pdf_path) as pdf:
    for start, end in split_points:
        first_page_text = cache.get_page_text(doc_id, start)
        second_page_text = cache.get_page_text(doc_id, start + 1)
        
        filename = naming_service.suggest_name(
            text_content=first_page_text,
            page_num=1,
            fallback_prefix="doc",
            second_page_text=second_page_text,
            use_llm=True  # Enable LLM naming
        )
        
        pdf.extract_page_range(start, end, f"{filename}.pdf")

# Phase 5: LLM Cleanup
from services.llm.manager import cleanup_llm
cleanup_llm()
```

### Example 2: Testing LLM Components

```bash
cd python-backend
.venv\Scripts\activate

# Run full integration test suite
python test_llm_integration.py

# Run individual component tests
python services/llm/loader.py
python services/llm/manager.py
python services/llm/split_analyzer.py
python services/llm/name_generator.py
python services/llm/settings.py
```

---

## See Also

- **HANDOFF_LLAMA_CPP_INTEGRATION.md** - Detailed implementation handoff
- **IMPLEMENTATION_SPEC_DEBUNDLING.md** - De-bundling architecture
- **EMBEDDING_SERVICE_IMPLEMENTATION.md** - Embedding service details
- **python-backend/models/llm/README.md** - Model installation guide
- **python-backend/requirements.txt** - Python dependencies
