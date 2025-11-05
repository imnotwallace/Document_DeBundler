# Document De-Bundler Architecture

## System Overview

Document De-Bundler is a desktop application built with **Tauri** (Rust + Svelte) that processes, splits, and organizes PDF documents using advanced OCR and ML-powered document separation. All processing runs locally with no cloud dependencies.

**Core Capabilities:**
- PDF processing (up to 5GB)
- OCR with GPU acceleration (PaddleOCR/Tesseract)
- ML-powered document separation (DBSCAN clustering)
- Semantic analysis (Nomic Embed v1.5)
- LLM integration (llama.cpp with Phi-3/Gemma)
- Fully local operation

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend Layer                       │
│              (Svelte + TypeScript + Tailwind)           │
│                                                         │
│  - User interface in Tauri webview                     │
│  - Calls Rust via tauri::invoke                        │
│  - Real-time progress updates                          │
└───────────────────┬─────────────────────────────────────┘
                    │ IPC (invoke/emit)
┌───────────────────▼─────────────────────────────────────┐
│                     Rust Core Layer                     │
│                   (Tauri Application)                   │
│                                                         │
│  - File system access (native dialogs)                 │
│  - Spawns and manages Python subprocess                │
│  - IPC bridge between frontend and Python              │
│  - Commands exposed via #[tauri::command]              │
└───────────────────┬─────────────────────────────────────┘
                    │ JSON stdin/stdout
┌───────────────────▼─────────────────────────────────────┐
│                   Python Backend Layer                  │
│                (PDF Processing & ML/AI)                 │
│                                                         │
│  - Communicates via JSON over stdin/stdout             │
│  - Services: PDF, OCR, embedding, LLM, bundling        │
│  - Streaming/incremental processing                    │
│  - ML: Nomic Embed, DBSCAN, PaddleOCR, llama.cpp       │
└─────────────────────────────────────────────────────────┘
```

## Communication Flow

### Example: Process PDF File

```
1. User clicks "Select PDF" button
   ↓
2. Frontend calls invoke("select_pdf_file")
   ↓
3. Rust opens native file picker dialog
   ↓
4. Returns file path to frontend
   ↓
5. Frontend calls invoke("start_processing", {filePath, options})
   ↓
6. Rust spawns Python subprocess (python-backend/main.py)
   ↓
7. Rust sends JSON command to Python stdin:
   {"command": "analyze", "file_path": "/path/to/file.pdf", "options": {...}}
   ↓
8. Python processes, emits events to stdout:
   {"type": "progress", "data": {"current": 10, "total": 100}}
   {"type": "result", "data": {...}}
   ↓
9. Rust forwards events to frontend via emit()
   ↓
10. Frontend updates UI with progress bars and results
```

## Layer Details

### Frontend Layer (src/)

**Technology Stack:**
- **Svelte 4** - Reactive UI framework
- **TypeScript** - Type-safe JavaScript
- **TailwindCSS** - Utility-first styling
- **Tauri API** - Rust bridge

**Key Components:**
- `App.svelte` - Main application component
- `main.ts` - Frontend entry point
- `app.css` - Global styles + Tailwind

**Responsibilities:**
- User interface rendering
- User input handling
- Tauri command invocation
- Progress visualization
- Error display

### Rust Core Layer (src-tauri/)

**Technology Stack:**
- **Tauri 2.x** - Desktop framework
- **Tokio** - Async runtime
- **serde_json** - JSON serialization

**Key Files:**
- `main.rs` - Application entry, command registration
- `commands.rs` - Tauri commands called from frontend
- `python_bridge.rs` - Python subprocess management
- `tauri.conf.json` - Configuration and permissions

**Tauri Commands:**
- `select_pdf_file()` - Opens native file picker
- `get_file_info()` - Retrieves file metadata
- `start_processing()` - Launches Python processing
- `cancel_processing()` - Stops active processing
- `get_processing_status()` - Returns current status

**Python Bridge:**
- Spawns Python process with venv activation
- Sends JSON commands via stdin
- Reads JSON events from stdout
- Forwards events to frontend via Tauri emit
- Handles process lifecycle (start/stop/cleanup)

### Python Backend Layer (python-backend/)

**Technology Stack:**
- **Python 3.11+** - Core language
- **PyMuPDF (fitz)** - PDF manipulation
- **PaddleOCR/PaddlePaddle 3.0** - Primary OCR (GPU)
- **Tesseract** - Fallback OCR (CPU)
- **sentence-transformers** - Embedding generation (Nomic Embed v1.5)
- **scikit-learn** - DBSCAN clustering
- **llama.cpp** - Local LLM inference

**Key Files:**
- `main.py` - IPC entry point (stdin/stdout interface)
- `services/pdf_processor.py` - PDF analysis and splitting
- `services/ocr_service.py` - OCR processing wrapper
- `services/split_detection.py` - ML-based document splitting
- `services/embedding_service.py` - Semantic analysis
- `services/llm/` - LLM integration module

**Service Architecture:**

```
services/
├── pdf_processor.py          # PDF manipulation (PyMuPDF)
├── ocr_service.py            # OCR facade
├── ocr/                      # OCR abstraction layer
│   ├── base.py              # Abstract engine interface
│   ├── config.py            # Hardware detection
│   ├── manager.py           # Engine factory
│   ├── engines/
│   │   ├── paddleocr_engine.py
│   │   └── tesseract_engine.py
│   ├── vram_monitor.py      # Real-time VRAM tracking
│   └── text_quality.py      # Quality analysis
├── embedding_service.py      # Nomic Embed v1.5
├── split_detection.py        # DBSCAN clustering
├── llm/                      # LLM integration
│   ├── config.py
│   └── prompts.py
├── cache_manager.py          # Performance caching
├── naming_service.py         # Document naming
└── bundler.py                # ZIP creation
```

## OCR Architecture

### Engine Selection

**Primary: PaddleOCR**
- Best accuracy (95-98%)
- 2-3x faster than alternatives
- GPU acceleration (CUDA, DirectML)
- Auto-downloads models (~10MB)
- PaddlePaddle 3.0+ required

**Fallback: Tesseract**
- Lightweight CPU-only
- Good for simple documents
- No deep learning dependencies
- Can be bundled for offline use

### Hardware Auto-Detection

```python
capabilities = detect_hardware_capabilities()
# Returns:
{
    'gpu_available': True/False,
    'cuda_available': True/False,
    'directml_available': True/False,  # Windows AMD/Intel
    'gpu_memory_gb': 4.0,
    'system_memory_gb': 16.0,
    'cpu_count': 8,
    'platform': 'Windows'
}
```

### Adaptive Batch Sizing (4GB VRAM Optimized)

| Hardware | Batch Size | Notes |
|----------|-----------|-------|
| 10GB+ VRAM | 60 pages | Maximum throughput |
| 8GB VRAM | 50 pages | High performance |
| 6GB VRAM | 35 pages | Good balance |
| **4GB VRAM** | **25 pages** | **Target hardware** |
| 3GB VRAM | 15 pages | Conservative |
| 2GB VRAM | 8 pages | Minimal |
| 16GB RAM (CPU) | 10 pages | No GPU |

### Memory Monitoring

- Real-time VRAM usage tracking
- Adaptive batch size reduction under pressure
- Sub-batch splitting for large batches
- Automatic batch size recovery

## Document De-Bundling Architecture

### ML Pipeline

```
PDF Input
    ↓
Content Extraction (OCR or text layer)
    ↓
Semantic Embeddings (Nomic Embed v1.5, 768-dim vectors)
    ↓
Similarity Analysis (Cosine similarity between pages)
    ↓
Clustering (DBSCAN groups related pages)
    ↓
Split Point Detection (Boundary detection with confidence scores)
    ↓
LLM Naming (Optional intelligent document names)
    ↓
Separated Documents
```

### Detection Methods

**1. Page Number Reset**
- Detects when page numbering restarts (e.g., 1, 2, 3... → 1, 2, 3...)
- High confidence signal for document boundaries

**2. Blank Page Detection**
- Identifies blank or near-blank separator pages
- Common in scanned bundled documents

**3. Semantic Discontinuity**
- Analyzes embedding similarity between consecutive pages
- Large similarity drop indicates topic change

**4. DBSCAN Clustering**
- Groups semantically similar pages
- Cluster boundaries suggest document splits
- Tunable via `eps` parameter (0.3-0.7)

### Signal Combination

Multiple detection methods run in parallel, signals are combined with confidence weighting:

```python
candidates = detector.combine_signals([
    page_number_reset_splits,
    blank_page_splits,
    semantic_discontinuity_splits,
    clustering_splits
])

# Extract high-confidence splits
split_pages = [c['page'] for c in candidates if c['confidence'] >= 0.5]
```

## Embedding Service Architecture

### Model Management

**Nomic Embed v1.5 Models:**
- **Text Model:** `nomic-embed-text-v1.5` (~550MB)
  - Document analysis, semantic search
  - 768-dimensional embeddings
- **Vision Model:** `nomic-embed-vision-v1.5` (~600MB)
  - Image understanding, visual analysis
  - 768-dimensional embeddings (aligned with text)
- **Multimodal:** Both models loaded (~1.15GB)
  - Cross-modal retrieval, aligned embeddings

**Model Sources:**
1. **Bundled:** `python-backend/models/embeddings/` (offline use)
2. **Auto-download:** HuggingFace cache (first-run convenience)

### Sequential GPU Processing

To avoid memory conflicts, GPU-intensive services run sequentially:

```python
# Phase 1: OCR (GPU)
ocr = OCRService(gpu=True)
texts = ocr.process_batch(images)
ocr.cleanup()  # Free GPU memory

# Phase 2: Embedding (GPU) - No conflict
embedder = EmbeddingService(device='cuda', model_type='multimodal')
embeddings = embedder.generate_embeddings(texts)
embedder.cleanup()  # Free GPU memory
```

## LLM Integration Architecture

### Local Inference Stack

**llama.cpp Integration:**
- CPU inference with optional GPU acceleration
- Models: Phi-3-mini-4k, Gemma-2-2B
- Quantized models (Q4_K_M) for efficiency
- GGUF format

**Use Cases:**
1. Intelligent document naming based on content
2. Splitting point suggestions
3. Content summarization

## Performance Characteristics

### OCR Processing (5GB PDFs, 5000+ pages)

| Hardware | Processing Time | Notes |
|----------|----------------|-------|
| 4GB VRAM + 16GB RAM | 8-20 min | Hybrid mode (target) |
| 4GB VRAM + 8GB RAM | 10-25 min | GPU only |
| 8GB+ VRAM | 8-15 min | GPU only, higher batch |
| CPU only (16GB RAM) | 40-80 min | No GPU |

**Speed per Page:**
- PaddleOCR (4GB GPU): ~0.15-0.35s/page
- PaddleOCR (8GB+ GPU): ~0.1-0.25s/page
- PaddleOCR (CPU): ~0.5-1s/page
- Tesseract (CPU): ~0.3-0.8s/page

### Embedding Generation

- **GPU (CUDA/DirectML):** ~0.3-0.8s/page
- **CPU:** ~1-3s/page
- **Batch Processing:** 32-64 pages at once

### DBSCAN Clustering

- Near-instant for <1000 pages
- Linear scaling with page count
- ~100MB memory per 1000 pages

## Data Flow Examples

### Full Document Processing

```
User → Frontend → Rust → Python
                    ↓
              1. Load PDF (PyMuPDF)
                    ↓
              2. Check text layer per page
                    ↓
              3. OCR if needed (PaddleOCR/Tesseract)
                    ↓
              4. Generate embeddings (Nomic Embed)
                    ↓
              5. Detect splits (DBSCAN + heuristics)
                    ↓
              6. Optional: LLM naming (llama.cpp)
                    ↓
              7. Split and save documents
                    ↓
              8. Create output bundle (ZIP)
                    ↓
Python → Rust → Frontend → User
```

### Progress Events

```json
// Progress update
{"type": "progress", "data": {"current": 50, "total": 100, "message": "Processing page 50/100"}}

// Result
{"type": "result", "data": {"output_path": "/path/to/output.zip", "documents": 5}}

// Error
{"type": "error", "data": {"message": "Failed to process page 42", "details": "..."}}
```

## Security & Isolation

### Self-Contained Operation

- **Python:** Virtual environment in `python-backend/venv/`
- **Node.js:** Dependencies in local `node_modules/`
- **Rust:** Cargo manages `src-tauri/target/`
- **No global installations** (except Python/Node/Rust themselves)

### Tauri Permissions

Configured in `tauri.conf.json`:

```json
{
  "fs.scope": ["$APPDATA/*", "$DESKTOP/*", "$DOCUMENT/*", "$DOWNLOAD/*", "$HOME/*"],
  "dialog": true,
  "shell.execute": true
}
```

### Data Privacy

- All processing occurs locally
- No network requests (except model downloads on first run)
- No telemetry or analytics
- User data never leaves the machine

## Deployment

### Development

```bash
npm run tauri:dev  # Hot reload enabled
```

### Production Build

```bash
npm run tauri:build
```

**Output:**
- Windows: `.msi` installer in `src-tauri/target/release/bundle/msi/`
- macOS: `.dmg` in `src-tauri/target/release/bundle/dmg/`
- Linux: `.AppImage` in `src-tauri/target/release/bundle/appimage/`

### Bundled Resources

Production builds can include:
- PaddleOCR models (`models/det/`, `models/rec/`, `models/cls/`)
- Nomic Embed models (`models/embeddings/text/`, `models/embeddings/vision/`)
- Tesseract binary and data files (Windows)
- LLM models in GGUF format

## Extension Points

### Adding New OCR Engine

1. Implement `OCREngine` interface in `services/ocr/engines/`
2. Register in `OCRManager._create_engine()`
3. Use: `OCRService(engine="new_engine")`

### Adding New Detection Method

1. Add method to `SplitDetector` class
2. Call in `detect_splits_for_document()`
3. Signals auto-combine with existing methods

### Adding New LLM Model

1. Download GGUF model
2. Update `services/llm/config.py`
3. Configure in application settings

## See Also

- [Developer Quick Start](DEVELOPER_QUICK_START.md) - Get started developing
- [OCR Documentation](features/LLM_INTEGRATION.md) - OCR system details
- [De-Bundling Guide](features/DEBUNDLING_QUICK_START.md) - Document separation
- [Implementation Status](IMPLEMENTATION_STATUS.md) - Current feature status

---

**Last Updated:** 2025-11-03
**Architecture Version:** 2.0
