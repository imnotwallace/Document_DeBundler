# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Document De-Bundler is a Tauri-based desktop application for processing, splitting, and organizing PDF documents with advanced OCR and ML-powered document separation capabilities. The app can handle PDFs up to 5GB and runs entirely locally with no cloud dependencies.

**Key Capabilities**:
- **PDF Processing**: Handle large PDFs with intelligent splitting and organization
- **OCR**: Optical character recognition with GPU acceleration (PaddleOCR/Tesseract)
- **Document De-Bundling**: ML-powered intelligent document separation using semantic analysis
- **Semantic Analysis**: Content-based document grouping using Nomic Embed v1.5 embeddings
- **LLM Integration**: AI-powered naming suggestions and intelligent splitting recommendations
- **Local-First**: All processing including ML/AI runs locally without cloud dependencies

**📚 For comprehensive documentation**: See `docs/README.md` for the full documentation index and navigation guide.

## Critical Project Constraints

### Self-Contained Isolation

**IMPORTANT**: This project MUST remain completely self-contained and isolated from the system and other projects.

- **Python**: Always use the virtual environment in `python-backend/venv/`
  - Uses `uv` for fast dependency management (10-100x faster than pip)
  - Activate: `python-backend\venv\Scripts\activate` (Windows) or `source python-backend/venv/bin/activate` (Unix)
  - Never install packages globally
  - All Python commands must run with venv activated

- **Node.js**: All dependencies in local `node_modules/`
  - Never use global npm packages for project dependencies

- **Rust**: Cargo manages dependencies in `src-tauri/target/` automatically

## Architecture Quick Reference

### Three-Layer Architecture

```
Frontend (Svelte) ←→ Rust Core (Tauri) ←→ Python Backend
```

1. **Frontend (src/)**: Svelte + TypeScript + TailwindCSS
   - User interface running in Tauri's webview
   - Calls Rust via Tauri's invoke API

2. **Rust Core (src-tauri/)**: Tauri application
   - File system access and native dialogs
   - Spawns and manages Python subprocess
   - IPC bridge between frontend and Python

3. **Python Backend (python-backend/)**: PDF processing
   - Communicates via JSON over stdin/stdout
   - Services: PDF, OCR, ML/AI (de-bundling, embeddings, LLM)
   - Dependencies: sentence-transformers, scikit-learn, PaddleOCR/PaddlePaddle 3.0+

### Communication Flow

```
User Action → Frontend invoke() → Rust Command → Python stdin/stdout → Progress Events → UI Updates
```

**📚 For detailed architecture**: See `docs/ARCHITECTURE.md` for comprehensive system design and component interactions.

## Development Commands

### Setup (First Time)
```bash
# Windows
setup.bat

# macOS/Linux
chmod +x setup.sh && ./setup.sh
```

### Running the App
```bash
# Development mode with hot reload
npm run tauri:dev

# Frontend only (for UI work)
npm run dev
```

### Building
```bash
# Production build
npm run tauri:build

# Frontend build only
npm run build
```

### Testing
```bash
# Python tests (activate venv first)
cd python-backend
venv\Scripts\activate  # or source venv/bin/activate
pytest
pytest --cov  # with coverage

# Or run with uv directly (no activation needed)
cd python-backend
uv run pytest
uv run pytest --cov

# Frontend type checking
npm run check

# Python linting/formatting
cd python-backend
black .
flake8 .
```

## Key Files and Their Roles

### Frontend
- `src/App.svelte`: Main application component
- `src/main.ts`: Frontend entry point
- `src/app.css`: Global styles + Tailwind imports

### Rust/Tauri
- `src-tauri/src/main.rs`: Tauri app entry, registers commands
- `src-tauri/src/commands.rs`: Tauri commands (select_pdf_file, start_processing, etc.)
- `src-tauri/src/python_bridge.rs`: Python subprocess management, IPC bridge
- `src-tauri/tauri.conf.json`: Tauri configuration, permissions, build settings

### Python Backend
- `python-backend/main.py`: IPC entry point (stdin/stdout interface)
- `python-backend/services/pdf_processor.py`: PDF analysis and splitting
- `python-backend/services/ocr_service.py`: OCR processing wrapper
- `python-backend/services/ocr/`: OCR abstraction layer (PaddleOCR/Tesseract engines)
- `python-backend/services/split_detection.py`: ML-based document splitting (DBSCAN)
- `python-backend/services/embedding_service.py`: Semantic analysis (Nomic Embed v1.5)
- `python-backend/services/cache_manager.py`: Performance caching
- `python-backend/services/naming_service.py`: Document naming logic
- `python-backend/services/bundler.py`: ZIP creation and file organization

## Code Patterns

### Adding a New Tauri Command

1. **Define in commands.rs**:
```rust
#[tauri::command]
pub fn my_new_command(param: String) -> Result<ReturnType, String> {
    // Implementation
    Ok(result)
}
```

2. **Register in main.rs**:
```rust
.invoke_handler(tauri::generate_handler![
    existing_command,
    my_new_command  // Add here
])
```

3. **Call from Frontend**:
```typescript
import { invoke } from "@tauri-apps/api/tauri";
const result = await invoke<ReturnType>("my_new_command", { param: "value" });
```

### Python IPC Protocol

**Commands sent to Python** (via stdin):
```json
{
  "command": "analyze",
  "file_path": "/path/to/file.pdf",
  "options": {}
}
```

**Events from Python** (via stdout):
```json
{"type": "progress", "data": {"current": 10, "total": 100, "message": "Processing..."}}
{"type": "result", "data": {...}}
{"type": "error", "data": {"message": "Error description"}}
```

## Quick References

### OCR System

**Engine Selection**:
- **PaddleOCR (Primary)**: Best accuracy (95-98%), 2-3x faster, GPU support (CUDA/DirectML)
- **Tesseract (Fallback)**: Lightweight CPU-only, good for simple documents

**Basic Usage**:
```python
from services.ocr_service import OCRService

# Initialize with auto-detection (GPU if available, CPU fallback)
ocr = OCRService(gpu=True)

if ocr.is_available():
    text = ocr.process_pdf_page(pdf_path, page_num=0)
    texts = ocr.process_batch(image_arrays)
    info = ocr.get_engine_info()

ocr.cleanup()
```

**Hardware Auto-Detection**: System automatically detects GPU/CPU capabilities and configures:
- Adaptive batch sizing (4GB VRAM: 25 pages @ 300 DPI, 8GB VRAM: 50 pages, CPU 16GB RAM: 10 pages)
- **High-DPI support**: Up to 1600 DPI (batch sizes auto-adjust: 1200 DPI = 1-2 pages, 1600 DPI = 1 page)
- Real-time VRAM monitoring with automatic batch adjustment
- Hybrid GPU/CPU processing on constrained hardware

**📚 For detailed OCR documentation**: See `docs/guides/PADDLEPADDLE_3.0_UPGRADE_AND_CUDA_FIX.md` for:
- Complete architecture layers and engine details
- Model management (auto-download vs pre-bundled)
- Memory optimization and hybrid processing
- Hardware requirements and performance tuning
- Troubleshooting guide

### Document De-Bundling

**Pipeline Overview**:
```
PDF Input → Content Extraction (OCR/text layer) → Semantic Embeddings (Nomic v1.5)
→ Similarity Analysis → DBSCAN Clustering → Split Point Detection → Separated Documents
```

**Basic Usage**:
```python
from services.split_detection import detect_splits_for_document
from services.embedding_service import EmbeddingService

# High-level API (handles caching and orchestration)
num_splits = detect_splits_for_document(
    doc_id='unique_doc_id',
    use_llm_refinement=False,
    progress_callback=lambda curr, total, msg: print(f"{msg} ({curr}/{total})")
)

# For custom workflows, use SplitDetector and EmbeddingService directly
embedder = EmbeddingService(device='cuda', model_type='multimodal')
embeddings = embedder.generate_embeddings(page_texts)
# ... run detection methods, combine signals
```

**Model Types**:
- **Text Model**: `nomic-embed-text-v1.5` (~550MB) - Document analysis
- **Vision Model**: `nomic-embed-vision-v1.5` (~600MB) - Image understanding
- **Multimodal** (default): Both models (~1.15GB) - Cross-modal retrieval

**📚 For detailed de-bundling documentation**: See `docs/features/DEBUNDLING_QUICK_START.md` for:
- Complete workflow with code examples
- Embedding model management and installation
- DBSCAN clustering and sensitivity tuning
- Multimodal semantic analysis
- Performance characteristics and caching
- Memory management and troubleshooting

### Performance Quick Facts

**OCR Processing Speed** (optimized for 4GB VRAM):

| Hardware Configuration | Speed per Page | 5GB PDF (5000+ pages) |
|------------------------|----------------|------------------------|
| 4GB VRAM + 16GB RAM (Hybrid) | ~0.15-0.35s | ~8-20 minutes |
| 4GB VRAM + 8GB RAM (GPU only) | ~0.15-0.35s | ~10-25 minutes |
| 8GB+ VRAM | ~0.1-0.25s | ~8-15 minutes |
| CPU only (16GB RAM) | ~0.5-1s | ~40-80 minutes |
| Tesseract (CPU) | ~0.3-0.8s | Varies |

**Memory Requirements**:

| Component | Memory Usage |
|-----------|--------------|
| PaddleOCR models | ~25-75MB base |
| Batch (25 pages @ 300 DPI) | ~2.5-3GB VRAM |
| Single page @ 600 DPI | ~250MB VRAM |
| Single page @ 1200 DPI | ~500MB VRAM |
| Single page @ 1600 DPI | ~900MB VRAM |
| CPU processing per page | ~100-150MB RAM |
| Embeddings (1000 pages) | ~3MB in memory, cached to disk |

**High-DPI Support** (1200-1600 DPI):
- System supports up to **1600 DPI** (18,000px max dimension)
- **1200 DPI** recommended for 4GB VRAM (batch_size=1-2)
- **1600 DPI** requires 6GB+ VRAM (batch_size=1)
- Auto-detection warns if DPI exceeds safe memory limits
- Adaptive batch sizing automatically adjusts for high DPI

**Optimizations**:
- Always check for text layer first (`has_text_layer()`) to skip unnecessary OCR
- GPU auto-detection with graceful CPU fallback
- Real-time VRAM monitoring prevents OOM errors
- Adaptive batch sizing based on available memory
- Hybrid mode enables CPU offload on constrained hardware

**📚 For detailed performance tuning**: See `docs/DEVELOPER_QUICK_START.md` for memory management, batch processing strategies, and optimization techniques.

## Common Development Tasks

### Adding a New UI Component
1. Create component in `src/lib/components/`
2. Import and use in `App.svelte`
3. Style with Tailwind classes
4. Type-check with `npm run check`

### Adding a New Python Service
1. Create service file in `python-backend/services/`
2. Import in `python-backend/services/__init__.py`
3. Use in `main.py` command handlers
4. Add tests in `python-backend/tests/`

### Modifying Tauri Permissions
1. Edit `src-tauri/tauri.conf.json`
2. Update `allowlist` section
3. Rebuild: `npm run tauri:dev` or `npm run tauri:build`

### Updating Dependencies

**Python** (using uv):
```bash
cd python-backend
venv\Scripts\activate  # or source venv/bin/activate
uv pip install <package>
uv pip freeze > requirements.txt
```

Or without activation:
```bash
cd python-backend
uv pip install <package>
uv pip freeze > requirements.txt
```

**Node.js**:
```bash
npm install <package>
# Automatically updates package.json
```

**Rust**:
```bash
cd src-tauri
cargo add <crate>
# Automatically updates Cargo.toml
```

## Debugging

### Frontend
- Use browser DevTools: Right-click in app → Inspect Element
- Console logs appear in DevTools console
- `console.log()` for debugging

### Rust
- Logs printed with `println!` appear in terminal running `npm run tauri:dev`
- Use `dbg!()` macro for debug output
- Enable debug mode: set `TAURI_DEBUG=true`

### Python
- Logs to stderr (configured in `main.py`)
- Check terminal output where `npm run tauri:dev` is running
- Use `logger.info()`, `logger.error()` for logging
- **Never use `print()` for logging** - stdout is reserved for IPC

### IPC Issues
- Check JSON formatting (both Rust and Python sides)
- Verify command names match exactly
- Check Python process is actually spawned
- Monitor stdin/stdout in Python with debug logging to stderr

### Common Troubleshooting

**"Python process not started"**: Ensure Python is in PATH, check venv setup

**Frontend can't call Tauri commands**: Ensure command is registered in `main.rs`, check spelling

**OCR not working**: Check PaddleOCR installed in venv, verify GPU availability if enabled

**Out of memory**: Reduce batch size, close other applications

## Documentation Navigation

Comprehensive documentation is organized in the `docs/` directory:

### Quick Start
- **[docs/README.md](docs/README.md)** - Documentation index and navigation
- **[docs/DEVELOPER_QUICK_START.md](docs/DEVELOPER_QUICK_START.md)** - Getting started guide
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design overview
- **[docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md)** - Current feature status (single source of truth)

### Feature Documentation (`docs/features/`)
- **DEBUNDLING_QUICK_START.md** - Document de-bundling feature guide
- **EMBEDDING_SERVICE_IMPLEMENTATION.md** - Semantic analysis with Nomic Embed v1.5
- **SPLIT_DETECTION_IMPLEMENTATION_REPORT.md** - Document boundary detection algorithms
- **LLM_INTEGRATION.md** - Local LLM integration (llama.cpp, Phi-3, Gemma)

### Setup Guides (`docs/guides/`)
- **TESSERACT_BUNDLING.md** - Bundling Tesseract for offline use
- **PADDLEPADDLE_3.0_UPGRADE_AND_CUDA_FIX.md** - GPU acceleration setup and OCR details

### Implementation Details (`docs/implementations/`)
- **OCR_EVENT_HANDLERS_IMPLEMENTATION.md** - Svelte event-driven UI
- **PHASE_3_STEP_2_IMPLEMENTATION_REPORT.md** - Rust Tauri commands
- **PYTHON_BRIDGE_IMPLEMENTATION.md** - Rust-Python async IPC bridge

### Testing (`docs/testing/`)
- **TEST_RESULTS_2025-11-01.md** - Latest test suite results (93% pass rate)

### Archive (`docs/archive/`)
- Historical documentation, session handoffs, completed checklists
- See `docs/archive/README.md` for organization and index

### Task → Documentation Quick Map

| When You Need To... | Consult This Document |
|---------------------|------------------------|
| Set up development environment | `docs/DEVELOPER_QUICK_START.md` |
| Understand system architecture | `docs/ARCHITECTURE.md` |
| Work with OCR features | `docs/guides/PADDLEPADDLE_3.0_UPGRADE_AND_CUDA_FIX.md` |
| Implement document de-bundling | `docs/features/DEBUNDLING_QUICK_START.md` |
| Integrate embedding models | `docs/features/EMBEDDING_SERVICE_IMPLEMENTATION.md` |
| Add LLM features | `docs/features/LLM_INTEGRATION.md` |
| Check feature completion status | `docs/IMPLEMENTATION_STATUS.md` |
| Troubleshoot issues | This file (Debugging section) + specific feature docs |
| Review test results | `docs/testing/TEST_RESULTS_2025-11-01.md` |

## Design Principles

1. **Local-First**: No cloud dependencies, all processing local
2. **Self-Contained**: All dependencies in project directory
3. **Streaming**: Handle large files without full buffering
4. **Progressive**: Show progress, don't block UI
5. **Fail-Safe**: Graceful error handling, informative messages
6. **Always use venv**: Always use venv python for any python testing or packaging in this project
7. **No Unicode in Python**: Never use unicode characters in python scripts
