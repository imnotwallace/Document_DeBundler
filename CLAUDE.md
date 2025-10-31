# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Document De-Bundler is a Tauri-based desktop application for processing, splitting, and organizing PDF documents with OCR capabilities. The app can handle PDFs up to 5GB and runs entirely locally with no cloud dependencies.

## Critical Project Constraints

### Self-Contained Isolation

**IMPORTANT**: This project MUST remain completely self-contained and isolated from the system and other projects.

- **Python**: Always use the virtual environment in `python-backend/venv/`
  - Activate: `python-backend\venv\Scripts\activate` (Windows) or `source python-backend/venv/bin/activate` (Unix)
  - Never install packages globally
  - All Python commands must run with venv activated

- **Node.js**: All dependencies in local `node_modules/`
  - Never use global npm packages for project dependencies

- **Rust**: Cargo manages dependencies in `src-tauri/target/` automatically

## Architecture

### Three-Layer Architecture

```
Frontend (Svelte) ←→ Rust Core (Tauri) ←→ Python Backend
```

1. **Frontend (src/)**: Svelte + TypeScript + TailwindCSS
   - User interface running in Tauri's webview
   - Calls Rust via Tauri's invoke API
   - Real-time progress updates

2. **Rust Core (src-tauri/)**: Tauri application
   - File system access (native dialogs)
   - Spawns and manages Python subprocess
   - IPC bridge between frontend and Python
   - Commands exposed via `#[tauri::command]`

3. **Python Backend (python-backend/)**: PDF processing
   - Communicates via JSON over stdin/stdout
   - Services: PDF processing, OCR, naming, bundling
   - Designed for streaming/incremental processing

### Communication Flow

```
User selects file
    → Frontend calls invoke("select_pdf_file")
    → Rust opens native file picker
    → Returns file path to frontend
    → Frontend calls invoke("start_processing", {filePath, options})
    → Rust spawns Python subprocess
    → Rust sends JSON command to Python stdin
    → Python processes, emits events to stdout
    → Rust forwards events to frontend
    → Frontend updates UI with progress
```

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
- `src-tauri/src/commands.rs`: Tauri commands called from frontend
  - `select_pdf_file()`: Opens file picker
  - `get_file_info()`: Gets file metadata
  - `start_processing()`: Launches Python processing
  - `cancel_processing()`: Stops processing
  - `get_processing_status()`: Returns current status
- `src-tauri/src/python_bridge.rs`: Python subprocess management
  - Spawns Python process
  - Sends JSON commands via stdin
  - Reads JSON events from stdout
- `src-tauri/tauri.conf.json`: Tauri configuration, permissions, build settings

### Python Backend
- `python-backend/main.py`: IPC entry point (stdin/stdout interface)
  - Reads JSON commands from stdin
  - Emits JSON events to stdout (progress, results, errors)
  - Logging goes to stderr (not stdout)
- `python-backend/services/pdf_processor.py`: PDF analysis and splitting
  - Uses PyMuPDF (fitz) for PDF manipulation
  - Streaming/incremental processing for large files
  - OCR integration for scanned pages
- `python-backend/services/ocr_service.py`: OCR processing wrapper
  - Uses OCR abstraction layer
  - Manages PaddleOCR/Tesseract engines
- `python-backend/services/ocr/`: OCR abstraction layer
  - `base.py`: Abstract OCR engine interface
  - `config.py`: Hardware detection and configuration
  - `manager.py`: Engine factory and lifecycle management
  - `engines/paddleocr_engine.py`: PaddleOCR implementation (primary)
  - `engines/tesseract_engine.py`: Tesseract implementation (fallback)
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

## OCR Architecture

### Overview

The OCR system uses a hybrid architecture with **PaddleOCR as the primary engine** and **Tesseract as a lightweight fallback**. The architecture provides:

- Automatic GPU detection with CPU fallback
- Multiple engine support via abstraction layer
- Auto-bundled models for offline operation
- Memory-efficient batch processing
- Hardware-adaptive configuration

### Engine Selection

**PaddleOCR (Primary)**:
- Best accuracy (95-98%) and performance
- 2-3x faster than alternatives
- GPU acceleration support (CUDA, DirectML)
- 50% lower memory footprint
- Auto-downloads models (~10MB) on first run
- **Version**: Uses PaddleOCR 3.x (not 2.x - breaking changes apply)
  - Logging is now controlled via PaddleOCR's logging system (not `show_log` parameter)
  - Parameters like `show_log` and `use_mp` from 2.x are not supported

**Tesseract (Fallback)**:
- Lightweight CPU-only option
- Lowest memory footprint (~100MB)
- Good for simple documents
- No deep learning dependencies

### Architecture Layers

```
OCR Service (ocr_service.py)
    ↓
OCR Manager (ocr/manager.py)
    ↓
OCR Engine Interface (ocr/base.py)
    ↓
    ├── PaddleOCR Engine (ocr/engines/paddleocr_engine.py)
    └── Tesseract Engine (ocr/engines/tesseract_engine.py)
```

### Using the OCR Service

**Basic Usage**:
```python
from services.ocr_service import OCRService

# Initialize with auto-detection (GPU if available, CPU fallback)
ocr = OCRService(gpu=True)

# Check availability
if ocr.is_available():
    # Process single page
    text = ocr.process_pdf_page(pdf_path, page_num=0)

    # Process batch for better performance
    texts = ocr.process_batch(image_arrays)

    # Get engine info
    info = ocr.get_engine_info()
    print(f"Using: {info['engine']}, GPU: {info['gpu_enabled']}")

# Cleanup when done
ocr.cleanup()
```

**With Context Manager**:
```python
from services.ocr import create_ocr_manager

with create_ocr_manager(use_gpu=True) as ocr:
    result = ocr.process_image(image_array)
    print(f"Text: {result.text}")
    print(f"Confidence: {result.confidence:.2f}")
```

### Hardware Auto-Detection

The OCR system automatically detects and configures based on available hardware:

```python
from services.ocr.config import detect_hardware_capabilities

capabilities = detect_hardware_capabilities()
# Returns:
# {
#     'gpu_available': True/False,
#     'cuda_available': True/False,
#     'directml_available': True/False,  # Windows AMD/Intel GPUs
#     'gpu_memory_gb': 4.0,
#     'system_memory_gb': 16.0,
#     'cpu_count': 8,
#     'platform': 'Windows'
# }
```

**Adaptive Batch Sizing** (Optimized for 4GB VRAM):
- GPU with 10GB+ VRAM: batch_size = 60
- GPU with 8GB VRAM: batch_size = 50
- GPU with 6GB VRAM: batch_size = 35
- **GPU with 4GB VRAM: batch_size = 25** (Target hardware - 2.5x improvement)
- GPU with 3GB VRAM: batch_size = 15
- GPU with 2GB VRAM: batch_size = 8
- CPU with 32GB+ RAM: batch_size = 20
- CPU with 24GB RAM: batch_size = 15
- CPU with 16GB RAM: batch_size = 10
- CPU with 8GB RAM: batch_size = 5
- CPU with 4GB RAM: batch_size = 3

**Real-Time Memory Monitoring**:
- Automatic VRAM usage tracking during processing
- Adaptive batch size reduction when memory pressure detected
- Sub-batch splitting for large batches under memory constraints
- Automatic batch size recovery when memory pressure relieved

### Memory Requirements and Optimization

**System Requirements by Configuration**:

1. **4GB VRAM + 8GB RAM** (Minimum):
   - Can process most PDFs with GPU acceleration
   - Batch size: 25 pages
   - DPI: 300 (standard quality)
   - Processing time: ~10-25 minutes for 5GB PDF (5000+ pages)
   - Hybrid mode disabled (GPU only)

2. **4GB VRAM + 16GB RAM** (Recommended):
   - Optimal for target hardware
   - Batch size: 25 pages (GPU), 10 pages (CPU fallback)
   - DPI: 300-400 (adaptive)
   - Processing time: ~8-20 minutes for 5GB PDF
   - **Hybrid mode enabled** - automatic CPU offload on memory pressure

3. **4GB VRAM + 32GB RAM** (Optimal):
   - Best performance for large documents
   - Batch size: 25 pages (GPU), 20 pages (CPU)
   - DPI: 400 (high quality)
   - Processing time: ~8-15 minutes for 5GB PDF
   - Hybrid mode with aggressive CPU parallel processing

**Hybrid GPU/CPU Processing**:

The system automatically enables hybrid mode on 4GB VRAM systems with 16GB+ RAM:
- Monitors VRAM usage in real-time
- Seamlessly offloads to CPU when GPU memory pressure detected
- Splits work between GPU and CPU based on available resources
- Prevents Out-Of-Memory (OOM) errors on constrained hardware

**Adaptive DPI Selection**:

DPI is automatically adjusted based on available memory:
```python
from services.ocr.config import get_adaptive_dpi, detect_hardware_capabilities

capabilities = detect_hardware_capabilities()
dpi = get_adaptive_dpi(
    use_gpu=True,
    gpu_memory_gb=capabilities['gpu_memory_gb'],
    system_memory_gb=capabilities['system_memory_gb'],
    target_quality="balanced"  # "low", "balanced", "high", "max"
)

# For 4GB VRAM: returns 300-400 DPI
# For 2GB VRAM: returns 200-300 DPI
# For 8GB+ VRAM: returns 400-600 DPI
```

**Memory Usage Estimates**:
- Per page at 300 DPI: ~26MB raw, ~80-100MB with processing overhead
- PaddleOCR models: ~25-75MB base memory
- Batch of 25 pages: ~2.5-3GB VRAM usage (safe for 4GB)
- CPU processing: ~100-150MB per page in system RAM

### Model Management

**Auto-Download** (default):
- Models auto-download on first run (~10MB for English)
- Cached in user directory (Windows: `%LOCALAPPDATA%\PaddleOCR\`)
- Progress shown to user during download

**Pre-Bundled Models** (for offline):
- Place models in `python-backend/models/`
- Structure: `models/det/`, `models/rec/`, `models/cls/`
- See `python-backend/models/README.md` for details
- Auto-detected and used if present

### Integration with PDF Processor

**Text Layer Detection**:
```python
from services.pdf_processor import PDFProcessor
from services.ocr_service import OCRService

with PDFProcessor(pdf_path) as pdf:
    # Check if page needs OCR
    if pdf.has_text_layer(page_num):
        # Extract from text layer (fast)
        text = pdf.extract_text(page_num)
    else:
        # Use OCR (slower but necessary)
        ocr = OCRService(gpu=True)
        text = ocr.process_pdf_page(pdf_path, page_num)
```

**Batch Processing with Memory Management**:
```python
def process_callback(current, total, message):
    print(f"{message} ({current}/{total})")

with PDFProcessor(pdf_path) as pdf:
    ocr = OCRService(gpu=True)

    # Process all pages with automatic text layer detection
    texts = pdf.process_pages_with_ocr(
        ocr_service=ocr,
        batch_size=10,  # Process 10 pages at a time
        progress_callback=process_callback
    )

    ocr.cleanup()
```

### Adding New OCR Engines

To add a new OCR engine (e.g., EasyOCR):

1. **Create Engine Adapter**:
```python
# python-backend/services/ocr/engines/easyocr_engine.py
from ..base import OCREngine, OCRResult, OCRConfig

class EasyOCREngine(OCREngine):
    def initialize(self):
        import easyocr
        self.reader = easyocr.Reader(self.config.languages, gpu=self.config.use_gpu)
        self._initialized = True

    def process_image(self, image):
        result = self.reader.readtext(image)
        # Extract and format result...
        return OCRResult(text=text, confidence=conf)

    # Implement other abstract methods...
```

2. **Register in Manager**:
```python
# python-backend/services/ocr/manager.py
def _create_engine(self, engine_name: str):
    if engine_name == "easyocr":
        from .engines.easyocr_engine import EasyOCREngine
        return EasyOCREngine(self.config)
    # ... existing engines
```

3. **Use**:
```python
ocr = OCRService(engine="easyocr", gpu=True)
```

### Performance Optimization

**For 5GB PDFs**:
1. Auto-detect text layer (skip OCR if possible)
2. Process in batches (10-20 pages)
3. Use GPU if available (10-15x faster)
4. Explicit memory cleanup every 100 pages
5. Progress updates every batch

**Example**:
```python
import gc

for batch_num in range(0, total_pages, batch_size):
    batch_results = process_batch(batch_num, batch_num + batch_size)
    save_results(batch_results)

    if batch_num % 100 == 0:
        gc.collect()  # Explicit cleanup
```

### Troubleshooting

**OCR not initializing**:
- Check `pip list | grep paddle` shows paddleocr and paddlepaddle
- Try CPU mode first: `OCRService(gpu=False)`
- Check logs in stderr for specific errors

**GPU not being used**:
- Verify CUDA installation: `python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"`
- Check GPU memory: `nvidia-smi` (NVIDIA) or Task Manager (Windows)
- May need `paddlepaddle-gpu` instead of `paddlepaddle`

**Models not loading**:
- Delete cache and re-download: Remove `%LOCALAPPDATA%\PaddleOCR\`
- Check internet connection for first-run download
- For bundled models, verify structure matches `models/README.md`

**Poor accuracy**:
- Increase DPI: `pdf.render_page_to_image(page_num, dpi=600)`
- Check image quality and preprocessing
- Try different engine: `OCRService(engine="tesseract")`

### Large File Handling

For 5GB PDFs, follow these patterns:

1. **Stream Processing**: Process page-by-page or in small batches
2. **Incremental Output**: Don't buffer entire results in memory
3. **Progress Reporting**: Send frequent progress updates
4. **Temp Files**: Use disk for intermediate results
5. **Memory Management**: Clear processed pages from memory

Example in `pdf_processor.py`:
```python
for i in range(0, total_pages, BATCH_SIZE):
    batch = process_batch(i, i + BATCH_SIZE)
    send_progress(i, total_pages, f"Processing pages {i}-{i+BATCH_SIZE}")
    write_batch_to_disk(batch)
    del batch  # Free memory
```

## File System Permissions

Tauri requires explicit permission grants. Current permissions in `tauri.conf.json`:

- **fs.scope**: `["$APPDATA/*", "$DESKTOP/*", "$DOCUMENT/*", "$DOWNLOAD/*", "$HOME/*"]`
- **dialog**: File open/save dialogs enabled
- **shell.execute**: Enabled for spawning Python subprocess

When adding file operations, ensure paths are within the configured scope.

## Performance Considerations

### OCR Processing (Optimized for 4GB VRAM)

**Processing Speed by Hardware**:
- **PaddleOCR with 4GB GPU**: ~0.15-0.35 seconds per page (optimized)
- **PaddleOCR with 8GB+ GPU**: ~0.1-0.25 seconds per page (fastest)
- **PaddleOCR with CPU**: ~0.5-1 second per page (moderate)
- **Tesseract (CPU only)**: ~0.3-0.8 seconds per page (fast for simple documents)

**Large PDF Processing (5GB PDFs, 5000+ pages)**:
- **4GB VRAM + 16GB RAM**: ~8-20 minutes (hybrid mode)
- **4GB VRAM + 8GB RAM**: ~10-25 minutes (GPU only)
- **8GB+ VRAM**: ~8-15 minutes (GPU only, higher batch size)
- **CPU only (16GB RAM)**: ~40-80 minutes

**Optimizations**:
- Always check for text layer first with `has_text_layer()` to skip unnecessary OCR
- GPU auto-detection enabled by default (falls back to CPU gracefully)
- Real-time VRAM monitoring prevents OOM errors
- Adaptive batch sizing automatically adjusts to available memory
- Hybrid mode enables CPU offload on 4GB VRAM systems with 16GB+ RAM
- Make OCR optional and show time estimates to users

### Memory Usage
- PyMuPDF loads pages incrementally ✓
- Use `with` statements to ensure proper cleanup
- Avoid loading entire PDF into memory
- Process in adaptive batches:
  - **4GB VRAM**: 25 pages (2.5x improvement over previous 10)
  - **8GB VRAM**: 50 pages
  - **CPU (16GB RAM)**: 10 pages
- Automatic sub-batch splitting under memory pressure
- Aggressive garbage collection every 10 pages

### Progress Updates
- Send progress every N pages (not every page)
- Too frequent = performance overhead
- Too infrequent = poor UX
- Recommended: Every 10-20 pages or every 2-5 seconds

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

**Python**:
```bash
cd python-backend
venv\Scripts\activate  # or source venv/bin/activate
pip install <package>
pip freeze > requirements.txt
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

## Build Output

After `npm run tauri:build`:
- **Windows**: `src-tauri/target/release/bundle/msi/` (.msi installer)
- **macOS**: `src-tauri/target/release/bundle/dmg/` (.dmg)
- **Linux**: `src-tauri/target/release/bundle/appimage/` (.AppImage)

## Troubleshooting Common Issues

### "Python process not started"
- Ensure Python is in PATH
- Check venv is properly set up
- Verify `python` command works (try `python3` on Unix)

### Frontend can't call Tauri commands
- Ensure command is registered in `main.rs`
- Check command name spelling (case-sensitive)
- Verify function signature matches frontend call

### OCR not working
- Ensure EasyOCR is installed in venv
- Check GPU availability (if enabled)
- Verify image conversion works (pdf2image requires poppler)

### Out of memory
- Reduce `BATCH_SIZE` in Python code
- Close other applications
- Process smaller PDFs for testing

## Design Principles

1. **Local-First**: No cloud dependencies, all processing local
2. **Self-Contained**: All deps in project directory
3. **Streaming**: Handle large files without full buffering
4. **Progressive**: Show progress, don't block UI
5. **Fail-Safe**: Graceful error handling, informative messages
