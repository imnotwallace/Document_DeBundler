# Document De-Bundler

A desktop application for processing, splitting, and organizing PDF documents with OCR capabilities.

## Features

- **PDF Processing**: Handle PDFs up to 5GB in size
- **OCR Support**: Optical character recognition for scanned documents with GPU acceleration
- **Document De-Bundling**: ML-powered intelligent document separation using semantic analysis
- **Intelligent Splitting**: Automatically recommend document splits using AI and content analysis
- **Smart Naming**: AI-powered naming suggestions for split documents via LLM integration
- **Semantic Analysis**: Content-based document grouping using Nomic Embed v1.5 embeddings
- **Batch Processing**: Process large documents efficiently with progress tracking
- **VRAM Monitoring**: Real-time memory management prevents out-of-memory errors
- **Local Processing**: All operations run locally - no cloud dependencies
- **Cross-Platform**: Built with Tauri 2.0 for Windows, macOS, and Linux

## Tech Stack

- **Frontend**: Svelte + Vite + TypeScript + TailwindCSS
- **Desktop Framework**: Tauri 2.0 (Rust)
- **Backend**: Python 3.8+
- **Dependency Management**: uv (10-100x faster than pip)
- **PDF Processing**: PyMuPDF (fitz)
- **OCR**: PaddleOCR 2.7.3+ with PaddlePaddle 3.0+ (primary) / Tesseract (fallback)
- **ML/AI**: 
  - sentence-transformers (Nomic Embed v1.5 for semantic analysis)
  - scikit-learn (DBSCAN clustering for document grouping)
  - LLM integration for intelligent naming and splitting suggestions

## Prerequisites

### Required

- **Node.js** 18.x or higher ([Download](https://nodejs.org/))
- **Rust** 1.70 or higher ([Install](https://www.rust-lang.org/tools/install))
- **Python** 3.8 or higher ([Download](https://www.python.org/downloads/))
- **uv** (Recommended): Fast Python package manager ([Install](https://github.com/astral-sh/uv)): `pip install uv`
  - 10-100x faster than pip for dependency installation
  - Optional but highly recommended for quick setup

### For Optimal Performance

- **16GB+ RAM** recommended (8GB minimum)
- **GPU with 4GB+ VRAM** for 10-15x faster OCR (NVIDIA CUDA or AMD/Intel DirectML)
- **15GB+ free disk space** for temporary files during large PDF processing
- **Quad-core processor or better** for parallel processing

## Installation

### Quick Setup (Recommended)

The setup scripts use **uv** for fast dependency installation (10-100x faster than pip). If you don't have uv installed, run `pip install uv` first.

#### Windows
```bash
setup.bat
```

#### macOS/Linux
```bash
chmod +x setup.sh
./setup.sh
```

### Manual Setup

1. **Set up Python virtual environment**

**Option A: Using uv (Recommended - Fast)**
```bash
cd python-backend
uv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies with uv (10-100x faster)
uv pip sync requirements.txt
cd ..
```

**Option B: Using standard pip**
```bash
cd python-backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
cd ..
```

2. **Install Node.js dependencies**
```bash
npm install
```

## Development

### Running the App

```bash
npm run tauri:dev
```

This will:
1. Start the Vite dev server for the frontend
2. Start the Tauri application
3. Hot reload on code changes

### Building for Production

```bash
npm run tauri:build
```

Output will be in `src-tauri/target/release/bundle/`

### Testing

#### Python Tests
```bash
cd python-backend

# Option A: With uv (no activation needed, faster)
uv run pytest
uv run pytest --cov  # with coverage

# Option B: With activated venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
pytest
```

#### Frontend Type Checking
```bash
npm run check
```

## Project Structure

```
Document-De-Bundler/
├── src/                      # Svelte frontend source
│   ├── App.svelte           # Main app component
│   ├── lib/                 # Components and utilities
│   └── main.ts              # Frontend entry point
│
├── src-tauri/               # Rust/Tauri backend
│   ├── src/
│   │   ├── main.rs          # Tauri app entry
│   │   ├── commands.rs      # Tauri commands (Rust → Frontend)
│   │   └── python_bridge.rs # Python subprocess management
│   └── Cargo.toml           # Rust dependencies
│
├── python-backend/          # Python processing engine
│   ├── main.py             # IPC entry point (stdin/stdout)
│   ├── services/
│   │   ├── pdf_processor.py   # PDF analysis & splitting
│   │   ├── ocr_service.py     # OCR processing wrapper
│   │   ├── ocr/               # OCR abstraction layer
│   │   │   ├── base.py           # OCR engine interface
│   │   │   ├── config.py         # Hardware detection
│   │   │   ├── manager.py        # Engine lifecycle
│   │   │   ├── vram_monitor.py   # Real-time memory tracking
│   │   │   ├── text_quality.py   # OCR quality analysis
│   │   │   └── engines/          # PaddleOCR & Tesseract
│   │   ├── llm/               # LLM integration module
│   │   │   ├── config.py         # LLM configuration
│   │   │   └── prompts.py        # Prompt templates
│   │   ├── split_detection.py  # ML-based document splitting
│   │   ├── embedding_service.py # Semantic analysis (Nomic Embed)
│   │   ├── naming_service.py   # Document naming
│   │   ├── bundler.py          # ZIP/folder creation
│   │   ├── cache_manager.py    # Performance caching
│   │   └── resource_path.py    # Bundled resource management
│   ├── models/             # OCR model storage (optional)
│   ├── bin/                # Bundled binaries (Tesseract)
│   └── requirements.txt
│
├── docs/                   # Implementation documentation
│   ├── DEBUNDLING_QUICK_START.md
│   ├── EMBEDDING_SERVICE_IMPLEMENTATION.md
│   ├── IMPLEMENTATION_SPEC_DEBUNDLING.md
│   └── SPLIT_DETECTION_IMPLEMENTATION_REPORT.md
│
└── package.json            # Node.js dependencies
```

## How It Works

1. **User Interface**: Svelte frontend running in Tauri webview
2. **File Selection**: Native file picker via Tauri API
3. **Processing**: Rust backend spawns Python subprocess
4. **Communication**: JSON-based IPC via stdin/stdout
5. **Progress Updates**: Real-time updates via event streaming
6. **Output**: Split PDFs saved to user-selected location

## Document De-Bundling

The application includes advanced **ML-powered document separation** capabilities for intelligently splitting bundled PDFs:

### How It Works

1. **Content Analysis**: Extracts text from each page using OCR or existing text layers
2. **Semantic Embeddings**: Generates vector embeddings using Nomic Embed v1.5 (sentence-transformers)
3. **Similarity Analysis**: Computes cosine similarity between consecutive pages to detect topic boundaries
4. **Clustering**: Uses DBSCAN algorithm to group related pages into coherent documents
5. **Smart Splitting**: Automatically suggests split points based on content discontinuities
6. **LLM Integration**: Provides intelligent naming suggestions for separated documents

### Key Features

- **Content-Based Separation**: Understands document semantics, not just visual breaks
- **Configurable Sensitivity**: Adjust similarity thresholds for different document types
- **Hybrid Approach**: Combines visual markers with semantic analysis for best results
- **Local Processing**: All ML operations run locally with no cloud dependencies

### Performance

- **CPU Mode**: ~1-3 seconds per page for embedding generation
- **GPU Mode**: ~0.3-0.8 seconds per page (CUDA/DirectML)
- **Smart Caching**: Previously analyzed documents use cached embeddings

For detailed implementation, see `docs/DEBUNDLING_QUICK_START.md`

## Self-Contained Design

This project is designed to be completely self-contained:

- **Python dependencies**: Installed in local `python-backend/venv/` (managed by uv or pip)
- **Node.js dependencies**: Installed in local `node_modules/`
- **Rust dependencies**: Managed by Cargo in `src-tauri/target/`
- **No global installations required** (except prerequisites)
- **No system pollution**: Delete the folder to completely remove

## Common Commands

```bash
# Development
npm run dev              # Frontend only (for UI development)
npm run tauri:dev        # Full app with hot reload

# Building
npm run build            # Build frontend
npm run tauri:build      # Build complete application

# Type checking
npm run check            # Svelte type checking

# Python (from python-backend/, with venv activated)
pytest                   # Run tests
pytest --cov            # Run tests with coverage

# Python with uv (faster, no activation needed)
cd python-backend
uv run pytest           # Run tests with uv
uv run pytest --cov     # Run tests with coverage
uv pip install <package> # Add new package
uv pip sync requirements.txt # Sync dependencies

# Code quality (with venv activated)
black .                 # Format code
flake8 .                # Lint code                # Lint code
```

## System Requirements

### Minimum Configuration (Basic Processing)
- **CPU**: Dual-core processor (2.0 GHz or higher)
- **RAM**: 8GB system memory
- **Storage**: 10GB free disk space
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Processing Capability**: Can handle most PDFs with CPU-only OCR
- **Expected Performance**: ~40-80 minutes for 5GB PDFs (5000+ pages)

### Recommended Configuration (Optimal Performance)
- **CPU**: Quad-core processor (2.5 GHz or higher)
- **RAM**: 16GB system memory
- **GPU**: 4GB VRAM (NVIDIA with CUDA or AMD/Intel with DirectML)
- **Storage**: 15GB+ free disk space
- **Processing Capability**: GPU-accelerated OCR with hybrid CPU fallback
- **Expected Performance**: ~8-20 minutes for 5GB PDFs (5000+ pages)

### High-Performance Configuration (Best Experience)
- **CPU**: 6+ core processor (3.0 GHz or higher)
- **RAM**: 32GB system memory
- **GPU**: 8GB+ VRAM (NVIDIA with CUDA or AMD/Intel with DirectML)
- **Storage**: 20GB+ free disk space
- **Processing Capability**: Maximum GPU acceleration with aggressive parallel processing
- **Expected Performance**: ~8-15 minutes for 5GB PDFs (5000+ pages)

### GPU Acceleration Details

The application automatically detects and configures GPU acceleration:

- **PaddleOCR** (Primary Engine):
  - Best accuracy (95-98%) and performance
  - 2-3x faster than alternatives
  - GPU support: NVIDIA CUDA, AMD/Intel DirectML (Windows)
  - ~0.15-0.35 seconds per page with 4GB VRAM
  - ~0.1-0.25 seconds per page with 8GB+ VRAM
  - Models auto-download on first run (~10MB)

- **Tesseract** (Fallback Engine):
  - CPU-only, lightweight option
  - Good for simple documents
  - ~0.3-0.8 seconds per page
  - No additional model downloads required

### Memory Management

The application includes intelligent memory management:

- **4GB VRAM + 8GB RAM**: GPU-only mode, batch size 25 pages
- **4GB VRAM + 16GB RAM**: Hybrid mode enabled, automatic CPU offload on memory pressure
- **4GB VRAM + 32GB RAM**: Optimal hybrid mode with aggressive parallel processing
- **8GB+ VRAM**: Higher batch sizes (50+ pages), fastest processing
- **CPU-only (16GB RAM)**: Batch size 10 pages, moderate performance
- **CPU-only (32GB+ RAM)**: Batch size 20 pages, improved performance

Real-time VRAM monitoring prevents out-of-memory errors and automatically adjusts processing strategies.

## Troubleshooting

### Python venv not activating
- Make sure Python is in your PATH
- Try using `python3` instead of `python`
- If using uv: Ensure uv is installed (`pip install uv`)
- Alternative: Use `uv run` commands without activating venv

### Rust/Cargo errors
- Run `rustup update` to update Rust
- Check Tauri prerequisites: https://tauri.app/v1/guides/getting-started/prerequisites

### OCR Performance Issues

**Slow Processing:**
- GPU acceleration auto-detects and enables if available
- Verify GPU is active: Check processing logs for "GPU: enabled"
- For NVIDIA GPUs, ensure CUDA drivers are installed
- For AMD/Intel GPUs on Windows, DirectML is used automatically
- CPU fallback happens automatically if GPU unavailable

**GPU Not Detected:**
- Check GPU drivers are up to date
- NVIDIA: Install CUDA toolkit (10.2 or higher)
- AMD/Intel (Windows): DirectML included with Windows 10/11
- Verify in Python: `python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"`
- May need `paddlepaddle-gpu` for NVIDIA: `pip install paddlepaddle-gpu` or `uv pip install paddlepaddle-gpu`
- **Note**: Project uses PaddlePaddle 3.0+ (breaking changes from 2.x)

**Poor OCR Accuracy:**
- PaddleOCR provides 95-98% accuracy on most documents
- Check document quality (low resolution or poor scans affect results)
- System automatically increases DPI for better quality when memory allows
- Text layer detection skips OCR when PDF already contains searchable text

### Memory Issues

**Out of Memory Errors:**
- Application automatically adjusts batch size based on available memory
- Real-time VRAM monitoring prevents most OOM errors
- If persistent, close other GPU-intensive applications
- On 4GB VRAM systems with 16GB+ RAM, hybrid mode offloads to CPU automatically
- For very large PDFs (10GB+), consider splitting into smaller files first

**High Memory Usage:**
- Normal for large PDF processing (temporary)
- Memory is released after each batch completes
- Aggressive garbage collection every 10 pages
- Check available disk space for temporary files (requires 2-3x PDF size)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
