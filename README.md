# Document De-Bundler

A desktop application for processing, splitting, and organizing PDF documents with OCR capabilities.

## Features

- **PDF Processing**: Handle PDFs up to 5GB in size
- **OCR Support**: Optical character recognition for scanned documents
- **Intelligent Splitting**: Automatically recommend document splits
- **Smart Naming**: AI-powered naming suggestions for split documents
- **Batch Processing**: Process large documents efficiently with progress tracking
- **Local Processing**: All operations run locally - no cloud dependencies
- **Cross-Platform**: Built with Tauri for Windows, macOS, and Linux

## Tech Stack

- **Frontend**: Svelte + Vite + TypeScript + TailwindCSS
- **Desktop Framework**: Tauri (Rust)
- **Backend**: Python
- **PDF Processing**: PyMuPDF
- **OCR**: PaddleOCR (primary) / Tesseract (fallback)

## Prerequisites

### Required

- **Node.js** 18.x or higher ([Download](https://nodejs.org/))
- **Rust** 1.70 or higher ([Install](https://www.rust-lang.org/tools/install))
- **Python** 3.8 or higher ([Download](https://www.python.org/downloads/))

### For Optimal Performance

- **16GB+ RAM** recommended (8GB minimum)
- **GPU with 4GB+ VRAM** for 10-15x faster OCR (NVIDIA CUDA or AMD/Intel DirectML)
- **15GB+ free disk space** for temporary files during large PDF processing
- **Quad-core processor or better** for parallel processing

## Installation

### Quick Setup (Recommended)

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
# Activate venv first
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
│   │   ├── ocr_service.py     # OCR processing
│   │   ├── naming_service.py  # Document naming
│   │   └── bundler.py         # ZIP/folder creation
│   └── requirements.txt
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

## Self-Contained Design

This project is designed to be completely self-contained:

- **Python dependencies**: Installed in local `python-backend/venv/`
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
black .                 # Format code
flake8 .                # Lint code
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
- May need `paddlepaddle-gpu` for NVIDIA: `pip install paddlepaddle-gpu`

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
