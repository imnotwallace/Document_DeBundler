#!/bin/bash

echo "========================================"
echo "Document De-Bundler - Setup Script"
echo "========================================"
echo

echo "[1/4] Setting up Python virtual environment..."
cd python-backend
uv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create Python virtual environment"
    echo "Make sure uv is installed: pip install uv"
    exit 1
fi

echo "[2/4] Activating virtual environment..."
source venv/bin/activate

echo "[3/4] Installing Python dependencies (with uv - fast!)..."
uv pip sync requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Python dependencies"
    exit 1
fi

cd ..

echo "[4/4] Installing Node.js dependencies..."
npm install
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Node.js dependencies"
    echo "Make sure Node.js and npm are installed"
    exit 1
fi

echo
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo
echo "To start development:"
echo "  1. Run: npm run tauri:dev"
echo
echo "OPTIONAL: Pre-install embedding models (for offline use)"
echo "  - Models will auto-download on first use (~1.15GB)"
echo "  - To pre-install: cd python-backend && python download_embedding_models.py"
echo "  - This downloads Nomic Embed v1.5 models locally"
echo
