#!/bin/bash

echo "========================================"
echo "Document De-Bundler - Setup Script"
echo "========================================"
echo

echo "[1/4] Setting up Python virtual environment..."
cd python-backend
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create Python virtual environment"
    echo "Make sure Python 3.8+ is installed"
    exit 1
fi

echo "[2/4] Activating virtual environment..."
source venv/bin/activate

echo "[3/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
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
