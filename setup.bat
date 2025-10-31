@echo off
echo ========================================
echo Document De-Bundler - Setup Script
echo ========================================
echo.

echo [1/4] Setting up Python virtual environment...
cd python-backend
uv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create Python virtual environment
    echo Make sure uv is installed: pip install uv
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing Python dependencies (with uv - fast!)...
uv pip sync requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)

cd ..

echo [4/4] Installing Node.js dependencies...
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Node.js dependencies
    echo Make sure Node.js and npm are installed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To start development:
echo   1. Run: npm run tauri:dev
echo.
pause
