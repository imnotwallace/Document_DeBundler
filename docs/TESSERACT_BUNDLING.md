# Tesseract OCR Bundling Guide

This document explains how Tesseract OCR is bundled with the Document De-Bundler application to ensure offline, dependency-free operation.

## Overview

The application now bundles Tesseract OCR executable and language data files, eliminating the need for users to install Tesseract separately on their systems. This ensures:

- **No system dependencies**: App works out-of-the-box
- **Consistent versions**: Everyone uses the same Tesseract version
- **Offline operation**: No internet required after initial app download
- **Controlled updates**: Tesseract updates are managed with app releases

## Architecture

### Directory Structure

```
python-backend/
└── bin/
    ├── README.md                      # Setup instructions
    ├── .gitignore                     # Prevents committing large binaries
    └── tesseract/
        ├── tesseract.exe              # Main executable (Windows)
        ├── *.dll                      # Required dependencies
        └── tessdata/
            ├── .gitkeep               # Preserves directory in git
            ├── eng.traineddata        # English language data
            ├── fra.traineddata        # French language data
            └── ...                    # Other languages
```

### Resource Path Resolution

The `python-backend/services/resource_path.py` module handles path resolution:

- **Development mode**: Uses system Tesseract (if available)
- **Production mode**: Uses bundled Tesseract from `bin/tesseract/`
- **Auto-detection**: Automatically detects which mode is active

### Modified Components

1. **tesseract_engine.py**: Updated to use bundled executable
   - Configures `pytesseract.pytesseract.tesseract_cmd`
   - Sets `TESSDATA_PREFIX` environment variable
   - Falls back to system Tesseract in development

2. **tauri.conf.json**: Configured to bundle resources
   - Added `python-backend/bin/**/*` to resources
   - Binaries are included in final application package

3. **resource_path.py**: New utility module
   - Detects production vs development mode
   - Resolves paths to bundled resources
   - Verifies Tesseract setup

## Setup Instructions

### For Development

1. **Option A: Use System Tesseract** (recommended for development)
   - Install Tesseract on your system
   - Application will automatically use it
   - No need to populate `bin/tesseract/`

2. **Option B: Use Bundled Tesseract** (test production setup)
   - Follow production setup instructions below
   - Application will use bundled version even in development

### For Building Production Release

Before building the application for distribution:

1. **Download Tesseract Binaries**

   **Windows (x64):**
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Or direct: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe

2. **Extract Required Files**

   From the Tesseract installation directory, copy:

   **Executable:**
   ```
   tesseract.exe → python-backend/bin/tesseract/
   ```

   **DLL Dependencies:**
   ```
   liblept-5.dll          → python-backend/bin/tesseract/
   libarchive-13.dll      → python-backend/bin/tesseract/
   libcurl-4.dll          → python-backend/bin/tesseract/
   libgcc_s_seh-1.dll     → python-backend/bin/tesseract/
   libgomp-1.dll          → python-backend/bin/tesseract/
   libiconv-2.dll         → python-backend/bin/tesseract/
   libintl-8.dll          → python-backend/bin/tesseract/
   libjpeg-8.dll          → python-backend/bin/tesseract/
   liblzma-5.dll          → python-backend/bin/tesseract/
   libpng16-16.dll        → python-backend/bin/tesseract/
   libstdc++-6.dll        → python-backend/bin/tesseract/
   libtiff-6.dll          → python-backend/bin/tesseract/
   libwebp-7.dll          → python-backend/bin/tesseract/
   zlib1.dll              → python-backend/bin/tesseract/
   ```

   **Language Data Files:**
   ```
   tessdata/*.traineddata → python-backend/bin/tesseract/tessdata/
   ```

3. **Verify Setup**

   Run the test script:
   ```bash
   cd python-backend
   python test_resource_path.py
   ```

   Should show:
   - Tesseract path found
   - Tessdata path found
   - Mode: bundled (if binaries installed)

4. **Build Application**

   ```bash
   npm run tauri:build
   ```

   The bundled binaries will be automatically included in the installer.

## Language Support

### Minimal Bundle (English Only)

For smallest file size, include only:
- `eng.traineddata` (~10MB)

Total bundle size: ~20MB

### Recommended Languages

Common languages for international use:
- `eng.traineddata` - English
- `fra.traineddata` - French
- `deu.traineddata` - German
- `spa.traineddata` - Spanish
- `ita.traineddata` - Italian
- `por.traineddata` - Portuguese

Total bundle size: ~50-60MB

### Full Language Support

Copy all `.traineddata` files from Tesseract installation.

Total bundle size: ~100-150MB

## How It Works

### Runtime Path Resolution

1. **Startup**: Python backend imports `resource_path` module

2. **Configuration**: `setup_tesseract_environment()` is called
   - Detects if running in bundled app (production) or development
   - Checks for bundled Tesseract in `bin/tesseract/`
   - Returns configuration with paths

3. **Initialization**: `tesseract_engine.py` uses configuration
   - Sets `pytesseract.pytesseract.tesseract_cmd` to bundled executable
   - Sets `TESSDATA_PREFIX` to bundled language data
   - Falls back to system Tesseract if bundled not found

4. **Verification**: `verify_tesseract_setup()` checks
   - Executable exists and is accessible
   - Tessdata directory exists
   - At least one `.traineddata` file is present

### Production vs Development

| Aspect | Development | Production |
|--------|------------|-----------|
| Binary location | System PATH or `bin/tesseract/` | Always `bin/tesseract/` |
| Fallback | System Tesseract if bundled missing | No fallback |
| Path detection | `sys.frozen = False` | `sys.frozen = True` |
| Testing | Can test either mode | Bundled only |

## Troubleshooting

### "Tesseract executable not found"

**Cause**: `tesseract.exe` not in `bin/tesseract/` or system PATH

**Solution**:
- Development: Install system Tesseract or populate `bin/tesseract/`
- Production: Ensure binaries were copied before building

### "Missing DLL" errors

**Cause**: Required DLL dependencies not copied

**Solution**:
- Copy ALL DLL files from Tesseract installation
- Check Tesseract version matches (5.x recommended)
- Use Dependency Walker to identify missing DLLs

### "Failed to load language data"

**Cause**: `.traineddata` files missing or corrupted

**Solution**:
- Verify `tessdata/` directory exists
- Ensure at least `eng.traineddata` is present
- Redownload language files if corrupted

### "OCR returns empty results"

**Cause**: Language code doesn't match available data files

**Solution**:
- Check requested language code (e.g., 'eng', 'fra')
- Ensure corresponding `.traineddata` file exists
- Verify file permissions

### Test script shows "system" mode but bundled version exists

**Cause**: Binaries might not be executable or accessible

**Solution**:
- Check file permissions
- Ensure `tesseract.exe` is directly in `bin/tesseract/`
- Run test script with elevated permissions

## Git Handling

### What's Tracked

- Directory structure (`bin/`, `bin/tesseract/`, `bin/tesseract/tessdata/`)
- Documentation (`bin/README.md`)
- Configuration files (`.gitignore`, `.gitkeep`)

### What's Ignored

- `tesseract.exe` (binary, ~50MB)
- `*.dll` files (binaries, ~20-30MB total)
- `*.traineddata` files (language data, 10-100MB+)

### Why Binaries Aren't Committed

1. **File size**: 100-200MB total (too large for git)
2. **Updates**: Easier to update binaries independently
3. **Licensing**: Clearer separation of our code vs third-party binaries
4. **Platforms**: Would need different binaries per platform

### Distribution

For releases, binaries are:
1. Downloaded by developer before building
2. Bundled by Tauri during build process
3. Included in final installer (.msi, .dmg, .AppImage)
4. End users get complete package with bundled Tesseract

## Future Enhancements

### Multi-Platform Support

Currently configured for Windows only. To support macOS/Linux:

1. Add platform detection to `resource_path.py`
2. Include binaries for each platform in `bin/tesseract/`
3. Update path resolution logic for platform-specific executable names
4. Test on each target platform

### Auto-Download

Could implement automatic binary download:

1. Check for bundled Tesseract on first run
2. If missing, prompt user to download
3. Download from official source
4. Extract and configure automatically

### Language Pack Management

Could allow users to download language packs on-demand:

1. Ship with English only by default
2. Detect required languages during OCR
3. Prompt to download missing language packs
4. Download and install to user data directory

## Testing

### Test Resource Path Module

```bash
cd python-backend
python test_resource_path.py
```

Expected output:
- Base path detected correctly
- Bin path exists
- Tesseract path (bundled or None)
- Mode: 'bundled' or 'system'

### Test Tesseract Engine

```bash
cd python-backend
python -c "from services.ocr.engines.tesseract_engine import TesseractEngine; from services.ocr.base import OCRConfig; engine = TesseractEngine(OCRConfig()); engine.initialize(); print('Success!')"
```

Expected: "Success!" (with log messages showing configuration)

### Test in Built Application

After building:
1. Install the application
2. Run OCR on a test document
3. Check logs for "Using bundled Tesseract" message
4. Verify OCR results are correct

## License Considerations

### Tesseract OCR

- License: Apache License 2.0
- Source: https://github.com/tesseract-ocr/tesseract
- Distribution: Allowed with attribution

### Language Data Files

- Various licenses (check individual files)
- Generally permissive (Apache, custom open-source)
- Distribution: Usually allowed with attribution

### This Application

When distributing, include:
- Tesseract license in about/licenses section
- Attribution for Tesseract project
- Links to source repositories

## Summary

The Tesseract bundling system:

✅ Eliminates system dependencies
✅ Ensures consistent user experience
✅ Supports offline operation
✅ Works in development and production
✅ Gracefully falls back in development
✅ Easy to update (replace binaries and rebuild)
✅ Tested and verified

The application is now self-contained and ready for distribution with full OCR capabilities built-in.
