# Bundled Binaries

This directory contains external binaries bundled with the application to ensure it runs without system dependencies.

## Tesseract OCR

### Directory Structure

```
bin/
└── tesseract/
    ├── tesseract.exe           # Main Tesseract executable
    ├── *.dll                   # Required DLL dependencies
    └── tessdata/               # Language data files
        ├── eng.traineddata     # English language data
        ├── fra.traineddata     # French language data
        ├── deu.traineddata     # German language data
        ├── spa.traineddata     # Spanish language data
        └── ...                 # Other language files
```

### Setup Instructions

#### Option 1: Download Pre-built Binaries (Recommended)

1. Download Tesseract Windows binaries from:
   - Official: https://github.com/UB-Mannheim/tesseract/wiki
   - Direct link (x64): https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe

2. Install or extract the files

3. Copy required files to this directory:
   ```
   From Tesseract installation directory (e.g., C:\Program Files\Tesseract-OCR\):

   Copy to python-backend/bin/tesseract/:
   - tesseract.exe
   - liblept-5.dll (or similar)
   - libarchive-13.dll
   - libcurl-4.dll
   - libgcc_s_seh-1.dll
   - libgomp-1.dll
   - libiconv-2.dll
   - libintl-8.dll
   - libjpeg-8.dll
   - liblzma-5.dll
   - libpng16-16.dll
   - libstdc++-6.dll
   - libtiff-6.dll
   - libwebp-7.dll
   - zlib1.dll

   Copy to python-backend/bin/tesseract/tessdata/:
   - All .traineddata files from tessdata/ directory
   ```

#### Option 2: Use Portable Package

1. Download portable Tesseract package
2. Extract entire contents to `python-backend/bin/tesseract/`
3. Ensure `tesseract.exe` is directly in `tesseract/` folder
4. Ensure `tessdata/` folder is inside `tesseract/` folder

### Language Data Files

Common language codes:
- `eng` - English
- `fra` - French
- `deu` - German
- `spa` - Spanish
- `ita` - Italian
- `por` - Portuguese
- `rus` - Russian
- `chi_sim` - Chinese Simplified
- `chi_tra` - Chinese Traditional
- `jpn` - Japanese
- `ara` - Arabic

For a minimal bundle (English only):
- Download only `eng.traineddata` (~10MB)

For full language support:
- Copy all `.traineddata` files from Tesseract installation (~100MB+)

### Verification

After setting up, verify the structure:

```bash
# Windows PowerShell
dir python-backend\bin\tesseract\

# Should show:
# - tesseract.exe
# - Multiple .dll files
# - tessdata\ directory with .traineddata files
```

### Important Notes

1. **Required Files**:
   - `tesseract.exe` is mandatory
   - DLL files are mandatory (missing DLLs will cause runtime errors)
   - At least one `.traineddata` file is mandatory (e.g., `eng.traineddata`)

2. **Version Compatibility**:
   - Tested with Tesseract 5.x
   - Should work with Tesseract 4.x
   - DLL dependencies may vary by version

3. **Build Process**:
   - These files will be automatically bundled when building the application
   - The application will use these bundled binaries instead of system Tesseract
   - Development mode can still use system Tesseract if this directory is empty

4. **License**:
   - Tesseract is licensed under Apache License 2.0
   - Language data files have various licenses (check tessdata repository)
   - Ensure compliance when distributing

### Troubleshooting

**"tesseract.exe not found"**:
- Ensure `tesseract.exe` is directly in `python-backend/bin/tesseract/`
- Check file permissions

**"Missing DLL" errors**:
- Copy ALL DLL files from Tesseract installation
- Use Dependency Walker to identify missing dependencies

**"Failed to load language data"**:
- Check `tessdata/` folder exists
- Ensure at least `eng.traineddata` is present
- Verify file permissions

**OCR returns empty results**:
- Check language code matches available `.traineddata` files
- Verify `.traineddata` files are not corrupted (redownload if needed)

### Development vs Production

- **Development**: System Tesseract is used if available (fallback)
- **Production**: Bundled Tesseract is always used
- Path resolution automatically detects running mode

## Future Binaries

Other binaries can be added to this directory following the same pattern:
```
bin/
├── tesseract/
├── another-tool/
└── README.md (this file)
```
