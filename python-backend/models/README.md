# OCR Models Directory

This directory contains bundled OCR models for offline operation.

## Auto-Download (Default Behavior)

By default, PaddleOCR will automatically download models on first run (~8-10MB for English).

Models are downloaded to the user's cache directory:
- **Windows**: `%LOCALAPPDATA%\PaddleOCR\`
- **macOS**: `~/Library/Caches/PaddleOCR/`
- **Linux**: `~/.cache/PaddleOCR/`

## Pre-Bundling Models (Optional)

For offline installations or faster first-run experience, you can pre-bundle models.

### PaddleOCR Model Structure

Create the following directory structure:

```
models/
├── det/                    # Text detection model
│   ├── inference.pdiparams
│   ├── inference.pdmodel
│   └── ...
├── rec/                    # Text recognition model
│   ├── inference.pdiparams
│   ├── inference.pdmodel
│   └── ...
└── cls/                    # Angle classification model (optional)
    ├── inference.pdiparams
    ├── inference.pdmodel
    └── ...
```

### Downloading Models

**Method 1: Let PaddleOCR download (recommended)**
1. Run the application once
2. Models will be downloaded automatically
3. Copy from cache directory to `python-backend/models/`

**Method 2: Manual download**

Download English models from PaddleOCR GitHub:
- Detection: [en_PP-OCRv4_det](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/models_list_en.md#1-text-detection-model)
- Recognition: [en_PP-OCRv4_rec](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/models_list_en.md#2-text-recognition-model)
- Angle Classifier: [ch_ppocr_mobile_v2.0_cls](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/models_list_en.md#3-text-angle-classification-model)

Extract each model into its respective directory (det/, rec/, cls/).

### Model Sizes

- **Detection model**: ~3-4 MB
- **Recognition model**: ~4-5 MB
- **Angle classifier**: ~1-2 MB
- **Total**: ~8-11 MB

### Using Bundled Models

The OCR service automatically checks for models in this directory. If found, it uses them instead of downloading.

No code changes needed - the configuration in `services/ocr/config.py` handles this automatically.

### Additional Languages

To support additional languages:

1. Download language-specific models from [PaddleOCR Model List](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/models_list_en.md)
2. Place in subdirectories: `models/{lang}_det/`, `models/{lang}_rec/`
3. Update OCR configuration to use the language code

Supported languages: English, Chinese, French, German, Japanese, Korean, and 80+ more.

## Tesseract Models

Tesseract uses system-installed language data files.

**Language data locations:**
- **Windows**: `C:\Program Files\Tesseract-OCR\tessdata\`
- **macOS**: `/usr/local/share/tessdata/` or `/opt/homebrew/share/tessdata/`
- **Linux**: `/usr/share/tesseract-ocr/*/tessdata/`

Download additional languages from: https://github.com/tesseract-ocr/tessdata

## Size Considerations for Distribution

### Minimal Bundle (No Pre-bundled Models)
- Application size: ~550MB (includes PaddlePaddle, OpenCV, PyMuPDF)
- First-run download: ~8-10MB
- **Recommended for** most users

### Full Bundle (Pre-bundled English Models)
- Application size: ~560MB
- No first-run download needed
- **Recommended for** offline installations or corporate environments

### Multi-Language Bundle
- Add ~5-10MB per additional language
- Only bundle languages users actually need

## Production Deployment

For production installers (PyInstaller, cx_Freeze):

1. **Include models** in the bundle by adding to PyInstaller spec:
```python
datas=[
    ('python-backend/models', 'models'),
],
```

2. **Configure path** in code to find bundled models:
```python
import sys
from pathlib import Path

if getattr(sys, 'frozen', False):
    # Running as compiled executable
    bundle_dir = Path(sys._MEIPASS)
    model_dir = bundle_dir / 'models'
else:
    # Running as script
    model_dir = Path(__file__).parent / 'models'
```

The current code already handles this via `get_model_directory()` in `services/ocr/config.py`.

## .gitignore

Model files are gitignored to keep the repository lightweight. Users either:
1. Auto-download on first run (default)
2. Download and place models manually for offline use
3. CI/CD pipelines download during build process

To track specific models in git (not recommended):
```bash
# In .gitignore, add exception:
!python-backend/models/det/
!python-backend/models/rec/
```

## Troubleshooting

### Models not loading
- Check directory structure matches above
- Verify model files are complete (not corrupted)
- Check file permissions

### Models still downloading despite bundling
- Ensure models are in correct subdirectories (det/, rec/, cls/)
- Check logs for model loading errors
- Verify `model_dir` configuration is set correctly

### Large application size
- Don't bundle unnecessary language models
- Consider using lightweight Tesseract instead for CPU-only scenarios
- Compress models (though PaddleOCR models are already compressed)
