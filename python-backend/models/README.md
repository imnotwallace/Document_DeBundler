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

**Method 1: Use the automated download script (easiest)**

Run the provided download script:
```bash
# From project root
cd python-backend
python download_models.py
```

This will automatically download and extract all three models to the correct directories.

**Method 2: Let PaddleOCR download (simple but requires internet on first run)**
1. Run the application once without bundled models
2. PaddleOCR will download models automatically to cache
3. Optionally copy from cache directory to `python-backend/models/` for future use

**Method 3: Manual download (for offline environments)**

Download these three English models:

**1. Detection Model** (English PP-OCRv3, ~3.9 MB):
```bash
# Download
curl -L https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar -o det_model.tar

# Extract to det/ directory
tar -xf det_model.tar
mv en_PP-OCRv3_det_infer/* det/
rm -rf en_PP-OCRv3_det_infer det_model.tar
```

**Windows PowerShell:**
```powershell
# Download
Invoke-WebRequest -Uri "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar" -OutFile "det_model.tar"

# Extract (requires tar.exe available in Windows 10+)
tar -xf det_model.tar
Move-Item -Path "en_PP-OCRv3_det_infer\*" -Destination "det\" -Force
Remove-Item -Recurse -Force en_PP-OCRv3_det_infer, det_model.tar
```

**2. Recognition Model** (English PP-OCRv4, ~8.6 MB):
```bash
# Download
curl -L https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar -o rec_model.tar

# Extract to rec/ directory
tar -xf rec_model.tar
mv en_PP-OCRv4_rec_infer/* rec/
rm -rf en_PP-OCRv4_rec_infer rec_model.tar
```

**Windows PowerShell:**
```powershell
# Download
Invoke-WebRequest -Uri "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar" -OutFile "rec_model.tar"

# Extract
tar -xf rec_model.tar
Move-Item -Path "en_PP-OCRv4_rec_infer\*" -Destination "rec\" -Force
Remove-Item -Recurse -Force en_PP-OCRv4_rec_infer, rec_model.tar
```

**3. Angle Classification Model** (Multilingual, ~2.2 MB):
```bash
# Download
curl -L https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar -o cls_model.tar

# Extract to cls/ directory
tar -xf cls_model.tar
mv ch_ppocr_mobile_v2.0_cls_infer/* cls/
rm -rf ch_ppocr_mobile_v2.0_cls_infer cls_model.tar
```

**Windows PowerShell:**
```powershell
# Download
Invoke-WebRequest -Uri "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar" -OutFile "cls_model.tar"

# Extract
tar -xf cls_model.tar
Move-Item -Path "ch_ppocr_mobile_v2.0_cls_infer\*" -Destination "cls\" -Force
Remove-Item -Recurse -Force ch_ppocr_mobile_v2.0_cls_infer, cls_model.tar
```

**Direct Download URLs** (for browser download):
- Detection: `https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar`
- Recognition: `https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar`
- Classification: `https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar`

### Verifying Installation

After downloading, verify the directory structure:
```
python-backend/models/
├── det/
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   ├── inference.pdmodel
│   └── inference.yml
├── rec/
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   ├── inference.pdmodel
│   └── inference.yml
└── cls/
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    ├── inference.pdmodel
    └── inference.yml
```

You can verify with:
```bash
# Unix/macOS/Linux
ls -R python-backend/models/

# Windows
tree /F python-backend\models
```

Each model directory should contain at least:
- `inference.pdmodel` - Model structure
- `inference.pdiparams` - Model weights
- `inference.yml` or `inference.json` - Configuration (PaddleOCR 3.0+)

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
