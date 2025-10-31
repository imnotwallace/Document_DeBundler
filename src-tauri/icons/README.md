# App Icons

This directory should contain app icons for Tauri builds.

## Required Icons

Generate icons using Tauri's icon generator:

```bash
npm install -g @tauri-apps/cli
tauri icon path/to/source-icon.png
```

This will generate:
- 32x32.png
- 128x128.png
- 128x128@2x.png
- icon.icns (macOS)
- icon.ico (Windows)

## Source Image Requirements

- PNG format
- 1024x1024 or larger
- Square aspect ratio
- Transparent background recommended

For now, the app will use Tauri's default icon during development.
