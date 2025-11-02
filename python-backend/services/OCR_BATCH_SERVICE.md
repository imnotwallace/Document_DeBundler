# OCR Batch Service Documentation

## Overview

The OCR Batch Service (`ocr_batch_service.py`) is a production-ready service for processing multiple PDF files with optical character recognition (OCR). It provides adaptive memory management, error recovery, progress tracking, and cancellation support.

## Features

- **Adaptive Memory Management**: Automatically detects GPU VRAM and system RAM to optimize batch sizes
- **VRAM Monitoring**: Real-time GPU memory tracking to prevent out-of-memory errors
- **Exponential Backoff Retry**: Automatic retry with 1s, 2s, 4s delays (max 3 attempts)
- **Text Layer Detection**: Skips OCR for pages with existing text layers (10-100x faster)
- **Progress Tracking**: Detailed progress updates with ETA calculation
- **Cancellation Support**: Graceful shutdown with partial result preservation
- **Memory Cleanup**: Explicit garbage collection and GPU cache clearing
- **Error Recovery**: Per-page error handling with continued processing

## Target Hardware

Optimized for 4GB VRAM GPU + 16GB RAM systems:
- Batch size: 25 pages (for 4GB VRAM)
- Hybrid GPU/CPU mode support
- Automatic fallback to CPU on memory pressure

## Usage

### Basic Usage

```python
from services.ocr_batch_service import OCRBatchService

# Initialize service
service = OCRBatchService(
    progress_callback=my_progress_callback,
    use_gpu=True
)

# Process files
results = service.process_batch(
    files=['/path/to/file1.pdf', '/path/to/file2.pdf'],
    output_dir='/path/to/output'
)

# Cleanup
service.cleanup()
```

### With Progress Callback

```python
def progress_callback(current, total, message, percent, eta):
    print(f"{percent:.1f}% - {message} - ETA: {int(eta)}s")

service = OCRBatchService(
    progress_callback=progress_callback,
    use_gpu=True
)
```

### With Cancellation Support

```python
import threading

cancellation_flag = threading.Event()

service = OCRBatchService(
    progress_callback=progress_callback,
    cancellation_flag=cancellation_flag,
    use_gpu=True
)

# In another thread or signal handler:
cancellation_flag.set()  # Triggers graceful shutdown
```

### Integration with IPC (main.py)

```python
from services.ocr_batch_service import OCRBatchService
import threading

class IPCHandler:
    def __init__(self):
        self.cancellation_flag = threading.Event()
        
    def handle_ocr_batch(self, command):
        files = command.get('files', [])
        output_dir = command.get('output_dir', './output')
        
        def progress_callback(current, total, message, percent, eta):
            self.send_progress(
                current=current,
                total=total,
                message=message,
                percent=percent
            )
        
        service = OCRBatchService(
            progress_callback=progress_callback,
            cancellation_flag=self.cancellation_flag,
            use_gpu=True
        )
        
        try:
            results = service.process_batch(files, output_dir)
            self.send_result(results)
        except Exception as e:
            self.send_error(str(e))
        finally:
            service.cleanup()
    
    def handle_cancel(self):
        self.cancellation_flag.set()
```

## API Reference

### OCRBatchService

#### Constructor

```python
OCRBatchService(
    progress_callback=None,
    cancellation_flag=None,
    use_gpu=True
)
```

**Parameters:**
- `progress_callback`: Callable with signature `(current, total, message, percent, eta)`
  - `current`: Current item index (int)
  - `total`: Total items (int)
  - `message`: Status message (str)
  - `percent`: Completion percentage 0-100 (float)
  - `eta`: Estimated time remaining in seconds (float)
- `cancellation_flag`: threading.Event for cancellation signaling
- `use_gpu`: Whether to use GPU acceleration if available (bool)

#### Methods

##### process_batch(files, output_dir)

Process multiple PDF files sequentially.

**Parameters:**
- `files`: List of PDF file paths (List[str])
- `output_dir`: Directory to save processed files (str)

**Returns:**
Dictionary with:
```python
{
    'successful': [
        {
            'file': '/path/to/file.pdf',
            'pages_processed': 100,
            'pages_ocr': 30,
            'pages_text_layer': 70,
            'output_path': '/output/file.pdf'
        }
    ],
    'failed': [
        {
            'file': '/path/to/failed.pdf',
            'error': 'Error message'
        }
    ],
    'total_pages_processed': 100,
    'statistics': {
        'total_files': 2,
        'successful_files': 1,
        'failed_files': 1,
        'total_pages_processed': 100,
        'total_pages_ocr': 30,
        'total_pages_text_layer': 70,
        'duration_seconds': 45.2,
        'pages_per_second': 2.21
    }
}
```

##### cleanup()

Release OCR resources and cleanup memory. Should always be called when done.

## Configuration

### Batch Sizes by Hardware

The service automatically detects optimal batch sizes:

**GPU Mode:**
- 4GB VRAM: 25 pages/batch (target hardware)
- 6GB VRAM: 35 pages/batch
- 8GB VRAM: 50 pages/batch
- 10GB+ VRAM: 60 pages/batch

**CPU Mode:**
- 8GB RAM: 5 pages/batch
- 16GB RAM: 10 pages/batch
- 24GB RAM: 15 pages/batch
- 32GB+ RAM: 20 pages/batch

### Retry Configuration

Built-in retry logic with exponential backoff:
- **Max retries**: 3 attempts per page
- **Initial delay**: 1 second
- **Backoff factor**: 2x (1s, 2s, 4s)
- **Behavior**: Skip page after 3 failures, continue processing

### Memory Management

Automatic memory optimization:
- **Cleanup interval**: Every 10 pages
- **VRAM monitoring**: Real-time tracking with adaptive batch sizing
- **GPU cache clearing**: After each file completion
- **Garbage collection**: Explicit cleanup between batches

## Performance Characteristics

### Processing Speed

Typical speeds for 300 DPI scanned pages:

**With 4GB VRAM GPU:**
- Pages with text layer: ~0.01s per page (instant extraction)
- Pages needing OCR: ~0.2-0.4s per page
- Mixed document (50% OCR): ~0.1-0.2s per page

**With CPU (16GB RAM):**
- Pages with text layer: ~0.01s per page
- Pages needing OCR: ~0.5-1s per page
- Mixed document (50% OCR): ~0.25-0.5s per page

### Large PDF Estimates

For a 5GB PDF with 5000+ pages:
- 100% scanned (needs OCR): 8-20 minutes (4GB VRAM)
- 50% scanned: 4-10 minutes
- 100% text layer: 1-2 minutes

## Error Handling

### Per-Page Errors

If a page fails after 3 retries:
- Page is skipped with empty text
- Processing continues with remaining pages
- Error is logged to stderr
- Statistics track failed pages

### Per-File Errors

If a file fails to open or process:
- File is marked as failed in results
- Processing continues with remaining files
- Partial results are preserved
- Error details included in failed list

### Cancellation

Graceful cancellation between pages:
- No data corruption
- Partial results preserved
- Resources cleaned up properly
- Returns results for completed files

## Memory Management Best Practices

### For 4GB VRAM Systems

1. **Enable VRAM monitoring**: Automatically enabled for GPU mode
2. **Use batch processing**: Don't process all pages at once
3. **Cleanup regularly**: Service does this automatically every 10 pages
4. **Monitor pressure**: Service adapts batch size on high memory pressure

### For Low RAM Systems

1. **Use CPU mode**: `use_gpu=False` for systems with <8GB RAM
2. **Reduce batch size**: Service automatically adjusts based on available RAM
3. **Close other applications**: Free up system memory before processing

## Logging

All logging goes to stderr (stdout reserved for IPC):

```python
import logging

# In your application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
```

**Log levels:**
- `DEBUG`: Per-page processing details
- `INFO`: File-level progress, statistics
- `WARNING`: Memory pressure, retry attempts
- `ERROR`: Failed operations, exceptions

## Testing

Run the test suite:

```bash
cd python-backend
python test_ocr_batch_service.py
```

Tests validate:
- Service initialization
- Batch size detection
- Cancellation handling
- Retry logic
- Statistics tracking

## Troubleshooting

### High Memory Usage

**Symptom**: VRAM or RAM exhaustion

**Solutions:**
1. Service automatically reduces batch size on high pressure
2. Check logs for "VRAM pressure" warnings
3. Verify GPU memory with `nvidia-smi` (NVIDIA) or Task Manager (Windows)
4. Consider using CPU mode for very large files

### Slow Processing

**Symptom**: Processing slower than expected

**Checks:**
1. Verify GPU is being used: Check logs for "GPU enabled"
2. Check if pages have text layers: Should see "Using text layer" logs
3. Verify batch size: Should be 25 for 4GB VRAM
4. Monitor CPU/GPU usage: Should be near 100% during OCR

### OCR Quality Issues

**Note**: This service focuses on text extraction, not quality. For quality issues:
1. Verify source PDF quality
2. Check DPI (currently 300, configurable in code)
3. Consider using stricter text quality validation

## Future Enhancements

Potential improvements for future versions:

1. **Adaptive DPI**: Adjust DPI based on page complexity
2. **Parallel file processing**: Process multiple files simultaneously
3. **Resume capability**: Resume interrupted batch processing
4. **OCR layer embedding**: Save OCR text back to PDF
5. **Quality metrics**: Track and report OCR confidence scores
6. **Streaming output**: Stream results as they complete

## Related Services

- **OCRService** (`ocr_service.py`): Low-level OCR operations
- **PDFProcessor** (`pdf_processor.py`): PDF manipulation and text extraction
- **VRAMMonitor** (`ocr/vram_monitor.py`): GPU memory monitoring

## License

Part of Document De-Bundler project. See root LICENSE file.
