"""
Pass 4: PDF Creation.

Creates per-page searchable PDFs with invisible text overlay.
CPU-only pass with parallel processing support.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List

from ..data_structures import PipelineState
from ..checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


def run_pass_4(
    state: PipelineState,
    checkpoint: CheckpointManager,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    """
    Pass 4: PDF page creation.

    - Load reading order JSON for each page
    - Load original image
    - Create PDF with invisible text overlay
    - Save to temp/pdf_pages/
    - Parallel processing with ThreadPoolExecutor

    Args:
        state: Pipeline state
        checkpoint: Checkpoint manager
        progress_callback: Optional progress callback(current, total, message)
    """
    logger.info("Pass 4: PDF Creation starting...")
    start_time = time.time()

    # Get pages to process
    remaining_pages = checkpoint.get_remaining_pages("pass4_pdf_creation", state.total_pages)

    if not remaining_pages:
        logger.info("All pages already processed, skipping pass")
        return

    logger.info(f"Processing {len(remaining_pages)}/{state.total_pages} pages")

    # Mark pass as in progress
    checkpoint.mark_pass_in_progress("pass4_pdf_creation")

    # Get configuration
    config = state.config
    max_workers = config.get("pdf_creation_workers", 4)
    jpeg_quality = config.get("jpeg_quality", 85)

    results_dir = state.pdf_pages_dir
    reading_order_dir = state.reading_order_dir
    images_dir = state.workspace_dir / "temp" / "images"

    processed_count = 0
    failed_count = 0

    # Process pages in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _create_page_pdf,
                page_num,
                reading_order_dir,
                images_dir,
                results_dir,
                jpeg_quality,
            ): page_num
            for page_num in remaining_pages
        }

        for future in as_completed(futures):
            page_num = futures[future]

            try:
                result = future.result()

                # Checkpoint
                checkpoint.mark_page_completed("pass4_pdf_creation", page_num)
                processed_count += 1

                # Progress callback
                if progress_callback:
                    progress_callback(
                        processed_count,
                        len(remaining_pages),
                        f"Page {page_num + 1}: {result['text_items']} text items",
                    )

                logger.debug(
                    f"Page {page_num + 1}: PDF created in {result['processing_time']:.2f}s"
                )

            except Exception as e:
                logger.error(f"Page {page_num + 1} failed: {e}")
                checkpoint.mark_page_failed("pass4_pdf_creation", page_num, str(e))
                failed_count += 1

    elapsed = time.time() - start_time
    logger.info(
        f"Pass 4 complete: {processed_count} processed, {failed_count} failed "
        f"in {elapsed:.2f}s"
    )


def _create_page_pdf(
    page_num: int,
    reading_order_dir: Path,
    images_dir: Path,
    results_dir: Path,
    jpeg_quality: int,
) -> Dict[str, Any]:
    """Create a single page PDF with invisible text overlay."""
    start_time = time.time()

    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF (fitz) required. Install with: pip install PyMuPDF")

    # Load reading order result
    reading_order_path = reading_order_dir / f"page_{page_num:04d}.json"
    if not reading_order_path.exists():
        raise FileNotFoundError(f"Reading order not found: {reading_order_path}")

    with open(reading_order_path, "r", encoding="utf-8") as f:
        reading_order_data = json.load(f)

    # Load image
    image_path = images_dir / f"page_{page_num:04d}.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    from PIL import Image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Create new PDF
    doc = fitz.open()

    # Create page with image dimensions (in points, assuming 72 DPI)
    # Since images are rendered at 300 DPI, scale down for PDF
    scale = 72.0 / 300.0
    page_width = img_width * scale
    page_height = img_height * scale

    page = doc.new_page(width=page_width, height=page_height)

    # Insert image
    page_rect = page.rect
    page.insert_image(page_rect, filename=str(image_path))

    # Insert invisible text - using SEQUENTIAL Y-POSITIONS to force reading order
    # PyMuPDF's get_text() extracts by physical position. For curved documents,
    # original word positions can cause wrong extraction order. Instead, we insert
    # text at sequential Y positions to guarantee correct reading order.
    text_items = 0
    ordered_blocks = reading_order_data.get("ordered_blocks", [])

    # Calculate line insertion parameters
    # Use a small font and tight spacing - text is invisible anyway
    font_size = 8  # Small but reasonable
    line_height = font_size * 1.2  # Typical line height
    
    # Start position - just below top margin
    current_y = 10  # Start near top
    left_margin = 10  # Left margin for text

    for block in sorted(ordered_blocks, key=lambda b: b.get("reading_order", 0)):
        for line in block.get("lines", []):
            words = line.get("words", [])
            if not words:
                continue

            # Build full line text from words
            line_text = " ".join(w.get("text", "") for w in words if w.get("text", "").strip())
            if not line_text.strip():
                continue

            try:
                # Insert full line as invisible text at SEQUENTIAL position
                # This guarantees correct extraction order regardless of original word positions
                page.insert_text(
                    (left_margin, current_y),
                    line_text,
                    fontsize=font_size,
                    fontname="helv",
                    color=(1, 1, 1),  # White (invisible on white)
                    render_mode=3,  # Invisible
                    overlay=True,
                )
                text_items += 1
                current_y += line_height  # Move to next line position
                
                # Wrap to avoid going off page (shouldn't happen in practice)
                if current_y > page_height - 10:
                    current_y = 10
                    
            except Exception as e:
                logger.debug(f"Failed to insert line text '{line_text[:30]}...': {e}")
                continue

    # Save PDF
    output_path = results_dir / f"page_{page_num:04d}.pdf"
    doc.save(str(output_path), garbage=4, deflate=True)
    doc.close()

    processing_time = time.time() - start_time

    return {
        "text_items": text_items,
        "processing_time": processing_time,
    }
