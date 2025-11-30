"""
Pass 5: PDF Merging.

Merges all per-page PDFs into final output document.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from ..data_structures import PipelineState
from ..checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


def run_pass_5(
    state: PipelineState,
    checkpoint: CheckpointManager,
) -> None:
    """
    Pass 5: PDF merging.

    - Collect all page PDFs from temp/pdf_pages/
    - Merge in order
    - Apply compression
    - Save to output path

    Args:
        state: Pipeline state
        checkpoint: Checkpoint manager
    """
    logger.info("Pass 5: PDF Merge starting...")
    start_time = time.time()

    # Check if already completed
    if checkpoint.is_pass_completed("pass5_pdf_merge"):
        logger.info("PDF merge already complete, skipping")
        return

    # Mark as in progress
    checkpoint.mark_pass_in_progress("pass5_pdf_merge")

    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF (fitz) required. Install with: pip install PyMuPDF")

    pdf_pages_dir = state.pdf_pages_dir
    output_path = state.output_path

    # Collect page PDFs in order
    page_pdfs = []
    for page_num in range(state.total_pages):
        page_pdf = pdf_pages_dir / f"page_{page_num:04d}.pdf"
        if page_pdf.exists():
            page_pdfs.append(page_pdf)
        else:
            logger.warning(f"Page PDF missing: {page_pdf}")

    if not page_pdfs:
        raise ValueError("No page PDFs found to merge")

    logger.info(f"Merging {len(page_pdfs)} page PDFs...")

    # Create output document
    output_doc = fitz.open()

    # Insert pages
    for i, page_pdf in enumerate(page_pdfs):
        try:
            src_doc = fitz.open(str(page_pdf))
            output_doc.insert_pdf(src_doc)
            src_doc.close()

            if (i + 1) % 50 == 0:
                logger.info(f"  Merged {i + 1}/{len(page_pdfs)} pages")

        except Exception as e:
            logger.error(f"Failed to merge {page_pdf}: {e}")
            continue

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with compression
    config = state.config
    pdf_compression = config.get("pdf_compression", True)

    if pdf_compression:
        output_doc.save(
            str(output_path),
            garbage=4,  # Maximum garbage collection
            deflate=True,  # Compress streams
            clean=True,  # Clean and sanitize content
        )
    else:
        output_doc.save(str(output_path))

    output_doc.close()

    # Get output size
    output_size_mb = output_path.stat().st_size / (1024 * 1024)

    elapsed = time.time() - start_time
    logger.info(
        f"Pass 5 complete: {len(page_pdfs)} pages merged to {output_path} "
        f"({output_size_mb:.1f}MB) in {elapsed:.2f}s"
    )
