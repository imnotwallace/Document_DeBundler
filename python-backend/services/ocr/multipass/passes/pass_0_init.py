"""
Pass 0: Initialization.

Validates inputs, counts pages, renders to images, creates workspace.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from ..data_structures import PipelineState, PassStatus
from ..pass_config import MultiPassConfig
from ..checkpoint_manager import CheckpointManager
from ..model_lifecycle import ModelLifecycleManager

logger = logging.getLogger(__name__)


def run_pass_0(
    pdf_path: Path,
    output_path: Path,
    workspace_dir: Path,
    config: MultiPassConfig,
    checkpoint: CheckpointManager,
    model_lifecycle: ModelLifecycleManager,
) -> PipelineState:
    """
    Pass 0: Initialization.

    - Validate input PDF exists
    - Count pages
    - Check VRAM availability
    - Create workspace directories
    - Render pages to images
    - Detect resume point from checkpoint

    Args:
        pdf_path: Input PDF path
        output_path: Output PDF path
        workspace_dir: Working directory
        config: Pipeline configuration
        checkpoint: Checkpoint manager
        model_lifecycle: Model lifecycle manager

    Returns:
        Initialized PipelineState
    """
    logger.info("Pass 0: Initialization starting...")
    start_time = time.time()

    # Validate input
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    # Create state
    state = PipelineState(
        doc_id=checkpoint.doc_id,
        workspace_dir=workspace_dir,
        pdf_path=pdf_path,
        output_path=output_path,
        total_pages=0,
        config=config.to_dict(),
    )

    # Create workspace directories
    for subdir in ["ocr_results", "layout_results", "reading_order", "pdf_pages", "images"]:
        (workspace_dir / "temp" / subdir).mkdir(parents=True, exist_ok=True)

    (workspace_dir / "output").mkdir(parents=True, exist_ok=True)

    # Count pages and render to images
    logger.info("Counting pages and rendering images...")
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF (fitz) required. Install with: pip install PyMuPDF")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    state.total_pages = total_pages
    checkpoint.set_total_pages(total_pages)

    logger.info(f"Document has {total_pages} pages")

    # Check VRAM
    vram_info = model_lifecycle.get_vram_info()
    if vram_info.get("available"):
        logger.info(
            f"GPU VRAM: {vram_info.get('total_gb', 0):.1f}GB total, "
            f"{vram_info.get('free_gb', 0):.1f}GB free"
        )
    else:
        logger.warning("No GPU detected, will use CPU mode")

    # Render pages to images
    images_dir = workspace_dir / "temp" / "images"
    image_paths = []

    # Use 300 DPI for rendering (standard OCR resolution)
    dpi = 300
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(total_pages):
        image_path = images_dir / f"page_{page_num:04d}.png"

        # Skip if already rendered (resume support)
        if image_path.exists():
            logger.debug(f"Image already exists: {image_path}")
            image_paths.append(image_path)
            continue

        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pix.save(str(image_path))
        image_paths.append(image_path)

        if (page_num + 1) % 10 == 0:
            logger.info(f"  Rendered {page_num + 1}/{total_pages} pages")

    doc.close()
    state.image_paths = image_paths

    # Check resume point
    resume_point = checkpoint.get_resume_point()
    if resume_point != (0, 0) and resume_point[0] > 0:
        logger.info(
            f"Resuming from checkpoint: pass {resume_point[0]}, page {resume_point[1]}"
        )

    elapsed = time.time() - start_time
    logger.info(f"Pass 0 complete in {elapsed:.2f}s")

    return state
