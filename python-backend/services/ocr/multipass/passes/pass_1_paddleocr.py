"""
Pass 1: PaddleOCR Text Extraction.

Loads PaddleOCR, processes all pages, saves per-page JSON, unloads model.
"""

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from ..data_structures import PipelineState, OCRPageResult
from ..checkpoint_manager import CheckpointManager
from ..model_lifecycle import ModelLifecycleManager

logger = logging.getLogger(__name__)


def run_pass_1(
    state: PipelineState,
    checkpoint: CheckpointManager,
    model_lifecycle: ModelLifecycleManager,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    """
    Pass 1: PaddleOCR text extraction.

    - Load PaddleOCR model
    - Process all pages (skip completed from checkpoint)
    - Save per-page JSON to temp/ocr_results/
    - Checkpoint after each page
    - Unload model + aggressive cleanup

    Args:
        state: Pipeline state
        checkpoint: Checkpoint manager
        model_lifecycle: Model lifecycle manager
        progress_callback: Optional progress callback(current, total, message)
    """
    logger.info("Pass 1: PaddleOCR starting...")
    start_time = time.time()

    # Get pages to process
    remaining_pages = checkpoint.get_remaining_pages("pass1_paddleocr", state.total_pages)

    if not remaining_pages:
        logger.info("All pages already processed, skipping pass")
        return

    logger.info(f"Processing {len(remaining_pages)}/{state.total_pages} pages")

    # Mark pass as in progress
    checkpoint.mark_pass_in_progress("pass1_paddleocr")

    # Load PaddleOCR
    config = state.config
    use_gpu = config.get("paddle_use_gpu", True)
    lang = config.get("paddle_lang", "en")

    engine = model_lifecycle.load_paddleocr(use_gpu=use_gpu, lang=lang)

    results_dir = state.ocr_results_dir
    processed_count = 0
    failed_count = 0

    try:
        for page_num in remaining_pages:
            image_path = state.image_paths[page_num]

            try:
                # Process page
                result = _process_page(engine, image_path, page_num)

                # Save to JSON
                output_path = results_dir / f"page_{page_num:04d}.json"
                result.save_to_file(output_path)

                # Checkpoint
                checkpoint.mark_page_completed("pass1_paddleocr", page_num)
                processed_count += 1

                # Progress callback
                if progress_callback:
                    progress_callback(
                        processed_count,
                        len(remaining_pages),
                        f"Page {page_num + 1}: {len(result.word_boxes)} words",
                    )

                logger.debug(
                    f"Page {page_num + 1}/{state.total_pages}: "
                    f"{len(result.word_boxes)} words in {result.processing_time:.2f}s"
                )

            except Exception as e:
                logger.error(f"Page {page_num + 1} failed: {e}")
                checkpoint.mark_page_failed("pass1_paddleocr", page_num, str(e))
                failed_count += 1

                # Continue with next page
                continue

    finally:
        # Always unload model
        model_lifecycle.unload_model(engine, "PaddleOCR")

    elapsed = time.time() - start_time
    logger.info(
        f"Pass 1 complete: {processed_count} processed, {failed_count} failed "
        f"in {elapsed:.2f}s ({elapsed / state.total_pages:.2f}s/page)"
    )


def _process_page(engine, image_path: Path, page_num: int) -> OCRPageResult:
    """Process a single page with PaddleOCR."""
    start_time = time.time()

    # Load image
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    image_array = np.array(image)

    # Run OCR
    ocr_result = engine.process_image(image_array)

    # Convert to our format
    word_boxes = []

    if ocr_result and ocr_result.text:
        # Parse from OCRResult
        # PaddleOCR returns text at line level, but we have bounding boxes
        if ocr_result.bbox:
            # Each bbox corresponds to a text line
            raw = ocr_result.raw_result
            # raw_result is a list with a single dict-like PaddleX OCRResult
            if raw and isinstance(raw, list) and len(raw) > 0:
                raw_item = raw[0]
                texts = raw_item.get("rec_texts", []) if hasattr(raw_item, 'get') else []
                scores = raw_item.get("rec_scores", []) if hasattr(raw_item, 'get') else []
                polys = raw_item.get("rec_polys", []) if hasattr(raw_item, 'get') else []

                for i, (text, score, poly) in enumerate(zip(texts, scores, polys)):
                    if not text.strip():
                        continue

                    # Convert polygon to bbox
                    if len(poly) == 4:
                        x_coords = [p[0] for p in poly]
                        y_coords = [p[1] for p in poly]
                        bbox = {
                            "x0": int(min(x_coords)),
                            "y0": int(min(y_coords)),
                            "x1": int(max(x_coords)),
                            "y1": int(max(y_coords)),
                        }
                    else:
                        continue

                    word_boxes.append({
                        "text": text,
                        "bbox": bbox,
                        "confidence": float(score),
                    })

    processing_time = time.time() - start_time

    return OCRPageResult(
        page_number=page_num,
        image_width=width,
        image_height=height,
        word_boxes=word_boxes,
        processing_time=processing_time,
        gpu_used=engine.supports_gpu(),
    )
