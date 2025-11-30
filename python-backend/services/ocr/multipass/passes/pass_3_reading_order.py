"""
Pass 3: Reading Order Detection.

Loads OCR + Layout results, runs fusion strategy, saves ordered blocks.
CPU-only pass with parallel processing support.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional, Dict, Any

from ..data_structures import PipelineState, FlorenceLayoutResult, OCRPageResult
from ..checkpoint_manager import CheckpointManager
from ...reading_order.data_structures import WordBox
from ...layout_analysis.fusion_strategies import SimpleFusion, RegionFirstFusion

logger = logging.getLogger(__name__)


def run_pass_3(
    state: PipelineState,
    checkpoint: CheckpointManager,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    """
    Pass 3: Reading order detection.

    - Load OCR + Layout JSON for each page
    - Run fusion strategy to combine results
    - Detect and assign reading order
    - Save per-page JSON to temp/reading_order/
    - Parallel processing with ThreadPoolExecutor

    Args:
        state: Pipeline state
        checkpoint: Checkpoint manager
        progress_callback: Optional progress callback(current, total, message)
    """
    logger.info("Pass 3: Reading Order starting...")
    start_time = time.time()

    # Get pages to process
    remaining_pages = checkpoint.get_remaining_pages("pass3_reading_order", state.total_pages)

    if not remaining_pages:
        logger.info("All pages already processed, skipping pass")
        return

    logger.info(f"Processing {len(remaining_pages)}/{state.total_pages} pages")

    # Mark pass as in progress
    checkpoint.mark_pass_in_progress("pass3_reading_order")

    # Get configuration
    config = state.config
    max_workers = config.get("reading_order_workers", 4)
    fusion_strategy = config.get("fusion_strategy", "region_first")
    min_region_overlap = config.get("min_region_overlap", 0.5)

    # Create fusion strategy
    if fusion_strategy == "simple":
        fusion = SimpleFusion(min_region_overlap=min_region_overlap)
    elif fusion_strategy == "region_first":
        fusion = RegionFirstFusion(min_region_overlap=min_region_overlap)
    else:
        # Default to region-first for better multi-column handling
        fusion = RegionFirstFusion(min_region_overlap=min_region_overlap)

    results_dir = state.reading_order_dir
    ocr_dir = state.ocr_results_dir
    layout_dir = state.layout_results_dir

    processed_count = 0
    failed_count = 0

    # Process pages in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_page,
                page_num,
                ocr_dir,
                layout_dir,
                results_dir,
                fusion,
            ): page_num
            for page_num in remaining_pages
        }

        for future in as_completed(futures):
            page_num = futures[future]

            try:
                result = future.result()

                # Checkpoint
                checkpoint.mark_page_completed("pass3_reading_order", page_num)
                processed_count += 1

                # Progress callback
                if progress_callback:
                    progress_callback(
                        processed_count,
                        len(remaining_pages),
                        f"Page {page_num + 1}: {result['block_count']} blocks",
                    )

                logger.debug(
                    f"Page {page_num + 1}: {result['block_count']} blocks "
                    f"in {result['processing_time']:.2f}s"
                )

            except Exception as e:
                logger.error(f"Page {page_num + 1} failed: {e}")
                checkpoint.mark_page_failed("pass3_reading_order", page_num, str(e))
                failed_count += 1

    elapsed = time.time() - start_time
    logger.info(
        f"Pass 3 complete: {processed_count} processed, {failed_count} failed "
        f"in {elapsed:.2f}s"
    )


def _process_page(
    page_num: int,
    ocr_dir: Path,
    layout_dir: Path,
    results_dir: Path,
    fusion,  # SimpleFusion or RegionFirstFusion
) -> Dict[str, Any]:
    """Process a single page for reading order detection."""
    start_time = time.time()

    # Load OCR result
    ocr_path = ocr_dir / f"page_{page_num:04d}.json"
    if not ocr_path.exists():
        raise FileNotFoundError(f"OCR result not found: {ocr_path}")

    ocr_result = OCRPageResult.load_from_file(ocr_path)

    # Load Layout result (from PP-DocLayout)
    layout_path = layout_dir / f"page_{page_num:04d}.json"
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout result not found: {layout_path}")

    layout_result = FlorenceLayoutResult.load_from_file(layout_path)

    # Convert OCR word boxes to WordBox objects
    word_boxes = []
    for wb in ocr_result.word_boxes:
        bbox = wb["bbox"]
        word = WordBox(
            text=wb["text"],
            x0=bbox["x0"],
            y0=bbox["y0"],
            x1=bbox["x1"],
            y1=bbox["y1"],
            page=page_num,
            confidence=wb.get("confidence", 1.0),
        )
        word_boxes.append(word)

    # Run fusion
    ordered_blocks = fusion.fuse(word_boxes, layout_result)

    # Convert blocks to serializable format
    blocks_data = []
    for block in ordered_blocks:
        block_dict = {
            "block_id": block.block_id,
            "block_type": block.block_type,
            "reading_order": block.reading_order,
            "bbox": {
                "x0": block.x0,
                "y0": block.y0,
                "x1": block.x1,
                "y1": block.y1,
            },
            "lines": [],
        }

        for line in block.lines:
            line_dict = {
                "line_id": line.line_id,
                "text": line.get_text(),
                "bbox": {
                    "x0": line.x0,
                    "y0": line.y0,
                    "x1": line.x1,
                    "y1": line.y1,
                },
                "words": [
                    {
                        "text": w.text,
                        "bbox": {"x0": w.x0, "y0": w.y0, "x1": w.x1, "y1": w.y1},
                        "confidence": w.confidence,
                    }
                    for w in line.words
                ],
            }
            block_dict["lines"].append(line_dict)

        blocks_data.append(block_dict)

    # Generate ordered text
    ordered_text = "\n\n".join(
        block.get_text()
        for block in sorted(ordered_blocks, key=lambda b: b.reading_order or 0)
    )

    processing_time = time.time() - start_time

    # Save result
    result_data = {
        "schema_version": "1.0",
        "page_number": page_num,
        "layout_type": layout_result.layout_type.value,
        "ordered_blocks": blocks_data,
        "ordered_text": ordered_text,
        "metadata": {
            "processing_time_seconds": processing_time,
            "fusion_method": fusion.__class__.__name__,
            "block_count": len(blocks_data),
        },
    }

    output_path = results_dir / f"page_{page_num:04d}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    return {
        "block_count": len(blocks_data),
        "processing_time": processing_time,
    }
