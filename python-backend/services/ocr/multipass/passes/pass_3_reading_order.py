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
from ...reading_order.pipeline import process_reading_order
from ...reading_order.config import ReadingOrderConfig
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
    fusion,  # SimpleFusion or RegionFirstFusion (fallback)
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

    # Convert OCR word boxes to format expected by process_reading_order
    ocr_results_for_pipeline = []
    all_xs = []
    all_ys = []
    for wb in ocr_result.word_boxes:
        bbox = wb["bbox"]
        x0, y0, x1, y1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
        all_xs.extend([x0, x1])
        all_ys.extend([y0, y1])

        # CRITICAL: Use original polygon if available (preserves curve information)
        # Otherwise fall back to reconstructing from bbox
        if "original_poly" in wb:
            poly = wb["original_poly"]
        else:
            # Convert bbox to polygon format [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
            poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

        ocr_item = {
            'text': wb["text"],
            'bbox': poly,
            'confidence': wb.get("confidence", 1.0),
            'page': page_num,
        }

        # Pass pre-calculated center_y if available (more accurate for curved text)
        if "center_y" in wb:
            ocr_item['center_y'] = wb["center_y"]

        ocr_results_for_pipeline.append(ocr_item)

    # Calculate page dimensions
    page_width = max(all_xs) * 1.1 if all_xs else 2550
    page_height = max(all_ys) * 1.1 if all_ys else 3300

    # Use our reading order pipeline with AUTO-DETECTION
    # This handles book spine curvature and skewed documents properly
    reading_order_config = ReadingOrderConfig(
        auto_detect_skew=True,
        skew_detection_threshold=0.5,  # degrees - triggers clustered formation
        line_baseline_tolerance_multiplier=0.6,
        line_y_spread_multiplier=1.5,
        # Let auto-detection choose the best method (clustered for curves/skew)
        use_clustered_line_formation=False,
        use_sequential_line_formation=False,
    )

    # Process reading order - returns structured blocks
    structured_result = process_reading_order(
        ocr_results=ocr_results_for_pipeline,
        page_width=page_width,
        page_height=page_height,
        config=reading_order_config,
        return_structured=True
    )

    # Convert structured result to blocks_data format
    blocks_data = []
    for block_idx, block in enumerate(structured_result):
        block_dict = {
            "block_id": block_idx,
            "block_type": block.get("type", "text"),
            "reading_order": block_idx,
            "bbox": block.get("bbox", {"x0": 0, "y0": 0, "x1": 0, "y1": 0}),
            "lines": [],
        }

        # Handle bbox format - may be [x0,y0,x1,y1] or dict
        bbox = block.get("bbox", [0, 0, 0, 0])
        if isinstance(bbox, list) and len(bbox) == 4:
            block_dict["bbox"] = {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]}

        for line_idx, line in enumerate(block.get("lines", [])):
            words = line.get("words", [])
            line_text = " ".join(w.get("text", "") for w in words)

            # Calculate line bbox from words
            if words:
                word_bboxes = [w.get("bbox", [0, 0, 0, 0]) for w in words]
                line_x0 = min(b[0] if isinstance(b, list) else b.get("x0", 0) for b in word_bboxes)
                line_y0 = min(b[1] if isinstance(b, list) else b.get("y0", 0) for b in word_bboxes)
                line_x1 = max(b[2] if isinstance(b, list) else b.get("x1", 0) for b in word_bboxes)
                line_y1 = max(b[3] if isinstance(b, list) else b.get("y1", 0) for b in word_bboxes)
            else:
                line_x0, line_y0, line_x1, line_y1 = 0, 0, 0, 0

            line_dict = {
                "line_id": line_idx,
                "text": line_text,
                "bbox": {"x0": line_x0, "y0": line_y0, "x1": line_x1, "y1": line_y1},
                "words": [],
            }

            for w in words:
                word_bbox = w.get("bbox", [0, 0, 0, 0])
                if isinstance(word_bbox, list) and len(word_bbox) == 4:
                    word_bbox_dict = {"x0": word_bbox[0], "y0": word_bbox[1], "x1": word_bbox[2], "y1": word_bbox[3]}
                else:
                    word_bbox_dict = word_bbox
                line_dict["words"].append({
                    "text": w.get("text", ""),
                    "bbox": word_bbox_dict,
                    "confidence": w.get("confidence", 1.0),
                })

            block_dict["lines"].append(line_dict)

        blocks_data.append(block_dict)

    # Generate ordered text
    ordered_text_parts = []
    for block in blocks_data:
        block_text_parts = []
        for line in block.get("lines", []):
            block_text_parts.append(line.get("text", ""))
        ordered_text_parts.append("\n".join(block_text_parts))
    ordered_text = "\n\n".join(ordered_text_parts)

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
            "fusion_method": "reading_order_pipeline_auto",
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
