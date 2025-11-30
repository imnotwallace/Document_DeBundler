"""
Pass 2: PP-DocLayout Layout Analysis.

Loads PP-DocLayout, processes all pages with optimistic batching, saves per-page JSON.
Replaced Florence-2 with PP-DocLayout for better document-specific region detection.
"""

import gc
import logging
import time
from pathlib import Path
from typing import Callable, Optional, List

import numpy as np
from PIL import Image

from ..data_structures import PipelineState
from ..checkpoint_manager import CheckpointManager
from ..model_lifecycle import ModelLifecycleManager
from ...layout_analysis.doclayout_engine import DocLayoutEngine
from ...layout_analysis.region_extractor import RegionExtractor
from ...layout_analysis.gap_filler import GapFiller
from ...layout_analysis.region_subdivider import RegionSubdivider
from ...reading_order.data_structures import WordBox
from ..data_structures import OCRPageResult

logger = logging.getLogger(__name__)


def run_pass_2(
    state: PipelineState,
    checkpoint: CheckpointManager,
    model_lifecycle: ModelLifecycleManager,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    """
    Pass 2: PP-DocLayout layout analysis.

    - Load PP-DocLayout model (much smaller than Florence-2)
    - Process pages with optimistic batching (start batch=4, fallback to 1)
    - Save per-page JSON to temp/layout_results/
    - Checkpoint after each batch
    - Unload model + aggressive cleanup

    Args:
        state: Pipeline state
        checkpoint: Checkpoint manager
        model_lifecycle: Model lifecycle manager
        progress_callback: Optional progress callback(current, total, message)
    """
    logger.info("Pass 2: PP-DocLayout starting...")
    start_time = time.time()

    # Get pages to process
    remaining_pages = checkpoint.get_remaining_pages("pass2_layout", state.total_pages)

    if not remaining_pages:
        logger.info("All pages already processed, skipping pass")
        return

    logger.info(f"Processing {len(remaining_pages)}/{state.total_pages} pages")

    # Mark pass as in progress
    checkpoint.mark_pass_in_progress("pass2_layout")

    # Get configuration
    config = state.config
    device = config.get("device", "gpu")  # PP-DocLayout uses "gpu" not "cuda"
    model_name = config.get("doclayout_model", "PP-DocLayout-M")
    confidence_threshold = config.get("layout_confidence_threshold", 0.5)

    # Get batch size from checkpoint (may have been reduced due to OOM)
    # PP-DocLayout is much smaller (22MB vs 2.5GB), so start with larger batch
    initial_batch_size = checkpoint.get_batch_size("pass2_layout", default=4)
    current_batch_size = initial_batch_size
    min_batch_size = 1

    # Load PP-DocLayout
    engine = model_lifecycle.load_doclayout(
        model_name=model_name,
        device=device,
        confidence_threshold=confidence_threshold,
    )

    region_extractor = RegionExtractor()
    results_dir = state.layout_results_dir

    # Initialize layout enhancement components
    gap_filler = GapFiller()
    region_subdivider = RegionSubdivider()

    # Check if layout enhancement is enabled
    enable_enhancement = config.get("enable_layout_enhancement", True)

    processed_count = 0
    failed_count = 0
    oom_count = 0

    try:
        i = 0
        while i < len(remaining_pages):
            batch_pages = remaining_pages[i:i + current_batch_size]
            batch_image_paths = [state.image_paths[p] for p in batch_pages]

            try:
                # Load images
                batch_images = []
                for img_path in batch_image_paths:
                    img = Image.open(img_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    batch_images.append(np.array(img))

                # Process batch
                batch_results = engine.analyze_batch(
                    batch_images,
                    page_nums=batch_pages,
                    batch_size=current_batch_size,
                )

                # Save results
                for page_num, layout_output in zip(batch_pages, batch_results):
                    # Extract structured layout
                    layout_result = region_extractor.extract(layout_output, page_num)

                    # Apply layout enhancement if enabled
                    if enable_enhancement:
                        layout_result = _enhance_layout(
                            layout_result,
                            page_num,
                            state,
                            gap_filler,
                            region_subdivider,
                        )

                    # Save to JSON
                    output_path = results_dir / f"page_{page_num:04d}.json"
                    layout_result.save_to_file(output_path)

                    # Checkpoint
                    checkpoint.mark_page_completed("pass2_layout", page_num)
                    processed_count += 1

                    # Progress callback
                    if progress_callback:
                        progress_callback(
                            processed_count,
                            len(remaining_pages),
                            f"Page {page_num + 1}: {len(layout_result.regions)} regions",
                        )

                logger.debug(
                    f"Batch {i // current_batch_size + 1}: "
                    f"pages {batch_pages[0] + 1}-{batch_pages[-1] + 1} "
                    f"(batch_size={current_batch_size})"
                )

                # Move to next batch
                i += current_batch_size

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                    # OOM detected - reduce batch size
                    oom_count += 1

                    if current_batch_size > min_batch_size:
                        old_batch_size = current_batch_size
                        current_batch_size = max(min_batch_size, current_batch_size // 2)

                        logger.warning(
                            f"GPU OOM detected (attempt {oom_count}), "
                            f"reducing batch size: {old_batch_size} -> {current_batch_size}"
                        )

                        # Update checkpoint
                        checkpoint.update_batch_size(
                            "pass2_layout",
                            current_batch_size,
                            f"OOM with batch_size={old_batch_size}",
                        )

                        # Clear cache and retry
                        gc.collect()
                        try:
                            import paddle
                            paddle.device.cuda.empty_cache()
                        except Exception:
                            pass

                        # Don't advance i - retry this batch with smaller size
                        continue
                    else:
                        # Already at minimum batch size
                        logger.error(f"OOM at minimum batch size ({min_batch_size})")
                        raise RuntimeError("Insufficient VRAM even with batch_size=1")
                else:
                    # Other error
                    logger.error(f"Batch failed: {e}")
                    for page_num in batch_pages:
                        checkpoint.mark_page_failed("pass2_layout", page_num, str(e))
                    failed_count += len(batch_pages)
                    i += current_batch_size

            except Exception as e:
                logger.error(f"Batch failed with error: {e}")
                for page_num in batch_pages:
                    checkpoint.mark_page_failed("pass2_layout", page_num, str(e))
                failed_count += len(batch_pages)
                i += current_batch_size

    finally:
        # Always unload model
        engine.cleanup()
        model_lifecycle.aggressive_cleanup()

    elapsed = time.time() - start_time
    logger.info(
        f"Pass 2 complete: {processed_count} processed, {failed_count} failed, "
        f"{oom_count} OOM fallbacks in {elapsed:.2f}s "
        f"({elapsed / state.total_pages:.2f}s/page)"
    )


def _enhance_layout(
    layout_result,
    page_num: int,
    state: PipelineState,
    gap_filler: GapFiller,
    region_subdivider: RegionSubdivider,
):
    """
    Enhance layout result with gap filling and region subdivision.

    Args:
        layout_result: FlorenceLayoutResult from region_extractor
        page_num: Page number
        state: Pipeline state (for accessing OCR results)
        gap_filler: GapFiller instance
        region_subdivider: RegionSubdivider instance

    Returns:
        Enhanced FlorenceLayoutResult
    """
    # Load OCR results to get word boxes
    ocr_path = state.ocr_results_dir / f"page_{page_num:04d}.json"

    if not ocr_path.exists():
        logger.warning(f"OCR result not found for page {page_num}, skipping enhancement")
        return layout_result

    try:
        ocr_result = OCRPageResult.load_from_file(ocr_path)

        # Convert word_boxes dicts to WordBox objects
        word_boxes = []
        for wb in ocr_result.word_boxes:
            bbox = wb.get("bbox", {})
            word = WordBox(
                text=wb.get("text", ""),
                x0=bbox.get("x0", 0),
                y0=bbox.get("y0", 0),
                x1=bbox.get("x1", 0),
                y1=bbox.get("y1", 0),
                page=page_num,
                confidence=wb.get("confidence", 1.0),
            )
            word_boxes.append(word)

        if not word_boxes:
            return layout_result

        # Apply gap filling to create regions for orphan words
        enhanced_regions = gap_filler.fill_gaps(
            regions=layout_result.regions,
            word_boxes=word_boxes,
            page_width=layout_result.image_width,
            page_height=layout_result.image_height,
        )

        # Apply region subdivision to split wide regions into columns
        enhanced_regions = region_subdivider.subdivide_regions(
            regions=enhanced_regions,
            word_boxes=word_boxes,
            page_width=layout_result.image_width,
            page_height=layout_result.image_height,
        )

        # Update layout result with enhanced regions
        layout_result.regions = enhanced_regions

        logger.debug(
            f"Page {page_num + 1}: Layout enhanced "
            f"({len(ocr_result.word_boxes)} words, {len(enhanced_regions)} regions)"
        )

    except Exception as e:
        logger.warning(f"Layout enhancement failed for page {page_num}: {e}")

    return layout_result
