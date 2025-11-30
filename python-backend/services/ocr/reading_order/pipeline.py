"""
Reading Order Detection Pipeline

Main entry point that executes all 5 layers in sequence to produce
properly ordered text from OCR results.

Pipeline Flow:
    Layer 1: Input Normalization
    Layer 2: Structural Grouping (Lines + Blocks)
    Layer 3: Region Detection (Layout + Columns + Zones)
    Layer 4: Reading Order Assignment
    Layer 5: Text Output Generation
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from .data_structures import WordBox, Line, Block, Region
from .config import ReadingOrderConfig
from .normalization import normalize_input, filter_noise, group_by_page
from .line_formation import form_lines, form_lines_clustered, form_lines_sequential, post_process_lines, get_average_word_height
from .block_formation import form_blocks
from .layout_detection import detect_layout_type
from .column_detection import detect_columns_by_projection
from .zone_classification import classify_special_zones
from .reading_order import assign_reading_order
from .text_output import generate_ordered_text, generate_ordered_text_with_coordinates

logger = logging.getLogger(__name__)


def process_reading_order(ocr_results: List[Dict[str, Any]],
                          page_width: float,
                          page_height: float,
                          config: Optional[ReadingOrderConfig] = None,
                          return_structured: bool = False,
                          florence_hints: Optional[Any] = None) -> str | List[Dict[str, Any]]:
    """
    Main pipeline: Process OCR results and return properly ordered text.

    Executes all 5 layers of reading order detection:
    1. Input Normalization: Validate and filter word boxes
    2. Structural Grouping: Form lines and blocks
    3. Region Detection: Detect layout, columns, and special zones
    4. Reading Order Assignment: Assign sequential order based on layout
    5. Text Output: Generate final ordered text

    Args:
        ocr_results: List of OCR word results, each containing:
            - 'text': str (word text)
            - 'bbox': list (bounding box coordinates)
            - 'confidence': float (optional, default 1.0)
            - 'page': int (optional, default 0)
        page_width: Width of the page in pixels
        page_height: Height of the page in pixels
        config: ReadingOrderConfig (uses default if None)
        return_structured: If True, returns structured data with coordinates
                          If False, returns plain text string
        florence_hints: Optional FlorenceLayoutResult from Florence-2 analysis.
                        If provided, uses Florence regions for enhanced layout detection.

    Returns:
        Ordered text string OR structured data (depending on return_structured)

    Raises:
        ValueError: If input is invalid or processing fails
    """
    if config is None:
        config = ReadingOrderConfig()

    # Check if Florence hints are provided for enhanced layout detection
    if florence_hints is not None:
        logger.info("Using Florence-2 hints for enhanced layout detection")
        return _process_with_florence_hints(
            ocr_results, florence_hints, page_width, page_height, config, return_structured
        )

    logger.info(f"Starting reading order pipeline for {len(ocr_results)} words")
    logger.info(f"Page dimensions: {page_width}x{page_height}px")

    # ==================================================================
    # LAYER 1: INPUT NORMALIZATION
    # ==================================================================

    logger.info("Layer 1: Normalizing input...")

    # Convert OCR results to WordBox objects
    word_boxes = []
    for ocr_item in ocr_results:
        try:
            word_box = WordBox.from_ocr_result(
                text=ocr_item['text'],
                bbox=ocr_item['bbox'],
                page=ocr_item.get('page', 0),
                confidence=ocr_item.get('confidence', 1.0)
            )
            word_boxes.append(word_box)
        except Exception as e:
            logger.warning(f"Failed to convert OCR result to WordBox: {e}")
            continue

    if not word_boxes:
        logger.warning("No valid word boxes created from OCR results")
        return "" if not return_structured else []

    # Normalize and filter
    word_boxes = normalize_input(word_boxes, page_width, page_height, config)

    if not word_boxes:
        logger.warning("No words remaining after normalization")
        return "" if not return_structured else []

    logger.info(f"Layer 1 complete: {len(word_boxes)} words after normalization")

    # ==================================================================
    # LAYER 2: STRUCTURAL GROUPING
    # ==================================================================

    logger.info("Layer 2: Forming lines and blocks...")

    # Group by page (if multi-page input)
    pages_dict = group_by_page(word_boxes)

    if len(pages_dict) > 1:
        logger.info(f"Multi-page input detected: {len(pages_dict)} pages")
        # For multi-page, process each page separately and combine
        return _process_multi_page(pages_dict, page_width, page_height, config, return_structured)

    # Single page processing
    page_num = list(pages_dict.keys())[0]
    page_words = pages_dict[page_num]

    # Form lines from words
    if config.use_sequential_line_formation:
        lines = form_lines_sequential(page_words, page_height, config)
    elif config.use_clustered_line_formation:
        lines = form_lines_clustered(page_words, page_height, config)
    else:
        lines = form_lines(page_words, page_height, config)

    # Post-process to fix common issues
    if config.enable_post_processing and lines:
        avg_height = get_average_word_height(page_words)
        lines = post_process_lines(lines, avg_height)

    if not lines:
        logger.warning("No lines formed from words")
        return "" if not return_structured else []

    logger.info(f"Formed {len(lines)} lines from {len(page_words)} words")
    
    # DEBUG: Check line word counts
    if lines:
        word_counts = [len(line.words) for line in lines]
        logger.info(f"DEBUG: Words per line - Min: {min(word_counts)}, Max: {max(word_counts)}, Avg: {sum(word_counts)/len(word_counts):.1f}")
        logger.info(f"DEBUG: First 5 lines: {[line.get_text()[:50] for line in lines[:5]]}")

    # Form blocks from lines
    blocks = form_blocks(lines, page_width, page_height, config)

    if not blocks:
        logger.warning("No blocks formed from lines")
        return "" if not return_structured else []

    logger.info(f"Layer 2 complete: {len(blocks)} blocks from {len(lines)} lines")

    # ==================================================================
    # LAYER 3: REGION DETECTION
    # ==================================================================

    logger.info("Layer 3: Detecting layout and regions...")

    # 3.1: Detect overall layout type
    layout_type = detect_layout_type(page_words, page_width, page_height, config)
    logger.info(f"Detected layout type: {layout_type}")

    # 3.2: Detect columns using projection histogram
    regions = detect_columns_by_projection(page_words, blocks, page_width, page_height, config)

    if not regions:
        logger.warning("No regions created, creating single default region")
        # Fallback: single region with all blocks
        default_region = Region(
            blocks=blocks,
            x0=0.0,
            x1=page_width,
            region_id=0,
            region_type="default"
        )
        regions = [default_region]

    logger.info(f"Created {len(regions)} regions")

    # 3.3: Classify special zones (headers, footers)
    headers, main_blocks, footers = classify_special_zones(blocks, page_height, config)

    logger.info(f"Special zones: {len(headers)} headers, {len(main_blocks)} main, {len(footers)} footers")

    # Update regions to only contain main content blocks
    for region in regions:
        region.blocks = [b for b in region.blocks if b in main_blocks]

    # Remove empty regions
    regions = [r for r in regions if r.blocks]

    logger.info(f"Layer 3 complete: {len(regions)} content regions")

    # ==================================================================
    # LAYER 4: READING ORDER ASSIGNMENT
    # ==================================================================

    logger.info("Layer 4: Assigning reading order...")

    # Assign reading order based on layout type
    ordered_regions = assign_reading_order(regions, layout_type, page_width, config)

    logger.info(f"Layer 4 complete: Assigned reading order to {len(ordered_regions)} regions")

    # ==================================================================
    # LAYER 5: TEXT OUTPUT GENERATION
    # ==================================================================

    logger.info("Layer 5: Generating output text...")

    if return_structured:
        # Return structured data with coordinates
        output = generate_ordered_text_with_coordinates(ordered_regions, headers, footers)
        logger.info(f"Generated structured output: {len(output)} blocks")
        return output
    else:
        # Return plain text string
        output_text = generate_ordered_text(ordered_regions, headers, footers, config)
        logger.info(f"Generated text output: {len(output_text)} characters")
        return output_text


def process_reading_order_safe(ocr_results: List[Dict[str, Any]],
                               page_width: float,
                               page_height: float,
                               config: Optional[ReadingOrderConfig] = None,
                               return_structured: bool = False) -> str | List[Dict[str, Any]]:
    """
    Safe wrapper for process_reading_order with error handling.

    Falls back to simple lexical ordering if processing fails.

    Args:
        Same as process_reading_order

    Returns:
        Ordered text OR structured data (or fallback on error)
    """
    try:
        return process_reading_order(
            ocr_results,
            page_width,
            page_height,
            config,
            return_structured
        )

    except Exception as e:
        logger.error(f"Reading order pipeline failed: {e}", exc_info=True)
        logger.warning("Falling back to simple lexical ordering")

        return _fallback_lexical_ordering(ocr_results, return_structured)


def _process_multi_page(pages_dict: Dict[int, List[WordBox]],
                       page_width: float,
                       page_height: float,
                       config: ReadingOrderConfig,
                       return_structured: bool) -> str | List[Dict[str, Any]]:
    """
    Process multiple pages and combine results.

    Args:
        pages_dict: Dictionary mapping page_num -> List[WordBox]
        page_width, page_height: Page dimensions
        config: Configuration
        return_structured: Return format flag

    Returns:
        Combined output from all pages
    """
    logger.info(f"Processing {len(pages_dict)} pages separately")

    all_outputs = []

    # Process each page
    for page_num in sorted(pages_dict.keys()):
        page_words = pages_dict[page_num]

        logger.info(f"Processing page {page_num} ({len(page_words)} words)")

        # Convert page words back to OCR result format
        page_ocr_results = [
            {
                'text': w.text,
                'bbox': [w.x0, w.y0, w.x1, w.y1],
                'confidence': w.confidence,
                'page': w.page
            }
            for w in page_words
        ]

        # Process this page
        page_output = process_reading_order(
            page_ocr_results,
            page_width,
            page_height,
            config,
            return_structured
        )

        all_outputs.append(page_output)

    # Combine outputs
    if return_structured:
        # Flatten list of lists
        combined = []
        for page_output in all_outputs:
            combined.extend(page_output)
        return combined
    else:
        # Join text with page separators
        return "\n\n--- PAGE BREAK ---\n\n".join(all_outputs)


def _fallback_lexical_ordering(ocr_results: List[Dict[str, Any]],
                               return_structured: bool) -> str | List[Dict[str, Any]]:
    """
    Fallback: Simple lexical ordering (top-to-bottom, left-to-right).

    Used when main pipeline fails.

    Args:
        ocr_results: Raw OCR results
        return_structured: Return format flag

    Returns:
        Simply ordered text or structured data
    """
    logger.info("Using fallback lexical ordering")

    # Sort by Y position first, then X position
    sorted_results = sorted(
        ocr_results,
        key=lambda r: (
            r['bbox'][1] if isinstance(r['bbox'][0], (int, float)) else r['bbox'][0][1],  # Y
            r['bbox'][0] if isinstance(r['bbox'][0], (int, float)) else r['bbox'][0][0]   # X
        )
    )

    if return_structured:
        # Return basic structured data
        structured = []
        for i, result in enumerate(sorted_results):
            bbox = result['bbox']

            # Normalize bbox format
            if isinstance(bbox[0], (list, tuple)):
                # [[x1,y1], [x2,y2], ...] format
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                normalized_bbox = [min(xs), min(ys), max(xs), max(ys)]
            else:
                # [x0, y0, x1, y1] format
                normalized_bbox = bbox

            structured.append({
                'text': result.get('text', ''),
                'bbox': normalized_bbox,
                'reading_order': i,
                'type': 'fallback',
                'category': 'main',
                'lines': [],
                'confidence': result.get('confidence', 1.0)
            })

        return structured
    else:
        # Return simple text concatenation
        texts = [result.get('text', '') for result in sorted_results]
        return ' '.join(texts)


# ==================================================================
# Convenience Functions
# ==================================================================

def process_paddleocr_result(paddleocr_result: List[List],
                            page_width: float,
                            page_height: float,
                            config: Optional[ReadingOrderConfig] = None) -> str:
    """
    Convenience function for PaddleOCR format results.

    PaddleOCR returns: [[[bbox], (text, confidence)], ...]

    Args:
        paddleocr_result: PaddleOCR result format
        page_width, page_height: Page dimensions
        config: Optional configuration

    Returns:
        Ordered text string
    """
    # Convert PaddleOCR format to standard format
    ocr_results = []

    for item in paddleocr_result:
        bbox = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        text, confidence = item[1]

        ocr_results.append({
            'text': text,
            'bbox': bbox,
            'confidence': confidence,
            'page': 0
        })

    return process_reading_order_safe(ocr_results, page_width, page_height, config)


def process_tesseract_result(tesseract_data: Dict[str, List],
                             page_width: float,
                             page_height: float,
                             config: Optional[ReadingOrderConfig] = None) -> str:
    """
    Convenience function for Tesseract format results.

    Tesseract returns: {'text': [...], 'left': [...], 'top': [...], 'width': [...], 'height': [...], 'conf': [...]}

    Args:
        tesseract_data: Tesseract result dict
        page_width, page_height: Page dimensions
        config: Optional configuration

    Returns:
        Ordered text string
    """
    # Convert Tesseract format to standard format
    ocr_results = []

    n = len(tesseract_data['text'])

    for i in range(n):
        text = tesseract_data['text'][i]

        # Skip empty entries
        if not text or text.strip() == '':
            continue

        left = tesseract_data['left'][i]
        top = tesseract_data['top'][i]
        width = tesseract_data['width'][i]
        height = tesseract_data['height'][i]
        conf = tesseract_data['conf'][i]

        # Convert to [x0, y0, x1, y1] format
        bbox = [left, top, left + width, top + height]

        # Normalize confidence to 0-1 range (Tesseract uses 0-100)
        confidence = conf / 100.0 if conf >= 0 else 0.0

        ocr_results.append({
            'text': text,
            'bbox': bbox,
            'confidence': confidence,
            'page': 0
        })

    return process_reading_order_safe(ocr_results, page_width, page_height, config)


def _process_with_florence_hints(
    ocr_results: List[Dict[str, Any]],
    florence_hints: Any,
    page_width: float,
    page_height: float,
    config: ReadingOrderConfig,
    return_structured: bool,
) -> str | List[Dict[str, Any]]:
    """
    Process reading order using Florence-2 layout hints.

    Uses Florence-2 detected regions and layout type to enhance
    the reading order detection, especially for multi-column layouts.

    Args:
        ocr_results: OCR word results
        florence_hints: FlorenceLayoutResult from Florence-2
        page_width, page_height: Page dimensions
        config: Reading order configuration
        return_structured: Whether to return structured data

    Returns:
        Ordered text or structured data
    """
    from ..layout_analysis.fusion_strategies import SimpleFusion
    from .data_structures import WordBox

    logger.info("Processing with Florence-2 hints...")

    # Convert OCR results to WordBox objects
    word_boxes = []
    for ocr_item in ocr_results:
        try:
            word_box = WordBox.from_ocr_result(
                text=ocr_item['text'],
                bbox=ocr_item['bbox'],
                page=ocr_item.get('page', 0),
                confidence=ocr_item.get('confidence', 1.0)
            )
            word_boxes.append(word_box)
        except Exception as e:
            logger.warning(f"Failed to convert OCR result: {e}")
            continue

    if not word_boxes:
        logger.warning("No valid word boxes from OCR results")
        return "" if not return_structured else []

    # Use SimpleFusion to combine OCR with Florence hints
    fusion = SimpleFusion(
        min_region_overlap=0.5,
        line_vertical_threshold=config.vertical_threshold if hasattr(config, 'vertical_threshold') else 0.5,
        block_vertical_gap=config.block_gap_multiplier if hasattr(config, 'block_gap_multiplier') else 2.0,
    )

    ordered_blocks = fusion.fuse(word_boxes, florence_hints)

    if not ordered_blocks:
        logger.warning("No blocks from fusion, falling back to standard processing")
        # Fall back to standard processing without florence hints
        return process_reading_order(
            ocr_results, page_width, page_height, config, return_structured, florence_hints=None
        )

    logger.info(f"Florence fusion produced {len(ordered_blocks)} ordered blocks")

    if return_structured:
        # Return structured data
        output = []
        for block in sorted(ordered_blocks, key=lambda b: b.reading_order or 0):
            block_data = {
                "block_id": block.block_id,
                "block_type": block.block_type,
                "reading_order": block.reading_order,
                "text": block.get_text(),
                "bbox": {
                    "x0": block.x0,
                    "y0": block.y0,
                    "x1": block.x1,
                    "y1": block.y1,
                },
                "lines": [
                    {
                        "text": line.get_text(),
                        "bbox": {"x0": line.x0, "y0": line.y0, "x1": line.x1, "y1": line.y1},
                    }
                    for line in block.lines
                ],
            }
            output.append(block_data)
        return output
    else:
        # Return plain text
        text_parts = []
        for block in sorted(ordered_blocks, key=lambda b: b.reading_order or 0):
            text_parts.append(block.get_text())
        return "\n\n".join(text_parts)
