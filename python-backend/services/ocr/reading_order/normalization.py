"""
Layer 1: Input Normalization

Validates, filters, and normalizes OCR output before processing.
"""

import logging
from typing import List, Dict
from .data_structures import WordBox
from .config import ReadingOrderConfig

logger = logging.getLogger(__name__)


def validate_word_box(word_box: WordBox, page_width: float, page_height: float) -> bool:
    """
    Validate a word box.

    Args:
        word_box: WordBox to validate
        page_width, page_height: Page dimensions for sanity checks

    Returns:
        True if valid, False otherwise
    """
    checks = [
        # Text must not be empty
        word_box.text and len(word_box.text.strip()) > 0,

        # Coordinates must be valid
        word_box.x1 > word_box.x0,
        word_box.y1 > word_box.y0,

        # Confidence must be in valid range
        0.0 <= word_box.confidence <= 1.0,

        # Dimensions must be reasonable
        word_box.width < page_width,
        word_box.height < page_height,

        # Coordinates must be within page bounds
        word_box.x0 >= 0,
        word_box.y0 >= 0,
        word_box.x1 <= page_width,
        word_box.y1 <= page_height,
    ]

    is_valid = all(checks)

    if not is_valid:
        # Log which specific check failed
        check_names = ['text_not_empty', 'x1>x0', 'y1>y0', 'confidence_valid',
                       'width<page_width', 'height<page_height',
                       'x0>=0', 'y0>=0', 'x1<=page_width', 'y1<=page_height']
        failed = [name for name, check in zip(check_names, checks) if not check]
        logger.debug(f"Invalid word box rejected: '{word_box.text}' "
                    f"bbox=[{word_box.x0:.1f}, {word_box.y0:.1f}, {word_box.x1:.1f}, {word_box.y1:.1f}] "
                    f"conf={word_box.confidence:.2f}, failed checks: {failed}")

    return is_valid


def filter_noise(word_boxes: List[WordBox],
                page_width: float,
                page_height: float,
                config: ReadingOrderConfig) -> List[WordBox]:
    """
    Filter out noise and invalid word boxes.

    Args:
        word_boxes: List of WordBox instances
        page_width, page_height: Page dimensions
        config: Configuration with thresholds

    Returns:
        Filtered list of WordBox instances
    """
    filtered = []

    for word_box in word_boxes:
        # Validate
        if not validate_word_box(word_box, page_width, page_height):
            continue

        # Filter by confidence
        if word_box.confidence < config.confidence_threshold:
            logger.debug(f"Low confidence word rejected: '{word_box.text}' "
                        f"conf={word_box.confidence:.2f}")
            continue

        # Filter by size (too small - likely noise)
        if (word_box.width < config.min_word_width or
            word_box.height < config.min_word_height):
            logger.debug(f"Too small word rejected: '{word_box.text}' "
                        f"size={word_box.width:.1f}x{word_box.height:.1f}")
            continue

        # Filter by size (too large - likely detection error)
        # BUT: preserve large words at top of page (likely headers)
        max_width = page_width * config.max_word_width_ratio
        is_in_header_zone = word_box.y0 < page_height * config.header_zone_ratio
        if word_box.width > max_width and not is_in_header_zone:
            logger.debug(f"Too large word rejected: '{word_box.text}' "
                        f"width={word_box.width:.1f} (max={max_width:.1f})")
            continue
        elif word_box.width > max_width and is_in_header_zone:
            logger.info(f"Large word preserved as header: '{word_box.text}' "
                       f"width={word_box.width:.1f} at y={word_box.y0:.1f}")

        filtered.append(word_box)

    logger.info(f"Filtered {len(word_boxes)} words -> {len(filtered)} valid words "
               f"({len(word_boxes) - len(filtered)} rejected)")

    return filtered


def normalize_coordinates(word_boxes: List[WordBox]) -> List[WordBox]:
    """
    Normalize coordinates if needed.

    Currently just ensures derived properties are calculated.
    Can be extended for coordinate system conversions.

    Args:
        word_boxes: List of WordBox instances

    Returns:
        Normalized list of WordBox instances
    """
    # In Python with dataclasses, __post_init__ already calculated
    # derived properties. This function is a placeholder for future
    # coordinate system conversions if needed.

    return word_boxes


def group_by_page(word_boxes: List[WordBox]) -> Dict[int, List[WordBox]]:
    """
    Group word boxes by page number.

    Args:
        word_boxes: List of WordBox instances

    Returns:
        Dictionary mapping page number -> list of word boxes
    """
    pages = {}

    for word_box in word_boxes:
        page_num = word_box.page
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(word_box)

    logger.info(f"Grouped words into {len(pages)} pages")

    return pages


def normalize_input(word_boxes: List[WordBox],
                    page_width: float,
                    page_height: float,
                    config: ReadingOrderConfig) -> List[WordBox]:
    """
    Complete Layer 1 normalization pipeline.

    Args:
        word_boxes: Raw OCR output as WordBox list
        page_width, page_height: Page dimensions
        config: Configuration

    Returns:
        Cleaned and normalized WordBox list
    """
    logger.info(f"Layer 1: Normalizing {len(word_boxes)} word boxes")

    # Filter noise
    filtered = filter_noise(word_boxes, page_width, page_height, config)

    if not filtered:
        logger.warning("No valid words after filtering!")
        return []

    # Normalize coordinates
    normalized = normalize_coordinates(filtered)

    logger.info(f"Layer 1 complete: {len(normalized)} words")

    return normalized
