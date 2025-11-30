"""
Reading Order Detection Package

Implements a 5-layer architecture for detecting and assigning reading order
to OCR-detected text, following industry-standard practices.

Layers:
1. Input Normalization - Clean and validate OCR output
2. Structural Grouping - Group words into lines and blocks
3. Region Detection - Detect columns, headers, footers, tables
4. Reading Order Assignment - Assign sequential order to regions
5. Text Output - Generate ordered text output

Usage:
    from services.ocr.reading_order import process_reading_order

    ordered_text = process_reading_order(
        word_boxes,
        page_width=2480,
        page_height=3508
    )
"""

from .pipeline import (
    process_reading_order,
    process_reading_order_safe,
    process_paddleocr_result,
    process_tesseract_result
)
from .data_structures import WordBox, Line, Block, Region
from .config import ReadingOrderConfig

__all__ = [
    'process_reading_order',
    'process_reading_order_safe',
    'process_paddleocr_result',
    'process_tesseract_result',
    'WordBox',
    'Line',
    'Block',
    'Region',
    'ReadingOrderConfig',
]
