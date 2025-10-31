"""
PDF Processing Service
Handles PDF analysis, page extraction, and splitting with OCR support
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import logging
import gc
import numpy as np

from .ocr.text_quality import (
    TextLayerValidator,
    OCRDecisionEngine,
    TextQualityMetrics,
    TextQualityThresholds
)

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF document processing and analysis with OCR support"""

    def __init__(
        self,
        file_path: str,
        quality_thresholds: Optional[TextQualityThresholds] = None,
        use_strict_validation: bool = False
    ):
        self.file_path = Path(file_path)
        self.doc: Optional[fitz.Document] = None

        # Initialize text quality validation
        self.text_validator = TextLayerValidator(quality_thresholds)
        self.ocr_decision_engine = OCRDecisionEngine(self.text_validator)
        self.use_strict_validation = use_strict_validation

        # Set quality threshold based on validation mode
        if use_strict_validation:
            self.ocr_decision_engine.set_preference('quality_threshold', 'strict')

    def open(self):
        """Open the PDF document"""
        try:
            self.doc = fitz.open(self.file_path)
            logger.info(f"Opened PDF: {self.file_path}, Pages: {len(self.doc)}")
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise

    def close(self):
        """Close the PDF document"""
        if self.doc:
            self.doc.close()
            self.doc = None

    def get_page_count(self) -> int:
        """Get total number of pages"""
        if not self.doc:
            raise ValueError("Document not opened")
        return len(self.doc)

    def has_valid_text_layer(
        self,
        page_num: int,
        return_metrics: bool = False
    ) -> Tuple[bool, Optional[TextQualityMetrics]]:
        """
        Check if a page has a valid, usable text layer using comprehensive quality validation.

        Args:
            page_num: Page number (0-indexed)
            return_metrics: If True, return metrics even if validation fails

        Returns:
            Tuple of (is_valid, metrics) if return_metrics=True
            Tuple of (is_valid, None) if return_metrics=False
        """
        if not self.doc:
            raise ValueError("Document not opened")

        page = self.doc[page_num]
        is_valid, metrics = self.text_validator.has_valid_text_layer(
            page,
            require_high_confidence=self.use_strict_validation
        )

        if is_valid:
            logger.debug(
                f"Page {page_num + 1}: Valid text layer "
                f"(confidence: {metrics.confidence_score:.2%}, "
                f"chars: {metrics.total_chars}, "
                f"words: {metrics.word_count})"
            )
        else:
            logger.debug(
                f"Page {page_num + 1}: Invalid text layer "
                f"(confidence: {metrics.confidence_score:.2%}, "
                f"chars: {metrics.total_chars})"
            )

        return (is_valid, metrics) if return_metrics else (is_valid, None)

    def extract_text(self, page_num: int) -> str:
        """
        Extract text from a page (from text layer).

        Args:
            page_num: Page number (0-indexed)

        Returns:
            Extracted text
        """
        if not self.doc:
            raise ValueError("Document not opened")

        page = self.doc[page_num]
        return page.get_text()

    def render_page_to_image(self, page_num: int, dpi: int = 300) -> np.ndarray:
        """
        Render a PDF page to an image as numpy array.

        Args:
            page_num: Page number (0-indexed)
            dpi: Resolution in DPI (higher = better quality but more memory)

        Returns:
            Image as numpy array (RGB)
        """
        if not self.doc:
            raise ValueError("Document not opened")

        page = self.doc[page_num]

        # Render page to image
        pix = page.get_pixmap(dpi=dpi)

        # Convert to numpy array
        image_array = np.frombuffer(pix.samples, dtype=np.uint8)
        image_array = image_array.reshape(pix.height, pix.width, pix.n)

        # If RGBA, convert to RGB
        if pix.n == 4:
            image_array = image_array[:, :, :3]

        return image_array

    def analyze_structure(self, ocr_service=None) -> Dict[str, Any]:
        """
        Analyze PDF structure to recommend splits using robust text quality validation.

        Args:
            ocr_service: Optional OCR service for scanned pages

        Returns:
            Analysis results with recommendations including quality metrics
        """
        if not self.doc:
            raise ValueError("Document not opened")

        total_pages = len(self.doc)

        # Sample up to 10 pages distributed throughout the document
        sample_size = min(10, total_pages)
        sample_indices = [
            int(i * total_pages / sample_size) for i in range(sample_size)
        ]

        # Check text quality for sample pages
        valid_pages = 0
        total_confidence = 0.0
        quality_metrics_samples = []

        for page_idx in sample_indices:
            is_valid, metrics = self.has_valid_text_layer(page_idx, return_metrics=True)
            if is_valid:
                valid_pages += 1
            if metrics:
                total_confidence += metrics.confidence_score
                quality_metrics_samples.append({
                    'page': page_idx + 1,
                    'confidence': metrics.confidence_score,
                    'chars': metrics.total_chars,
                    'words': metrics.word_count
                })

        # Calculate average confidence across samples
        avg_confidence = total_confidence / sample_size if sample_size > 0 else 0.0

        # Determine if OCR is needed (majority of samples have invalid text)
        needs_ocr = valid_pages < (sample_size / 2)

        logger.info(
            f"Document analysis: {total_pages} pages, "
            f"{valid_pages}/{sample_size} sample pages have valid text, "
            f"avg confidence: {avg_confidence:.2%}, "
            f"needs OCR: {needs_ocr}"
        )

        # TODO: Implement intelligent split detection
        # - Look for title pages
        # - Detect content changes
        # - Use OCR if needed

        return {
            "total_pages": total_pages,
            "needs_ocr": needs_ocr,
            "has_valid_text_layer": valid_pages > 0,
            "valid_text_pages": valid_pages,
            "sampled_pages": sample_size,
            "avg_confidence": avg_confidence,
            "quality_samples": quality_metrics_samples,
            "recommended_splits": [],
        }

    def process_pages_with_ocr(
        self,
        ocr_service,
        start_page: int = 0,
        end_page: Optional[int] = None,
        batch_size: int = 10,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        force_ocr: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process PDF pages with intelligent OCR decision-making in batches.

        Uses OCRDecisionEngine to intelligently decide whether to use existing text layer
        or perform OCR based on comprehensive quality validation.

        Args:
            ocr_service: OCR service instance
            start_page: Starting page number (0-indexed)
            end_page: Ending page number (exclusive), or None for all pages
            batch_size: Number of pages to process at once
            progress_callback: Optional callback(current, total, message)
            force_ocr: Force OCR for all pages regardless of text quality

        Returns:
            List of dicts containing 'text', 'method', 'page_num', and optional 'quality_metrics'
        """
        if not self.doc:
            raise ValueError("Document not opened")

        if end_page is None:
            end_page = len(self.doc)

        total_pages = end_page - start_page
        results = []
        stats = {'text_layer': 0, 'ocr': 0}

        logger.info(f"Processing {total_pages} pages (batch size: {batch_size}, force_ocr: {force_ocr})")

        for batch_start in range(start_page, end_page, batch_size):
            batch_end = min(batch_start + batch_size, end_page)

            if progress_callback:
                progress_callback(
                    batch_start - start_page,
                    total_pages,
                    f"Processing pages {batch_start + 1}-{batch_end}"
                )

            # Collect pages that need OCR in this batch
            pages_needing_ocr = []

            for page_num in range(batch_start, batch_end):
                page = self.doc[page_num]

                # Use OCRDecisionEngine to make intelligent decision
                should_ocr, reason, metrics = self.ocr_decision_engine.should_perform_ocr(
                    page,
                    user_override=force_ocr if force_ocr else None
                )

                if should_ocr:
                    # Queue page for OCR processing
                    pages_needing_ocr.append(page_num)
                    logger.debug(f"Page {page_num + 1}: {reason}")
                else:
                    # Use existing text layer
                    text = self.extract_text(page_num)
                    results.append({
                        'page_num': page_num,
                        'text': text,
                        'method': 'text_layer',
                        'quality_metrics': metrics,
                        'reason': reason
                    })
                    stats['text_layer'] += 1
                    logger.debug(f"Page {page_num + 1}: {reason}")

            # Batch process pages that need OCR
            if pages_needing_ocr:
                logger.debug(f"OCR batch: processing {len(pages_needing_ocr)} pages")

                # Render all pages in this OCR batch to images
                images = []
                for page_num in pages_needing_ocr:
                    image = self.render_page_to_image(page_num)
                    images.append(image)

                # Process all images with OCR
                texts = ocr_service.process_batch(images)

                # Validate batch results match input count
                if len(texts) != len(images):
                    logger.error(
                        f"OCR batch result mismatch: expected {len(images)} results, "
                        f"got {len(texts)}. Some pages may have failed silently."
                    )
                    # Pad with empty strings if fewer results returned
                    while len(texts) < len(images):
                        texts.append("")

                # Store results with metadata
                for page_num, text in zip(pages_needing_ocr, texts):
                    results.append({
                        'page_num': page_num,
                        'text': text,
                        'method': 'ocr',
                        'quality_metrics': None,
                        'reason': 'OCR performed'
                    })
                    stats['ocr'] += 1

                # Clean up memory
                del images
                gc.collect()

        # Sort results by page number (since we may have processed out of order)
        results.sort(key=lambda x: x['page_num'])

        if progress_callback:
            progress_callback(total_pages, total_pages, "Processing complete")

        logger.info(
            f"Completed processing {total_pages} pages: "
            f"{stats['text_layer']} from text layer, "
            f"{stats['ocr']} from OCR"
        )

        return results

    def extract_page_range(self, start: int, end: int, output_path: Path):
        """
        Extract a range of pages to a new PDF.

        Args:
            start: Starting page number (0-indexed, inclusive)
            end: Ending page number (0-indexed, inclusive)
            output_path: Path to save the extracted PDF
        """
        if not self.doc:
            raise ValueError("Document not opened")

        new_doc = fitz.open()
        new_doc.insert_pdf(self.doc, from_page=start, to_page=end)
        new_doc.save(output_path)
        new_doc.close()

        logger.info(f"Extracted pages {start + 1}-{end + 1} to {output_path}")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
