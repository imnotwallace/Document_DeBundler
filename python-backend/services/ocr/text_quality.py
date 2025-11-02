"""
Text Layer Quality Detection
Validates PDF text layers to detect corrupt/garbage text that should be re-OCR'd
"""

import re
import unicodedata
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Set

logger = logging.getLogger(__name__)


@dataclass
class TextQualityMetrics:
    """Metrics for evaluating text layer quality"""
    total_chars: int
    printable_ratio: float
    word_count: int
    avg_word_length: float
    whitespace_ratio: float
    alphanumeric_ratio: float
    sentence_markers: int
    unicode_errors: int
    common_words_found: int
    text_density: float  # Characters per page area
    confidence_score: float  # Overall 0-1 score
    
    # Coverage detection metrics
    text_coverage_ratio: float = 0.0  # % of page covered by text bboxes
    has_images: bool = False  # Page contains images
    has_drawings: bool = False  # Page contains vector graphics
    image_coverage_ratio: float = 0.0  # % of page covered by images
    uncovered_area_ratio: float = 0.0  # % of page with no text layer
    coverage_confidence: float = 1.0  # Confidence that coverage is adequate (0-1)
    text_blocks_count: int = 0  # Number of text blocks detected


@dataclass
class TextQualityThresholds:
    """Configurable thresholds for text quality validation"""
    min_chars: int = 100
    min_printable_ratio: float = 0.85
    min_alphanumeric_ratio: float = 0.60
    max_whitespace_ratio: float = 0.40
    min_avg_word_length: float = 2.0
    max_avg_word_length: float = 20.0
    min_words: int = 10
    min_confidence_score: float = 0.65
    min_text_density: float = 0.001  # chars per square point
    
    # Coverage detection thresholds
    enable_coverage_detection: bool = True  # Feature toggle
    min_text_coverage_ratio: float = 0.70  # Minimum text coverage (70%)
    max_uncovered_area_ratio: float = 0.30  # Maximum uncovered area (30%)
    min_coverage_confidence: float = 0.50  # Minimum coverage confidence


class TextLayerValidator:
    """Validates quality of embedded text layers in PDFs"""

    # Common English words for sanity checking (can be expanded/localized)
    COMMON_WORDS: Set[str] = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'is',
        'was', 'are', 'been', 'has', 'had', 'were', 'said', 'did', 'get', 'may',
        'your', 'can', 'who', 'what', 'which', 'their', 'if', 'so', 'up', 'out',
        'about', 'into', 'than', 'them', 'could', 'some', 'these', 'me', 'only'
    }

    def __init__(self, thresholds: Optional[TextQualityThresholds] = None):
        self.thresholds = thresholds or TextQualityThresholds()

    def has_valid_text_layer(
        self,
        page,
        require_high_confidence: bool = False
    ) -> Tuple[bool, TextQualityMetrics]:
        """
        Determine if a PDF page has a valid, usable text layer.

        Args:
            page: PyMuPDF page object
            require_high_confidence: If True, use stricter thresholds (80% vs 65%)

        Returns:
            Tuple of (is_valid, metrics)
        """
        # Extract text
        text = page.get_text().strip()

        # Calculate metrics
        metrics = self._calculate_metrics(text, page)

        # Adjust thresholds for high confidence mode
        if require_high_confidence:
            min_confidence = 0.80
        else:
            min_confidence = self.thresholds.min_confidence_score

        # Validate against thresholds
        is_valid = (
            metrics.total_chars >= self.thresholds.min_chars and
            metrics.printable_ratio >= self.thresholds.min_printable_ratio and
            metrics.alphanumeric_ratio >= self.thresholds.min_alphanumeric_ratio and
            metrics.whitespace_ratio <= self.thresholds.max_whitespace_ratio and
            metrics.word_count >= self.thresholds.min_words and
            metrics.avg_word_length >= self.thresholds.min_avg_word_length and
            metrics.avg_word_length <= self.thresholds.max_avg_word_length and
            metrics.confidence_score >= min_confidence and
            metrics.text_density >= self.thresholds.min_text_density
        )
        
        # Add coverage validation if enabled
        # Skip coverage validation for high-quality text content (e.g., invisible searchable PDF layers)
        # Coverage checks are mainly for detecting partial OCR on scanned documents
        if is_valid and self.thresholds.enable_coverage_detection:
            if metrics.confidence_score < 0.80:
                logger.debug(
                    f"Applying coverage validation (uncertain quality): "
                    f"confidence={metrics.confidence_score:.2%}, "
                    f"coverage_conf={metrics.coverage_confidence:.2%}"
                )
                is_valid = (
                    is_valid and
                    metrics.coverage_confidence >= self.thresholds.min_coverage_confidence and
                    metrics.text_coverage_ratio >= self.thresholds.min_text_coverage_ratio and
                    metrics.uncovered_area_ratio <= self.thresholds.max_uncovered_area_ratio
                )
            else:
                # High-quality text, coverage validation skipped
                logger.debug(
                    f"Skipped coverage validation (high quality text): "
                    f"confidence={metrics.confidence_score:.2%}"
                )

        if not is_valid:
            logger.debug(
                f"Text layer failed validation: "
                f"chars={metrics.total_chars}, "
                f"confidence={metrics.confidence_score:.2%}, "
                f"coverage={metrics.text_coverage_ratio:.2%}, "
                f"coverage_conf={metrics.coverage_confidence:.2%}"
            )

        return is_valid, metrics

    def _calculate_metrics(self, text: str, page) -> TextQualityMetrics:
        """Calculate comprehensive text quality metrics"""

        if not text:
            return TextQualityMetrics(
                total_chars=0,
                printable_ratio=0.0,
                word_count=0,
                avg_word_length=0.0,
                whitespace_ratio=0.0,
                alphanumeric_ratio=0.0,
                sentence_markers=0,
                unicode_errors=0,
                common_words_found=0,
                text_density=0.0,
                confidence_score=0.0,
                # Coverage fields (defaults for empty text)
                text_coverage_ratio=0.0,
                has_images=False,
                has_drawings=False,
                image_coverage_ratio=0.0,
                uncovered_area_ratio=1.0,
                coverage_confidence=0.0,
                text_blocks_count=0
            )

        total_chars = len(text)

        # 1. Printable character ratio
        printable_count = sum(1 for c in text if c.isprintable() or c.isspace())
        printable_ratio = printable_count / total_chars if total_chars > 0 else 0.0

        # 2. Alphanumeric ratio
        alphanumeric_count = sum(1 for c in text if c.isalnum())
        alphanumeric_ratio = alphanumeric_count / total_chars if total_chars > 0 else 0.0

        # 3. Whitespace ratio
        whitespace_count = sum(1 for c in text if c.isspace())
        whitespace_ratio = whitespace_count / total_chars if total_chars > 0 else 0.0

        # 4. Word analysis
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        word_count = len(words)
        avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0

        # 5. Common words check (language sanity)
        common_words_found = sum(1 for w in words if w in self.COMMON_WORDS)

        # 6. Sentence structure markers
        sentence_markers = text.count('.') + text.count('!') + text.count('?')

        # 7. Unicode errors/corrupted encoding detection
        unicode_errors = self._count_unicode_errors(text)

        # 8. Text density (chars per page area)
        page_area = page.rect.width * page.rect.height
        text_density = total_chars / page_area if page_area > 0 else 0.0

        # 9. Calculate overall confidence score
        confidence_score = self._calculate_confidence(
            printable_ratio=printable_ratio,
            alphanumeric_ratio=alphanumeric_ratio,
            whitespace_ratio=whitespace_ratio,
            avg_word_length=avg_word_length,
            word_count=word_count,
            common_words_found=common_words_found,
            unicode_errors=unicode_errors,
            text_density=text_density,
            total_chars=total_chars
        )

        # 10. Calculate coverage metrics if enabled
        coverage_metrics = {}
        if self.thresholds.enable_coverage_detection:
            coverage_metrics = self._calculate_coverage_metrics(page)
            # Adjust overall confidence based on coverage confidence
            # ONLY if base confidence is uncertain (< 0.80)
            # High-quality text should not be penalized for low spatial coverage
            if coverage_metrics['coverage_confidence'] < 0.5 and confidence_score < 0.80:
                confidence_score *= coverage_metrics['coverage_confidence']
        else:
            coverage_metrics = self._empty_coverage_metrics()

        return TextQualityMetrics(
            total_chars=total_chars,
            printable_ratio=printable_ratio,
            word_count=word_count,
            avg_word_length=avg_word_length,
            whitespace_ratio=whitespace_ratio,
            alphanumeric_ratio=alphanumeric_ratio,
            sentence_markers=sentence_markers,
            unicode_errors=unicode_errors,
            common_words_found=common_words_found,
            text_density=text_density,
            confidence_score=confidence_score,
            # Coverage metrics
            text_coverage_ratio=coverage_metrics['text_coverage_ratio'],
            has_images=coverage_metrics['has_images'],
            has_drawings=coverage_metrics['has_drawings'],
            image_coverage_ratio=coverage_metrics['image_coverage_ratio'],
            uncovered_area_ratio=coverage_metrics['uncovered_area_ratio'],
            coverage_confidence=coverage_metrics['coverage_confidence'],
            text_blocks_count=coverage_metrics['text_blocks_count']
        )

    def _count_unicode_errors(self, text: str) -> int:
        """Count characters that indicate encoding problems"""
        error_indicators = [
            '\ufffd',  # Replacement character
            '\x00',    # Null character
        ]

        error_count = sum(text.count(indicator) for indicator in error_indicators)

        # Check for excessive control characters (non-printable, non-space)
        control_chars = sum(
            1 for c in text
            if unicodedata.category(c)[0] == 'C' and not c.isspace()
        )

        return error_count + control_chars

    def _empty_coverage_metrics(self) -> dict:
        """
        Return conservative coverage metrics for error cases.

        Uses confidence=0.0 to trigger OCR (fail-safe behavior).
        If coverage detection fails, we should OCR the page to be safe.
        """
        return {
            'text_coverage_ratio': 0.0,
            'has_images': False,
            'has_drawings': False,
            'image_coverage_ratio': 0.0,
            'uncovered_area_ratio': 1.0,
            'coverage_confidence': 0.0,  # ✅ FIXED: Trigger OCR on detection failure
            'text_blocks_count': 0
        }

    def _calculate_coverage_confidence(
        self,
        text_coverage_ratio: float,
        has_images: bool,
        has_drawings: bool,
        uncovered_area_ratio: float,
        text_blocks_count: int
    ) -> float:
        """
        Calculate confidence that text layer coverage is adequate.
        
        Args:
            text_coverage_ratio: Fraction of page covered by text bboxes
            has_images: Whether page contains images
            has_drawings: Whether page contains vector graphics
            uncovered_area_ratio: Fraction of page with no text layer
            text_blocks_count: Number of text blocks detected
            
        Returns:
            Confidence score 0-1 (1 = high confidence coverage is good)
        """
        # Base confidence on text coverage ratio
        if text_coverage_ratio >= 0.8:
            confidence = 1.0
        elif text_coverage_ratio >= 0.6:
            confidence = 0.9
        elif text_coverage_ratio >= 0.4:
            confidence = 0.7
        elif text_coverage_ratio >= 0.2:
            confidence = 0.4
        else:
            confidence = 0.2
        
        # Penalize large uncovered areas if images/drawings present
        # This catches cases where header has text but scanned body doesn't
        if (has_images or has_drawings) and uncovered_area_ratio > 0.3:
            # Likely scanned document with partial OCR text layer
            confidence *= 0.5
        
        # Penalize very few text blocks (likely incomplete OCR)
        if text_blocks_count < 3 and text_coverage_ratio < 0.5:
            confidence *= 0.6
        
        return max(0.0, min(1.0, confidence))

    def _calculate_coverage_metrics(self, page) -> dict:
        """
        Detect if text layer covers all visible content on the page.
        
        Uses bounding box analysis to estimate spatial coverage of text layer.
        Helps detect pages where text layer only covers part of the page
        (e.g., header has text layer but scanned body doesn't).
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Dict with coverage metrics:
                - text_coverage_ratio: % of page covered by text bboxes
                - has_images: Boolean, page contains images
                - has_drawings: Boolean, page contains vector graphics
                - image_coverage_ratio: % of page covered by images
                - uncovered_area_ratio: % of page with no text layer
                - coverage_confidence: Confidence score 0-1
                - text_blocks_count: Number of text blocks detected
        """
        try:
            import fitz
            
            # Get page dimensions
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height
            
            if page_area <= 0:
                logger.warning("Page has zero or negative area")
                return self._empty_coverage_metrics()
            
            # 1. Get text block bounding boxes
            try:
                blocks = page.get_text("blocks")  # Returns [(x0,y0,x1,y1,text,block_no,block_type),...]
            except Exception as e:
                logger.warning(f"Failed to get text blocks: {e}")
                blocks = []
            
            # Calculate total area covered by text blocks
            text_area = 0.0
            text_rects = []
            
            for block in blocks:
                if len(block) >= 4:
                    x0, y0, x1, y1 = block[:4]
                    text_rect = fitz.Rect(x0, y0, x1, y1)
                    
                    # Clip to page bounds (safety check)
                    text_rect = text_rect & page_rect
                    
                    if not text_rect.is_empty:
                        text_rects.append(text_rect)
                        text_area += text_rect.width * text_rect.height
            
            # 2. Detect images
            try:
                images = page.get_images(full=False)
                has_images = len(images) > 0
            except Exception as e:
                logger.debug(f"Failed to get images: {e}")
                has_images = False
                images = []
            
            # Calculate image coverage (approximate)
            image_area = 0.0
            if has_images:
                try:
                    # Get detailed image info with positions
                    image_info = page.get_image_info()
                    for img in image_info:
                        if 'bbox' in img:
                            img_rect = fitz.Rect(img['bbox'])
                            img_rect = img_rect & page_rect
                            if not img_rect.is_empty:
                                image_area += img_rect.width * img_rect.height
                except Exception as e:
                    logger.debug(f"Failed to get image positions: {e}")
                    # Fallback: assume images take significant space if present
                    image_area = page_area * 0.3  # Conservative estimate
            
            # 3. Detect vector drawings
            try:
                drawings = page.get_drawings()
                has_drawings = len(drawings) > 0
            except Exception as e:
                logger.debug(f"Failed to get drawings: {e}")
                has_drawings = False
            
            # 4. Calculate ratios
            text_coverage_ratio = min(1.0, text_area / page_area)
            image_coverage_ratio = min(1.0, image_area / page_area)
            
            # Total covered area (conservative - may have overlap)
            total_covered_area = text_area + image_area
            total_coverage_ratio = min(1.0, total_covered_area / page_area)
            
            uncovered_area_ratio = max(0.0, 1.0 - total_coverage_ratio)
            
            # 5. Calculate coverage confidence
            coverage_confidence = self._calculate_coverage_confidence(
                text_coverage_ratio=text_coverage_ratio,
                has_images=has_images,
                has_drawings=has_drawings,
                uncovered_area_ratio=uncovered_area_ratio,
                text_blocks_count=len(text_rects)
            )
            
            logger.debug(
                f"Coverage metrics: text={text_coverage_ratio:.2f}, "
                f"images={has_images}, drawings={has_drawings}, "
                f"uncovered={uncovered_area_ratio:.2f}, "
                f"confidence={coverage_confidence:.2f}"
            )
            
            return {
                'text_coverage_ratio': text_coverage_ratio,
                'has_images': has_images,
                'has_drawings': has_drawings,
                'image_coverage_ratio': image_coverage_ratio,
                'uncovered_area_ratio': uncovered_area_ratio,
                'coverage_confidence': coverage_confidence,
                'text_blocks_count': len(text_rects)
            }
            
        except Exception as e:
            logger.warning(f"Coverage detection failed: {e}, using quality-only validation")
            return self._empty_coverage_metrics()

    def _calculate_confidence(
        self,
        printable_ratio: float,
        alphanumeric_ratio: float,
        whitespace_ratio: float,
        avg_word_length: float,
        word_count: int,
        common_words_found: int,
        unicode_errors: int,
        text_density: float,
        total_chars: int
    ) -> float:
        """
        Calculate overall confidence score (0-1) for text quality.

        Uses weighted scoring of multiple factors.
        """
        scores = []

        # 1. Printable ratio score (weight: 0.15)
        scores.append(min(printable_ratio / 0.95, 1.0) * 0.15)

        # 2. Alphanumeric ratio score (weight: 0.15)
        scores.append(min(alphanumeric_ratio / 0.70, 1.0) * 0.15)

        # 3. Whitespace ratio score (weight: 0.10)
        # Ideal range: 0.10 - 0.25
        if 0.10 <= whitespace_ratio <= 0.25:
            whitespace_score = 1.0
        elif whitespace_ratio < 0.10:
            whitespace_score = whitespace_ratio / 0.10
        else:
            whitespace_score = max(0, 1.0 - (whitespace_ratio - 0.25) / 0.15)
        scores.append(whitespace_score * 0.10)

        # 4. Average word length score (weight: 0.15)
        # Ideal range: 4-7 characters
        if 4.0 <= avg_word_length <= 7.0:
            word_length_score = 1.0
        elif 2.0 <= avg_word_length < 4.0:
            word_length_score = (avg_word_length - 2.0) / 2.0
        elif 7.0 < avg_word_length <= 15.0:
            word_length_score = max(0, 1.0 - (avg_word_length - 7.0) / 8.0)
        else:
            word_length_score = 0.0
        scores.append(word_length_score * 0.15)

        # 5. Word count score (weight: 0.10)
        word_count_score = min(word_count / 50, 1.0)  # 50+ words = full score
        scores.append(word_count_score * 0.10)

        # 6. Common words score (weight: 0.15)
        if word_count > 0:
            common_word_ratio = common_words_found / word_count
            # At least 10% of words should be common
            common_words_score = min(common_word_ratio / 0.10, 1.0)
        else:
            common_words_score = 0.0
        scores.append(common_words_score * 0.15)

        # 7. Unicode errors penalty (weight: 0.10)
        if total_chars > 0:
            error_ratio = unicode_errors / total_chars
            unicode_score = max(0, 1.0 - error_ratio * 10)  # Heavy penalty
        else:
            unicode_score = 0.0
        scores.append(unicode_score * 0.10)

        # 8. Text density score (weight: 0.10)
        # Reasonable density: 0.001 - 0.1 chars per square point
        if 0.001 <= text_density <= 0.1:
            density_score = 1.0
        elif text_density < 0.001:
            density_score = text_density / 0.001
        else:
            density_score = max(0, 1.0 - (text_density - 0.1) / 0.1)
        scores.append(density_score * 0.10)

        return sum(scores)

    def generate_quality_report(self, metrics: TextQualityMetrics) -> str:
        """Generate human-readable quality report"""
        status = 'VALID' if metrics.confidence_score >= self.thresholds.min_confidence_score else 'INVALID - OCR RECOMMENDED'

        # Build coverage section if enabled
        coverage_section = ""
        if self.thresholds.enable_coverage_detection:
            coverage_section = f"""
Coverage Analysis:
------------------
Text Coverage: {metrics.text_coverage_ratio:.2%}
Image Coverage: {metrics.image_coverage_ratio:.2%}
Uncovered Area: {metrics.uncovered_area_ratio:.2%}
Text Blocks: {metrics.text_blocks_count}
Has Images: {metrics.has_images}
Has Drawings: {metrics.has_drawings}
Coverage Confidence: {metrics.coverage_confidence:.2%}
"""

        return f"""
Text Quality Report:
-------------------
Total Characters: {metrics.total_chars}
Printable Ratio: {metrics.printable_ratio:.2%}
Alphanumeric Ratio: {metrics.alphanumeric_ratio:.2%}
Whitespace Ratio: {metrics.whitespace_ratio:.2%}
Word Count: {metrics.word_count}
Avg Word Length: {metrics.avg_word_length:.1f}
Common Words: {metrics.common_words_found}
Sentence Markers: {metrics.sentence_markers}
Unicode Errors: {metrics.unicode_errors}
Text Density: {metrics.text_density:.4f}
{coverage_section}
Overall Confidence: {metrics.confidence_score:.2%}
Status: {status}
"""


class OCRDecisionEngine:
    """Engine to decide between using text layer or performing OCR"""

    def __init__(self, validator: Optional[TextLayerValidator] = None):
        self.validator = validator or TextLayerValidator()
        self.user_preferences = {
            'auto_ocr_on_low_quality': True,
            'preview_before_ocr': False,
            'force_ocr_all': False,
            'quality_threshold': 'normal'  # 'normal' or 'strict'
        }

    def set_preference(self, key: str, value):
        """Update a user preference"""
        if key in self.user_preferences:
            self.user_preferences[key] = value
            logger.info(f"Updated preference {key} = {value}")

    def should_perform_ocr(
        self,
        page,
        user_override: Optional[bool] = None
    ) -> Tuple[bool, str, Optional[TextQualityMetrics]]:
        """
        Decide whether to perform OCR on a page.

        Returns:
            Tuple of (should_ocr, reason, metrics)
        """
        # User override takes precedence
        if user_override is not None:
            if user_override:
                return True, "User forced OCR", None
            else:
                return False, "User forced text extraction", None

        # Force OCR mode
        if self.user_preferences['force_ocr_all']:
            return True, "Force OCR mode enabled", None

        # Validate text layer
        require_strict = self.user_preferences['quality_threshold'] == 'strict'
        is_valid, metrics = self.validator.has_valid_text_layer(
            page,
            require_high_confidence=require_strict
        )

        if is_valid:
            return False, f"Valid text layer (confidence: {metrics.confidence_score:.2%})", metrics
        else:
            reason = self._explain_invalidity(metrics)
            return True, reason, metrics

    def _explain_invalidity(self, metrics: TextQualityMetrics) -> str:
        """Explain why text layer is invalid"""
        reasons = []

        if metrics.total_chars < self.validator.thresholds.min_chars:
            reasons.append(f"insufficient text ({metrics.total_chars} chars)")

        if metrics.printable_ratio < self.validator.thresholds.min_printable_ratio:
            reasons.append(f"low printable ratio ({metrics.printable_ratio:.1%})")

        if metrics.alphanumeric_ratio < self.validator.thresholds.min_alphanumeric_ratio:
            reasons.append(f"low alphanumeric ratio ({metrics.alphanumeric_ratio:.1%})")

        if metrics.unicode_errors > 0:
            reasons.append(f"encoding errors ({metrics.unicode_errors})")

        if metrics.common_words_found < 5:
            reasons.append("few recognizable words")

        if metrics.text_density < self.validator.thresholds.min_text_density:
            reasons.append("sparse text density")

        if not reasons:
            reasons.append(f"low confidence ({metrics.confidence_score:.1%})")

        return "OCR needed: " + ", ".join(reasons)
