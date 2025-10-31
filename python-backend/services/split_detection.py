"""
Split Detection Service
Combines heuristic rules and clustering to detect document boundaries
"""

import logging
import re
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import DBSCAN
from collections import Counter

logger = logging.getLogger(__name__)


class SplitDetector:
    """
    Detects document split points using multiple signals:
    - Heuristic rules (page numbers, headers, blank pages)
    - Semantic clustering (embeddings)
    - Layout analysis
    """

    # Confidence thresholds
    THRESHOLD_HIGH = 0.8  # Auto-accept
    THRESHOLD_MEDIUM = 0.5  # Mark for LLM review
    THRESHOLD_LOW = 0.3  # Ignore

    def __init__(self):
        self.signals = []

    def detect_page_number_reset(self, pages: List[Dict]) -> List[Tuple[int, float, str]]:
        """
        Detect page number resets (1,2,3 -> 1,2,3).

        Returns:
            List of (page_num, confidence, reason)
        """
        splits = []

        for i in range(1, len(pages)):
            curr_page = pages[i]
            prev_page = pages[i - 1]

            # Extract page numbers from text
            curr_num = self._extract_page_number(curr_page['text'])
            prev_num = self._extract_page_number(prev_page['text'])

            if curr_num and prev_num:
                # Check for reset (previous high, current low)
                if prev_num >= 2 and curr_num == 1:
                    splits.append((
                        i,
                        0.9,
                        f"Page number reset: {prev_num} -> {curr_num}"
                    ))
                elif curr_num < prev_num - 5:  # Large backward jump
                    splits.append((
                        i,
                        0.7,
                        f"Page number jump backward: {prev_num} -> {curr_num}"
                    ))

        return splits

    def _extract_page_number(self, text: str) -> Optional[int]:
        """Extract page number from text"""
        if not text:
            return None

        # Look for common page number patterns
        patterns = [
            r'Page\s+(\d+)',
            r'p\.\s*(\d+)',
            r'^(\d+)$',  # Standalone number
            r'[-−]\s*(\d+)\s*[-−]',  # - 5 -
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    page_num = int(match.group(1))
                    # Validate: reasonable page number range (1-9999)
                    # Filters out years (2024), amounts, phone numbers, etc.
                    if 1 <= page_num <= 9999:
                        return page_num
                except ValueError:
                    continue

        return None

    def detect_header_footer_changes(self, pages: List[Dict]) -> List[Tuple[int, float, str]]:
        """
        Detect changes in header/footer patterns.

        Returns:
            List of (page_num, confidence, reason)
        """
        splits = []

        # Extract headers/footers for all pages
        headers = []
        footers = []

        for page in pages:
            text = page['text'] or ''
            lines = text.split('\n')

            if len(lines) >= 2:
                headers.append(lines[0].strip())
                footers.append(lines[-1].strip())
            else:
                headers.append('')
                footers.append('')

        # Detect pattern changes
        # Only flag changes when both headers/footers are meaningful (non-empty, >= 3 chars)
        MIN_LENGTH = 3  # Minimum length for meaningful header/footer
        
        for i in range(1, len(pages)):
            # Check if current and previous headers/footers are meaningful
            curr_header_valid = len(headers[i]) >= MIN_LENGTH
            prev_header_valid = len(headers[i-1]) >= MIN_LENGTH
            curr_footer_valid = len(footers[i]) >= MIN_LENGTH
            prev_footer_valid = len(footers[i-1]) >= MIN_LENGTH
            
            # Only flag changes when both sides have valid content
            header_changed = (
                curr_header_valid and prev_header_valid and 
                headers[i] != headers[i-1]
            )
            footer_changed = (
                curr_footer_valid and prev_footer_valid and 
                footers[i] != footers[i-1]
            )

            if header_changed and footer_changed:
                splits.append((
                    i,
                    0.6,
                    "Both header and footer changed"
                ))
            elif header_changed or footer_changed:
                splits.append((
                    i,
                    0.4,
                    "Header or footer changed"
                ))

        return splits

    def detect_blank_pages(self, pages: List[Dict]) -> List[Tuple[int, float, str]]:
        """
        Detect blank separator pages.

        Returns:
            List of (page_num, confidence, reason)
        """
        splits = []

        for i, page in enumerate(pages):
            text = page['text'] or ''
            text_length = len(text.strip())

            if text_length < 50:  # Very little text
                # Check if next page has content
                if i + 1 < len(pages):
                    next_text = pages[i + 1]['text'] or ''
                    if len(next_text.strip()) > 100:
                        splits.append((
                            i + 1,  # Split AFTER blank page
                            0.85,
                            f"Blank separator page (only {text_length} chars)"
                        ))

        return splits

    def detect_semantic_discontinuity(
        self,
        doc_id: str,
        embeddings: np.ndarray
    ) -> List[Tuple[int, float, str]]:
        """
        Detect semantic discontinuity using embedding similarity.

        Args:
            doc_id: Document ID
            embeddings: Array of page embeddings (n, 768)

        Returns:
            List of (page_num, confidence, reason)
        """
        splits = []

        # Compute consecutive page similarities
        for i in range(1, len(embeddings)):
            similarity = float(np.dot(embeddings[i-1], embeddings[i]))

            # Low similarity = likely different documents
            if similarity < 0.3:
                splits.append((
                    i,
                    0.7,
                    f"Low semantic similarity ({similarity:.2f})"
                ))
            elif similarity < 0.5:
                splits.append((
                    i,
                    0.5,
                    f"Moderate semantic discontinuity ({similarity:.2f})"
                ))

        return splits

    def detect_with_clustering(
        self,
        embeddings: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 3
    ) -> List[Tuple[int, float, str]]:
        """
        Use DBSCAN clustering to detect document boundaries.

        Args:
            embeddings: Page embeddings
            eps: DBSCAN epsilon parameter
            min_samples: Minimum cluster size

        Returns:
            List of (page_num, confidence, reason)
        """
        if len(embeddings) < min_samples:
            return []

        # Run DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        splits = []

        # Detect cluster boundaries
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                # Cluster changed = potential split
                splits.append((
                    i,
                    0.6,
                    f"Cluster boundary (cluster {labels[i-1]} -> {labels[i]})"
                ))

        return splits

    def combine_signals(
        self,
        all_splits: List[List[Tuple[int, float, str]]]
    ) -> List[Dict[str, Any]]:
        """
        Combine multiple detection signals into final split candidates.

        Args:
            all_splits: List of split lists from different detectors

        Returns:
            List of split candidates with aggregated confidence
        """
        # Aggregate by page number
        page_scores = {}
        page_reasons = {}

        for split_list in all_splits:
            for page_num, confidence, reason in split_list:
                if page_num not in page_scores:
                    page_scores[page_num] = []
                    page_reasons[page_num] = []

                page_scores[page_num].append(confidence)
                page_reasons[page_num].append(reason)

        # Compute final candidates
        candidates = []

        for page_num in sorted(page_scores.keys()):
            scores = page_scores[page_num]
            reasons = page_reasons[page_num]

            # Aggregate confidence (max + bonus for multiple signals)
            max_conf = max(scores)
            bonus = min(0.2, (len(scores) - 1) * 0.1)  # Up to 0.2 bonus
            final_conf = min(1.0, max_conf + bonus)

            if final_conf >= self.THRESHOLD_LOW:
                candidates.append({
                    'page': page_num,
                    'confidence': final_conf,
                    'signals': reasons,
                    'signal_count': len(reasons)
                })

        return candidates


def detect_splits_for_document(
    doc_id: str,
    progress_callback=None
) -> int:
    """
    Detect split points for a document.

    Args:
        doc_id: Document ID
        progress_callback: Optional callback(current, total, message)

    Returns:
        Number of splits detected
    """
    from .cache_manager import get_cache_manager

    cache = get_cache_manager()
    detector = SplitDetector()

    if progress_callback:
        progress_callback(0, 5, "Loading document data...")

    # Load pages and embeddings
    pages = cache.get_all_pages(doc_id)

    if not pages:
        logger.warning(f"No pages found for document {doc_id[:8]}...")
        return 0

    # Get embeddings
    embeddings = []
    for page in pages:
        emb = cache.get_page_embedding(doc_id, page['page_num'])
        if emb is not None:
            embeddings.append(emb)
        else:
            logger.warning(f"Missing embedding for page {page['page_num']}")
            embeddings.append(np.zeros(768))  # Placeholder

    embeddings = np.array(embeddings)

    if progress_callback:
        progress_callback(1, 5, "Running heuristic detection...")

    # Run all detectors
    all_splits = []

    # Heuristic detectors
    all_splits.append(detector.detect_page_number_reset(pages))
    all_splits.append(detector.detect_header_footer_changes(pages))
    all_splits.append(detector.detect_blank_pages(pages))

    if progress_callback:
        progress_callback(2, 5, "Analyzing semantic similarity...")

    # Embedding-based detectors
    all_splits.append(detector.detect_semantic_discontinuity(doc_id, embeddings))

    if progress_callback:
        progress_callback(3, 5, "Running clustering analysis...")

    all_splits.append(detector.detect_with_clustering(embeddings))

    if progress_callback:
        progress_callback(4, 5, "Combining signals...")

    # Combine signals
    candidates = detector.combine_signals(all_splits)

    # Save to database
    for candidate in candidates:
        method = 'heuristic' if candidate['confidence'] >= 0.8 else 'clustering'

        cache.save_split_candidate(
            doc_id=doc_id,
            split_page=candidate['page'],
            confidence=candidate['confidence'],
            detection_method=method,
            reasoning={
                'signals': candidate['signals'],
                'signal_count': candidate['signal_count']
            }
        )

    if progress_callback:
        progress_callback(5, 5, f"Detected {len(candidates)} potential splits")

    logger.info(f"Detected {len(candidates)} splits for document {doc_id[:8]}...")
    return len(candidates)
