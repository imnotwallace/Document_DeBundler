"""
Split Analyzer - LLM-based Split Refinement

Uses LLM to refine DBSCAN-detected split points by analyzing semantic context,
structure, and heuristic signals.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class SplitAnalyzer:
    """
    LLM-based analyzer for refining document split points.

    Takes DBSCAN-detected candidates and uses LLM to validate/refine them
    by analyzing actual page content and structure.
    """

    def __init__(self, cache_manager=None):
        """
        Initialize SplitAnalyzer.

        Args:
            cache_manager: Optional cache manager for caching LLM decisions
        """
        self.cache = cache_manager
        logger.info("SplitAnalyzer initialized")

    def analyze_split(
        self,
        split_page: int,
        page_texts: List[str],
        heuristic_signals: Optional[List[str]] = None,
        context_pages: int = 3
    ) -> Tuple[bool, float, str]:
        """
        Analyze a single split point using LLM.

        Args:
            split_page: Page number where split is being considered
            page_texts: List of all page texts from document
            heuristic_signals: List of detected heuristic signals
            context_pages: Number of pages before/after to provide as context

        Returns:
            Tuple of (should_split, confidence, reasoning)
        """
        try:
            from services.llm.manager import get_llm_manager
            from services.llm.prompts import format_split_prompt, parse_split_decision

            # Check if LLM is available
            manager = get_llm_manager()
            if not manager.is_available():
                logger.warning("LLM not available, cannot analyze split")
                return False, 0.0, "LLM not available"

            # Check cache first
            if self.cache:
                cached = self._get_cached_decision(split_page, page_texts, heuristic_signals)
                if cached:
                    logger.debug(f"Using cached decision for split at page {split_page}")
                    return cached

            # Prepare context pages
            before_pages = self._get_context_pages(
                page_texts,
                start_idx=max(0, split_page - context_pages),
                end_idx=split_page
            )

            after_pages = self._get_context_pages(
                page_texts,
                start_idx=split_page,
                end_idx=min(len(page_texts), split_page + context_pages)
            )

            # Format prompt
            signals = heuristic_signals or []
            prompt = format_split_prompt(
                split_page=split_page,
                before_pages=before_pages,
                after_pages=after_pages,
                heuristic_signals=signals
            )

            logger.info(f"Analyzing split at page {split_page}...")
            logger.debug(f"Signals: {signals}")

            # Generate decision
            response = manager.generate(
                prompt=prompt,
                task_type="split_refinement"
            )

            if not response:
                logger.error("LLM returned empty response")
                return False, 0.0, "Empty LLM response"

            # Parse decision
            should_split, reasoning = parse_split_decision(response)

            # Calculate confidence based on LLM confidence and signal count
            confidence = self._calculate_confidence(should_split, len(signals), reasoning)

            logger.info(f"Split decision for page {split_page}: {should_split} (confidence: {confidence:.2f})")
            logger.debug(f"Reasoning: {reasoning}")

            # Cache decision
            if self.cache:
                self._cache_decision(split_page, page_texts, heuristic_signals, should_split, confidence, reasoning)

            return should_split, confidence, reasoning

        except Exception as e:
            logger.error(f"Split analysis error: {e}", exc_info=True)
            return False, 0.0, f"Analysis error: {str(e)}"

    def analyze_splits_batch(
        self,
        split_candidates: List[Dict[str, Any]],
        page_texts: List[str],
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple split candidates.

        Args:
            split_candidates: List of split candidate dicts with 'split_page' and 'signals'
            page_texts: List of all page texts
            progress_callback: Optional callback(current, total, message)

        Returns:
            List of refined split candidates with LLM decisions
        """
        results = []

        for i, candidate in enumerate(split_candidates):
            if progress_callback:
                progress_callback(
                    i,
                    len(split_candidates),
                    f"LLM analyzing split {i+1}/{len(split_candidates)}"
                )

            split_page = candidate['split_page']
            signals = candidate.get('heuristic_signals', [])

            # Analyze split
            should_split, confidence, reasoning = self.analyze_split(
                split_page=split_page,
                page_texts=page_texts,
                heuristic_signals=signals
            )

            # Create result
            result = candidate.copy()
            result.update({
                'llm_decision': should_split,
                'llm_confidence': confidence,
                'llm_reasoning': reasoning,
                'refined': True
            })

            results.append(result)

        if progress_callback:
            progress_callback(
                len(split_candidates),
                len(split_candidates),
                "LLM refinement complete"
            )

        return results

    def filter_splits_by_llm(
        self,
        split_candidates: List[Dict[str, Any]],
        page_texts: List[str],
        min_confidence: float = 0.6,
        progress_callback=None
    ) -> List[int]:
        """
        Filter split candidates using LLM analysis.

        Args:
            split_candidates: DBSCAN-detected split candidates
            page_texts: List of all page texts
            min_confidence: Minimum confidence threshold (0-1)
            progress_callback: Optional progress callback

        Returns:
            List of page numbers where splits should occur
        """
        # Analyze all candidates
        refined = self.analyze_splits_batch(
            split_candidates,
            page_texts,
            progress_callback
        )

        # Filter by LLM decision and confidence
        approved_splits = [
            candidate['split_page']
            for candidate in refined
            if candidate['llm_decision'] and candidate['llm_confidence'] >= min_confidence
        ]

        logger.info(f"LLM approved {len(approved_splits)}/{len(split_candidates)} splits (threshold: {min_confidence})")

        return sorted(approved_splits)

    def _get_context_pages(
        self,
        page_texts: List[str],
        start_idx: int,
        end_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Extract context pages for analysis.

        Args:
            page_texts: All page texts
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)

        Returns:
            List of page dicts with 'page_num' and 'text'
        """
        context = []

        for i in range(start_idx, end_idx):
            if 0 <= i < len(page_texts):
                context.append({
                    'page_num': i,
                    'text': page_texts[i]
                })

        return context

    def _calculate_confidence(
        self,
        should_split: bool,
        signal_count: int,
        reasoning: str
    ) -> float:
        """
        Calculate confidence score for split decision.

        Args:
            should_split: LLM decision
            signal_count: Number of heuristic signals detected
            reasoning: LLM reasoning text

        Returns:
            Confidence score (0-1)
        """
        # Base confidence
        if should_split:
            # Split decision: higher confidence with more signals
            base = 0.7
            signal_boost = min(signal_count * 0.05, 0.2)  # Up to +0.2
            confidence = base + signal_boost
        else:
            # No split: moderate confidence
            base = 0.6
            # Penalize if many signals but LLM says no split (conflicting)
            signal_penalty = min(signal_count * 0.03, 0.15)
            confidence = base - signal_penalty

        # Reasoning quality boost
        reasoning_lower = reasoning.lower()
        strong_keywords = ['reset', 'changes', 'new', 'different', 'boundary', 'clear']
        weak_keywords = ['unclear', 'maybe', 'possibly', 'might', 'uncertain']

        keyword_score = 0
        for kw in strong_keywords:
            if kw in reasoning_lower:
                keyword_score += 0.05

        for kw in weak_keywords:
            if kw in reasoning_lower:
                keyword_score -= 0.05

        confidence += keyword_score

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def _get_cached_decision(
        self,
        split_page: int,
        page_texts: List[str],
        signals: Optional[List[str]]
    ) -> Optional[Tuple[bool, float, str]]:
        """Get cached LLM decision if available."""
        if not self.cache:
            return None

        try:
            # Create cache key from split page and context hash
            import hashlib
            import json

            context = {
                'split_page': split_page,
                'before': page_texts[max(0, split_page-3):split_page] if split_page < len(page_texts) else [],
                'after': page_texts[split_page:min(len(page_texts), split_page+3)] if split_page < len(page_texts) else [],
                'signals': signals or []
            }

            context_str = json.dumps(context, sort_keys=True)
            cache_key = f"llm_split_{hashlib.md5(context_str.encode()).hexdigest()}"

            # Try to get from cache
            # This assumes cache has get() method
            cached = self.cache.get(cache_key)

            if cached:
                return cached['should_split'], cached['confidence'], cached['reasoning']

        except Exception as e:
            logger.debug(f"Cache lookup failed: {e}")

        return None

    def _cache_decision(
        self,
        split_page: int,
        page_texts: List[str],
        signals: Optional[List[str]],
        should_split: bool,
        confidence: float,
        reasoning: str
    ):
        """Cache LLM decision for future use."""
        if not self.cache:
            return

        try:
            import hashlib
            import json

            context = {
                'split_page': split_page,
                'before': page_texts[max(0, split_page-3):split_page] if split_page < len(page_texts) else [],
                'after': page_texts[split_page:min(len(page_texts), split_page+3)] if split_page < len(page_texts) else [],
                'signals': signals or []
            }

            context_str = json.dumps(context, sort_keys=True)
            cache_key = f"llm_split_{hashlib.md5(context_str.encode()).hexdigest()}"

            # Cache decision
            self.cache.set(cache_key, {
                'should_split': should_split,
                'confidence': confidence,
                'reasoning': reasoning
            }, ttl=86400 * 30)  # 30 days

            logger.debug(f"Cached decision for split at page {split_page}")

        except Exception as e:
            logger.debug(f"Cache write failed: {e}")


def test_split_analyzer():
    """Test function for SplitAnalyzer."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("SplitAnalyzer Test")
    print("=" * 60)

    # Mock page texts
    page_texts = [
        "Page 1: This is the first page of Document A. Important contract details...",
        "Page 2: Continued from page 1. More contract terms and conditions...",
        "Page 3: Final page of contract. Signatures and dates...",
        # Potential split here
        "Page 4: INVOICE\n\nInvoice Number: INV-001\nDate: 2024-01-15\n\nBill To: Acme Corp...",
        "Page 5: Invoice line items. Service 1: $500, Service 2: $750...",
        "Page 6: Invoice total: $1,250. Payment terms: Net 30..."
    ]

    # Create analyzer
    analyzer = SplitAnalyzer()

    # Test single split analysis
    print("\n1. Analyzing potential split at page 3...")
    should_split, confidence, reasoning = analyzer.analyze_split(
        split_page=3,
        page_texts=page_texts,
        heuristic_signals=[
            "Content type change detected",
            "Header format changed",
            "Low semantic similarity (0.15)"
        ]
    )

    print(f"  Decision: {should_split}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Reasoning: {reasoning}")

    # Test batch analysis
    print("\n2. Batch analysis of multiple candidates...")
    candidates = [
        {
            'split_page': 3,
            'heuristic_signals': [
                "Content type change",
                "Header changed",
                "Low similarity (0.15)"
            ]
        },
        {
            'split_page': 5,
            'heuristic_signals': [
                "Page continues same topic"
            ]
        }
    ]

    refined = analyzer.analyze_splits_batch(
        candidates,
        page_texts,
        progress_callback=lambda c, t, m: print(f"  {m}")
    )

    print("  Results:")
    for result in refined:
        print(f"    Page {result['split_page']}: {result['llm_decision']} (conf: {result['llm_confidence']:.2f})")

    # Test filtering
    print("\n3. Filtering splits with min confidence 0.6...")
    approved = analyzer.filter_splits_by_llm(
        candidates,
        page_texts,
        min_confidence=0.6
    )

    print(f"  Approved splits: {approved}")

    print("\n" + "=" * 60)
    print("✓ Test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_split_analyzer()
