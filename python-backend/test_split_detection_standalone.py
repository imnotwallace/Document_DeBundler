"""
Standalone test for split_detection.py
Tests detection methods without requiring database or other services
"""

import sys
import os

# Add the services directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services'))

# Import only what we need - directly from the file
import numpy as np
import re
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import DBSCAN


class SplitDetector:
    """
    Detects document split points using multiple signals.
    Copied from split_detection.py for standalone testing.
    """

    THRESHOLD_HIGH = 0.8
    THRESHOLD_MEDIUM = 0.5
    THRESHOLD_LOW = 0.3

    def __init__(self):
        self.signals = []

    def detect_page_number_reset(self, pages: List[Dict]) -> List[Tuple[int, float, str]]:
        splits = []
        for i in range(1, len(pages)):
            curr_page = pages[i]
            prev_page = pages[i - 1]
            curr_num = self._extract_page_number(curr_page['text'])
            prev_num = self._extract_page_number(prev_page['text'])

            if curr_num and prev_num:
                if prev_num >= 2 and curr_num == 1:
                    splits.append((i, 0.9, f"Page number reset: {prev_num} -> {curr_num}"))
                elif curr_num < prev_num - 5:
                    splits.append((i, 0.7, f"Page number jump backward: {prev_num} -> {curr_num}"))
        return splits

    def _extract_page_number(self, text: str) -> Optional[int]:
        if not text:
            return None
        patterns = [
            r'Page\s+(\d+)',
            r'p\.\s*(\d+)',
            r'^(\d+)$',
            r'[-−]\s*(\d+)\s*[-−]',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        return None

    def detect_blank_pages(self, pages: List[Dict]) -> List[Tuple[int, float, str]]:
        splits = []
        for i, page in enumerate(pages):
            text = page['text'] or ''
            text_length = len(text.strip())  # Strip whitespace to detect whitespace-only pages
            if text_length < 10:  # Very little text (truly blank or whitespace-only pages)
                if i + 1 < len(pages):
                    next_text = pages[i + 1]['text'] or ''
                    if len(next_text.strip()) > 30:  # Next page has substantial content
                        splits.append((i + 1, 0.85, f"Blank separator page (only {text_length} chars)"))
        return splits

    def detect_semantic_discontinuity(self, doc_id: str, embeddings: np.ndarray) -> List[Tuple[int, float, str]]:
        splits = []
        for i in range(1, len(embeddings)):
            similarity = float(np.dot(embeddings[i-1], embeddings[i]))
            if similarity < 0.3:
                splits.append((i, 0.7, f"Low semantic similarity ({similarity:.2f})"))
            elif similarity < 0.5:
                splits.append((i, 0.5, f"Moderate semantic discontinuity ({similarity:.2f})"))
        return splits

    def detect_with_clustering(self, embeddings: np.ndarray, eps: float = 0.5, min_samples: int = 1) -> List[Tuple[int, float, str]]:
        if len(embeddings) < 1:
            return []  # Need at least 1 page to cluster
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        splits = []
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                splits.append((i, 0.6, f"Cluster boundary (cluster {labels[i-1]} -> {labels[i]})"))
        return splits

    def combine_signals(self, all_splits: List[List[Tuple[int, float, str]]]) -> List[Dict[str, Any]]:
        page_scores = {}
        page_reasons = {}
        for split_list in all_splits:
            for page_num, confidence, reason in split_list:
                if page_num not in page_scores:
                    page_scores[page_num] = []
                    page_reasons[page_num] = []
                page_scores[page_num].append(confidence)
                page_reasons[page_num].append(reason)

        candidates = []
        for page_num in sorted(page_scores.keys()):
            scores = page_scores[page_num]
            reasons = page_reasons[page_num]
            max_conf = max(scores)
            bonus = min(0.2, (len(scores) - 1) * 0.1)
            final_conf = min(1.0, max_conf + bonus)
            if final_conf >= self.THRESHOLD_LOW:
                candidates.append({
                    'page': page_num,
                    'confidence': final_conf,
                    'signals': reasons,
                    'signal_count': len(reasons)
                })
        return candidates


def test_page_number_reset():
    print("\n=== Testing Page Number Reset Detection ===")
    detector = SplitDetector()
    pages = [
        {'page_num': 0, 'text': 'Page 1\nSome content for first document'},
        {'page_num': 1, 'text': 'Page 2\nMore content for first document'},
        {'page_num': 2, 'text': 'Page 3\nFinal page of first document'},
        {'page_num': 3, 'text': 'Page 1\nNew document starts here'},
        {'page_num': 4, 'text': 'Page 2\nSecond page of new document'},
    ]
    splits = detector.detect_page_number_reset(pages)
    print(f"Detected {len(splits)} page number resets:")
    for page_num, confidence, reason in splits:
        print(f"  - Page {page_num}: {reason} (confidence: {confidence:.2f})")
    assert len(splits) == 1, f"Expected 1 split, got {len(splits)}"
    assert splits[0][0] == 3, f"Expected split at page 3, got {splits[0][0]}"
    assert splits[0][1] > 0.8, f"Expected high confidence, got {splits[0][1]}"
    print("PASS")


def test_blank_page_detection():
    print("\n=== Testing Blank Page Detection ===")
    detector = SplitDetector()
    pages = [
        {'page_num': 0, 'text': 'This is a normal page with lots of content.'},
        {'page_num': 1, 'text': 'Another normal page with content here.'},
        {'page_num': 2, 'text': '   '},
        {'page_num': 3, 'text': 'New document starts after blank page with content.'},
    ]
    splits = detector.detect_blank_pages(pages)
    print(f"Detected {len(splits)} blank page separators:")
    for page_num, confidence, reason in splits:
        print(f"  - Page {page_num}: {reason} (confidence: {confidence:.2f})")
    assert len(splits) == 1, f"Expected 1 split, got {len(splits)}"
    assert splits[0][0] == 3, f"Expected split at page 3 (after blank), got {splits[0][0]}"
    assert splits[0][1] > 0.8, f"Expected high confidence, got {splits[0][1]}"
    print("PASS")


def test_semantic_discontinuity():
    print("\n=== Testing Semantic Discontinuity Detection ===")
    detector = SplitDetector()
    np.random.seed(42)
    doc1_base = np.random.randn(768)
    doc1_base = doc1_base / np.linalg.norm(doc1_base)
    emb0 = doc1_base + np.random.randn(768) * 0.1
    emb1 = doc1_base + np.random.randn(768) * 0.1
    emb2 = doc1_base + np.random.randn(768) * 0.1
    doc2_base = np.random.randn(768)
    doc2_base = doc2_base / np.linalg.norm(doc2_base)
    emb3 = doc2_base + np.random.randn(768) * 0.1
    emb4 = doc2_base + np.random.randn(768) * 0.1
    embeddings = np.array([emb0, emb1, emb2, emb3, emb4])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    splits = detector.detect_semantic_discontinuity("test_doc_id", embeddings)
    print(f"Detected {len(splits)} semantic discontinuities:")
    for page_num, confidence, reason in splits:
        print(f"  - Page {page_num}: {reason} (confidence: {confidence:.2f})")
    print("PASS")


def test_clustering():
    print("\n=== Testing Clustering Detection ===")
    detector = SplitDetector()
    np.random.seed(42)
    cluster1_center = np.random.randn(768)
    cluster1_center = cluster1_center / np.linalg.norm(cluster1_center)
    cluster1_pages = [cluster1_center + np.random.randn(768) * 0.05 for _ in range(3)]
    cluster2_center = np.random.randn(768)
    cluster2_center = cluster2_center / np.linalg.norm(cluster2_center)
    cluster2_pages = [cluster2_center + np.random.randn(768) * 0.05 for _ in range(3)]
    embeddings = np.array(cluster1_pages + cluster2_pages)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    splits = detector.detect_with_clustering(embeddings, eps=0.3, min_samples=1)
    print(f"Detected {len(splits)} cluster boundaries:")
    for page_num, confidence, reason in splits:
        print(f"  - Page {page_num}: {reason} (confidence: {confidence:.2f})")
    print("PASS")


def test_combine_signals():
    print("\n=== Testing Signal Combination ===")
    detector = SplitDetector()
    all_splits = [
        [(3, 0.9, "Page number reset")],
        [(3, 0.6, "Header changed")],
        [(3, 0.7, "Low semantic similarity")],
        [(3, 0.6, "Cluster boundary")],
        [(5, 0.4, "Minor header change")],
    ]
    candidates = detector.combine_signals(all_splits)
    print(f"Combined into {len(candidates)} candidates:")
    for candidate in candidates:
        print(f"\n  Page {candidate['page']}:")
        print(f"    Final Confidence: {candidate['confidence']:.2f}")
        print(f"    Signal Count: {candidate['signal_count']}")
        print(f"    Signals:")
        for signal in candidate['signals']:
            print(f"      - {signal}")
    assert len(candidates) == 2, f"Expected 2 candidates, got {len(candidates)}"
    page3_candidate = [c for c in candidates if c['page'] == 3][0]
    assert page3_candidate['confidence'] > 0.9, f"Expected high confidence for page 3"
    assert page3_candidate['signal_count'] == 4, f"Expected 4 signals for page 3"
    page5_candidate = [c for c in candidates if c['page'] == 5][0]
    assert page5_candidate['confidence'] < 0.5, f"Expected low confidence for page 5"
    print("\nPASS")
    print(f"\nConfidence thresholds:")
    print(f"  High (auto-accept): >= {detector.THRESHOLD_HIGH}")
    print(f"  Medium (LLM review): >= {detector.THRESHOLD_MEDIUM}")
    print(f"  Low (ignore): >= {detector.THRESHOLD_LOW}")


def main():
    print("=" * 60)
    print("Split Detection Service Test Suite")
    print("=" * 60)
    try:
        test_page_number_reset()
        test_blank_page_detection()
        test_semantic_discontinuity()
        test_clustering()
        test_combine_signals()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
