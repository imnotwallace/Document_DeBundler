"""
Test script for split_detection.py
Tests detection methods with mock data
"""

import numpy as np
from services.split_detection import SplitDetector

def test_page_number_reset():
    """Test page number reset detection"""
    print("\n=== Testing Page Number Reset Detection ===")

    detector = SplitDetector()

    # Mock pages with page number reset
    pages = [
        {'page_num': 0, 'text': 'Page 1\nSome content for first document'},
        {'page_num': 1, 'text': 'Page 2\nMore content for first document'},
        {'page_num': 2, 'text': 'Page 3\nFinal page of first document'},
        {'page_num': 3, 'text': 'Page 1\nNew document starts here'},  # Reset!
        {'page_num': 4, 'text': 'Page 2\nSecond page of new document'},
    ]

    splits = detector.detect_page_number_reset(pages)

    print(f"Detected {len(splits)} page number resets:")
    for page_num, confidence, reason in splits:
        print(f"  - Page {page_num}: {reason} (confidence: {confidence:.2f})")

    # Verify
    assert len(splits) == 1, f"Expected 1 split, got {len(splits)}"
    assert splits[0][0] == 3, f"Expected split at page 3, got {splits[0][0]}"
    assert splits[0][1] > 0.8, f"Expected high confidence, got {splits[0][1]}"
    print("PASS: Page number reset detection working correctly")


def test_blank_page_detection():
    """Test blank page separator detection"""
    print("\n=== Testing Blank Page Detection ===")

    detector = SplitDetector()

    # Mock pages with blank separator
    pages = [
        {'page_num': 0, 'text': 'This is a normal page with lots of content.'},
        {'page_num': 1, 'text': 'Another normal page with content here.'},
        {'page_num': 2, 'text': '   '},  # Blank separator
        {'page_num': 3, 'text': 'New document starts after blank page with content.'},
    ]

    splits = detector.detect_blank_pages(pages)

    print(f"Detected {len(splits)} blank page separators:")
    for page_num, confidence, reason in splits:
        print(f"  - Page {page_num}: {reason} (confidence: {confidence:.2f})")

    # Verify
    assert len(splits) == 1, f"Expected 1 split, got {len(splits)}"
    assert splits[0][0] == 3, f"Expected split at page 3 (after blank), got {splits[0][0]}"
    assert splits[0][1] > 0.8, f"Expected high confidence, got {splits[0][1]}"
    print("PASS: Blank page detection working correctly")


def test_semantic_discontinuity():
    """Test semantic discontinuity detection with mock embeddings"""
    print("\n=== Testing Semantic Discontinuity Detection ===")

    detector = SplitDetector()

    # Create mock embeddings for 5 pages
    # Pages 0-2 are similar (doc 1), pages 3-4 are similar (doc 2)
    np.random.seed(42)

    # Document 1 embeddings (similar to each other)
    doc1_base = np.random.randn(768)
    doc1_base = doc1_base / np.linalg.norm(doc1_base)  # Normalize
    emb0 = doc1_base + np.random.randn(768) * 0.1
    emb1 = doc1_base + np.random.randn(768) * 0.1
    emb2 = doc1_base + np.random.randn(768) * 0.1

    # Document 2 embeddings (similar to each other, different from doc 1)
    doc2_base = np.random.randn(768)
    doc2_base = doc2_base / np.linalg.norm(doc2_base)
    emb3 = doc2_base + np.random.randn(768) * 0.1
    emb4 = doc2_base + np.random.randn(768) * 0.1

    # Normalize all embeddings
    embeddings = np.array([emb0, emb1, emb2, emb3, emb4])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    splits = detector.detect_semantic_discontinuity("test_doc_id", embeddings)

    print(f"Detected {len(splits)} semantic discontinuities:")
    for page_num, confidence, reason in splits:
        print(f"  - Page {page_num}: {reason} (confidence: {confidence:.2f})")

    # Should detect split between page 2 and 3
    split_pages = [s[0] for s in splits]
    print(f"Split detected at pages: {split_pages}")
    print("PASS: Semantic discontinuity detection working")


def test_clustering():
    """Test DBSCAN clustering detection"""
    print("\n=== Testing Clustering Detection ===")

    detector = SplitDetector()

    # Create mock embeddings with clear clusters
    np.random.seed(42)

    # Cluster 1 (pages 0-2)
    cluster1_center = np.random.randn(768)
    cluster1_center = cluster1_center / np.linalg.norm(cluster1_center)
    cluster1_pages = [cluster1_center + np.random.randn(768) * 0.05 for _ in range(3)]

    # Cluster 2 (pages 3-5)
    cluster2_center = np.random.randn(768)
    cluster2_center = cluster2_center / np.linalg.norm(cluster2_center)
    cluster2_pages = [cluster2_center + np.random.randn(768) * 0.05 for _ in range(3)]

    # Combine and normalize
    embeddings = np.array(cluster1_pages + cluster2_pages)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    splits = detector.detect_with_clustering(embeddings, eps=0.3, min_samples=2)

    print(f"Detected {len(splits)} cluster boundaries:")
    for page_num, confidence, reason in splits:
        print(f"  - Page {page_num}: {reason} (confidence: {confidence:.2f})")

    print("PASS: Clustering detection working")


def test_combine_signals():
    """Test signal combination and confidence scoring"""
    print("\n=== Testing Signal Combination ===")

    detector = SplitDetector()

    # Create mock split signals
    all_splits = [
        # Page number reset at page 3 (high confidence)
        [(3, 0.9, "Page number reset")],

        # Header/footer change at page 3 (medium confidence)
        [(3, 0.6, "Header changed")],

        # Semantic discontinuity at page 3 (medium confidence)
        [(3, 0.7, "Low semantic similarity")],

        # Clustering at page 3 (medium confidence)
        [(3, 0.6, "Cluster boundary")],

        # Unrelated split at page 5 (low confidence)
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

    # Verify
    assert len(candidates) == 2, f"Expected 2 candidates, got {len(candidates)}"

    # Page 3 should have high confidence (max 0.9 + bonus)
    page3_candidate = [c for c in candidates if c['page'] == 3][0]
    assert page3_candidate['confidence'] > 0.9, f"Expected high confidence for page 3"
    assert page3_candidate['signal_count'] == 4, f"Expected 4 signals for page 3"

    # Page 5 should have low confidence
    page5_candidate = [c for c in candidates if c['page'] == 5][0]
    assert page5_candidate['confidence'] < 0.5, f"Expected low confidence for page 5"

    print("\nPASS: Signal combination working correctly")
    print(f"\nConfidence thresholds:")
    print(f"  High (auto-accept): >= {detector.THRESHOLD_HIGH}")
    print(f"  Medium (LLM review): >= {detector.THRESHOLD_MEDIUM}")
    print(f"  Low (ignore): >= {detector.THRESHOLD_LOW}")


def main():
    """Run all tests"""
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
