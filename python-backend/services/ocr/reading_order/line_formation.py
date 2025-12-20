"""
Layer 2.1: Line Formation

Groups words into horizontal lines using vertical proximity clustering.

This is the CRITICAL layer that fixes text ordering issues.
Uses vertical overlap + baseline alignment + horizontal gap,
all relative to font height, not fixed pixel thresholds.
"""

import logging
from typing import List
from .data_structures import WordBox, Line
from .config import ReadingOrderConfig
from .utils import calculate_vertical_overlap, get_average_word_height

logger = logging.getLogger(__name__)



def estimate_skew_angle(word_boxes: List[WordBox]) -> float:
    """
    Estimate document skew angle from word positions using linear regression.
    
    For a skewed document, words on the same visual line will have increasing
    y-coordinates as x increases (or vice versa). We estimate this slope.
    
    Returns:
        Skew angle in radians (positive = clockwise rotation)
    """
    if len(word_boxes) < 10:
        return 0.0
    
    import numpy as np
    
    # Get word center coordinates
    x_coords = np.array([w.center_x for w in word_boxes])
    y_coords = np.array([w.center_y for w in word_boxes])
    
    # Use robust linear regression (median of local slopes)
    # Group words by approximate y position
    sorted_by_y = sorted(word_boxes, key=lambda w: w.center_y)
    
    slopes = []
    window_size = max(3, len(word_boxes) // 20)  # ~5% of words per window
    
    for i in range(0, len(sorted_by_y) - window_size, window_size // 2):
        window = sorted_by_y[i:i + window_size]
        # Sort by x within window
        window_sorted = sorted(window, key=lambda w: w.center_x)
        
        if len(window_sorted) >= 2:
            # Calculate slope
            x1, y1 = window_sorted[0].center_x, window_sorted[0].center_y
            x2, y2 = window_sorted[-1].center_x, window_sorted[-1].center_y
            
            if abs(x2 - x1) > 50:  # Minimum horizontal span
                slope = (y2 - y1) / (x2 - x1)
                slopes.append(slope)
    
    if not slopes:
        return 0.0
    
    # Use median slope
    median_slope = float(np.median(slopes))
    
    # Convert to angle
    import math
    angle = math.atan(median_slope)
    
    logger.debug(f"Estimated skew angle: {math.degrees(angle):.2f} degrees")
    
    return angle



def _split_overlapping_clusters(cluster_words: List[WordBox], avg_height: float, page_width: float) -> List[List[WordBox]]:
    """
    Split a cluster if it contains words that overlap horizontally.
    
    When two full-width text blocks from adjacent lines get merged into one cluster,
    they need to be split back into separate lines based on y-coordinate.
    
    Detection criteria:
    - Words are "wide" if they span > 40% of page width
    - Two wide words "overlap" if their x-ranges intersect significantly (> 50%)
    - If overlapping wide words exist, split by y-coordinate
    
    Args:
        cluster_words: Words in a single cluster
        avg_height: Average word height
        page_width: Page width in pixels
        
    Returns:
        List of word groups (each becomes a separate line)
    """
    if len(cluster_words) <= 1:
        return [cluster_words]
    
    # Check for wide words (spanning significant page width)
    wide_threshold = page_width * 0.35  # Word is "wide" if > 35% of page width
    wide_words = [(i, w) for i, w in enumerate(cluster_words) if (w.x1 - w.x0) > wide_threshold]
    
    if len(wide_words) < 2:
        # No overlapping wide words possible
        return [cluster_words]
    
    # Check for significant x-overlap between wide words
    has_overlap = False
    for i in range(len(wide_words)):
        for j in range(i + 1, len(wide_words)):
            w1, w2 = wide_words[i][1], wide_words[j][1]
            # Calculate overlap ratio
            overlap_start = max(w1.x0, w2.x0)
            overlap_end = min(w1.x1, w2.x1)
            if overlap_end > overlap_start:
                overlap_width = overlap_end - overlap_start
                min_width = min(w1.x1 - w1.x0, w2.x1 - w2.x0)
                overlap_ratio = overlap_width / min_width if min_width > 0 else 0
                if overlap_ratio > 0.5:
                    has_overlap = True
                    break
        if has_overlap:
            break
    
    if not has_overlap:
        return [cluster_words]
    
    # Split by y-coordinate - sort by center_y and find natural gaps
    sorted_words = sorted(cluster_words, key=lambda w: w.center_y)
    
    # Find gaps between consecutive words
    groups = [[sorted_words[0]]]
    gap_threshold = avg_height * 0.5  # Split if gap > 50% of avg height
    
    for i in range(1, len(sorted_words)):
        prev_word = sorted_words[i - 1]
        curr_word = sorted_words[i]
        gap = curr_word.center_y - prev_word.center_y
        
        if gap > gap_threshold:
            # Start new group
            groups.append([curr_word])
        else:
            # Add to current group
            groups[-1].append(curr_word)
    
    if len(groups) > 1:
        logger.debug(f"Split cluster with {len(cluster_words)} overlapping words into {len(groups)} groups")
    
    return groups


def deskew_coordinates(word_boxes: List[WordBox], angle: float) -> None:
    """
    Adjust word box y-coordinates to compensate for skew.
    
    This doesn't actually rotate the boxes, just adjusts y0/y1
    to represent the "deskewed" positions for sorting purposes.
    
    Args:
        word_boxes: List of WordBox to modify in-place
        angle: Skew angle in radians
    """
    import math
    
    if abs(angle) < 0.001:  # Less than 0.06 degrees
        return
    
    # Find page center x
    all_x = [w.center_x for w in word_boxes]
    center_x = (min(all_x) + max(all_x)) / 2
    
    tan_angle = math.tan(angle)
    
    for word in word_boxes:
        # Adjust y coordinates based on x position relative to center
        dx = word.center_x - center_x
        dy = -dx * tan_angle  # Counter-rotate
        
        word.y0 += dy
        word.y1 += dy
        word.center_y = (word.y0 + word.y1) / 2  # Update center_y after deskewing!


def form_lines_clustered(word_boxes: List[WordBox],
                         page_height: float,
                         config: ReadingOrderConfig,
                         y_variance_ratio: float = 0.0,
                         page_width: float = 0.0) -> List[Line]:
    """
    Alternative line formation using y-coordinate clustering.
    
    Better for skewed documents with tight line spacing where the
    greedy approach fails.
    
    Uses adaptive thresholds based on document characteristics:
    - Estimates skew angle and compensates
    - Measures line spacing to set appropriate cluster threshold
    - Boosts threshold for curved documents (high y_variance_ratio)
    
    Args:
        word_boxes: List of WordBox instances
        page_height: Page height in pixels
        config: ReadingOrderConfig
        y_variance_ratio: Detected y-variance ratio (0.0 = no variance, >0.5 = curved)
    """
    if not word_boxes:
        return []
    
    import numpy as np
    from scipy.cluster.hierarchy import fclusterdata
    
    logger.info(f"Forming lines (clustered) from {len(word_boxes)} words, y_variance_ratio={y_variance_ratio:.2f}")

    # Apply skew compensation first
    skew_angle = estimate_skew_angle(word_boxes)
    if abs(skew_angle) > 0.001:
        import math
        logger.info(f"Compensating for skew: {math.degrees(skew_angle):.2f} degrees")
        deskew_coordinates(word_boxes, skew_angle)

    # Calculate average word height for threshold
    avg_height = get_average_word_height(word_boxes)
    if avg_height == 0:
        return []

    # Get y-centers for clustering (now deskewed)
    y_centers = np.array([w.center_y for w in word_boxes])
    
    # ADAPTIVE THRESHOLD: Analyze document to determine optimal clustering
    # 
    # Strategy: Find the natural gaps between lines by looking at
    # the distribution of y-coordinate differences
    
    sorted_y = np.sort(y_centers)
    y_diffs = np.diff(sorted_y)
    
    if len(y_diffs) > 10:
        # Simple adaptive approach:
        # 1. Filter out very small gaps (within-word overlaps)
        # 2. Use median of remaining gaps as threshold
        
        significant_gaps = y_diffs[y_diffs > avg_height * 0.3]
        
        if len(significant_gaps) > 5:
            # Use a balanced threshold - between 55th and 65th percentile
            p54 = np.percentile(significant_gaps, 54)
            p62 = np.percentile(significant_gaps, 62)
            cluster_threshold = (p54 + p62) / 2
        else:
            cluster_threshold = avg_height * 1.0
        
        # BOOST THRESHOLD FOR CURVED DOCUMENTS
        # When y_variance_ratio is high, words on the same visual line have
        # significant y-coordinate differences due to book spine curvature.
        # We need a more generous threshold to keep them together.
        #
        # Key insight: For curved book pages:
        # - Intra-line y-spread: ~40px (what we need to accommodate)
        # - Inter-line y-gap: ~60-80px (what we must NOT merge)
        # So we need a threshold around 50-60px, not 100px+
        if y_variance_ratio > 0.5:
            # Conservative boost: minimal adjustment to prevent merging adjacent lines
            # Scale boost: 0.5 -> 1.075x, 1.0 -> 1.15x (very conservative)
            curvature_boost = 1.0 + y_variance_ratio * 0.15
            cluster_threshold = cluster_threshold * curvature_boost
            logger.info(f"Boosting threshold for curved document: {curvature_boost:.2f}x (y_variance={y_variance_ratio:.2f})")

        # Clamp to reasonable range
        # For curved documents, use a higher MAX but still conservative
        max_multiplier = 2.5 if y_variance_ratio <= 0.5 else 3.0
        cluster_threshold = max(avg_height * 0.6, min(cluster_threshold, avg_height * max_multiplier))
        
        logger.debug(f"Adaptive threshold: {cluster_threshold:.1f}, avg_height={avg_height:.1f}")
    else:
        # Fall back to default for small documents
        cluster_threshold = avg_height * 1.5
    
    # Reshape for clustering
    y_centers_2d = y_centers.reshape(-1, 1)
    
    # Log the final threshold being used
    logger.info(f"CLUSTERING: threshold={cluster_threshold:.1f}px, avg_height={avg_height:.1f}px, words={len(word_boxes)}")

    try:
        # Hierarchical clustering
        clusters = fclusterdata(
            y_centers_2d,
            t=cluster_threshold,
            criterion='distance',
            method='complete'  # Use max distance between cluster members
        )

        # Log clustering results
        num_clusters = len(set(clusters))
        logger.info(f"CLUSTERING: Formed {num_clusters} clusters from {len(word_boxes)} words")

        # Debug: Find target words and their cluster assignments
        target_texts = ['lives', 'helps', 'remove', 'burdens', 'excite', 'hopes', 'increase', 'faith']
        target_clusters = {}
        for i, wb in enumerate(word_boxes):
            for target in target_texts:
                if target.lower() in wb.text.lower():
                    cluster_id = clusters[i]
                    if cluster_id not in target_clusters:
                        target_clusters[cluster_id] = []
                    target_clusters[cluster_id].append((wb.text, wb.center_y))
                    break

        if target_clusters:
            logger.debug(f"TARGET WORDS: Found in {len(target_clusters)} cluster(s)")
            for cid, words in target_clusters.items():
                word_str = ", ".join([f"'{w[0]}'({w[1]:.1f})" for w in words])
                logger.debug(f"  Cluster {cid}: {word_str}")

    except Exception as e:
        logger.warning(f"Clustering failed: {e}, falling back to greedy")
        return form_lines(word_boxes, page_height, config)
    
    # Group words by cluster
    cluster_dict = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(word_boxes[i])
    
    # Sort clusters by average y-position (top to bottom)
    sorted_clusters = sorted(
        cluster_dict.items(),
        key=lambda x: np.mean([w.center_y for w in x[1]])
    )
    
    # Create lines from clusters, splitting any with overlapping words
    lines = []
    line_id = 0
    for cluster_id, words in sorted_clusters:
        # Check for and split clusters with overlapping full-width words
        # This fixes the bug where adjacent lines get merged due to aggressive clustering
        word_groups = _split_overlapping_clusters(words, avg_height, page_width if page_width > 0 else 2630)
        
        for group in word_groups:
            # Sort words within group by x-position (left to right)
            sorted_words = sorted(group, key=lambda w: w.x0)
            
            # Create line
            line = _create_line(sorted_words, line_id)
            lines.append(line)
            line_id += 1
    
    logger.info(f"Formed {len(lines)} lines using adaptive clustering")
    
    return lines


def form_lines_sequential(word_boxes: List[WordBox],
                          page_height: float,
                          config: ReadingOrderConfig) -> List[Line]:
    """
    Form lines using horizontal-continuity-first approach.

    For curved/skewed documents, words on the same visual line may have
    different y-coordinates. This algorithm:
    1. Sorts words by x0 (left to right)
    2. Groups horizontally-adjacent words
    3. Sorts groups by average y to get top-to-bottom order
    4. Merges groups that are close in y (same visual line wrapping)

    This handles:
    - Book spine curvature (left side lower, right side higher)
    - Skewed documents
    - Single-column wrapped text
    """
    if not word_boxes:
        return []

    import numpy as np

    logger.info(f"Forming lines (sequential) from {len(word_boxes)} words")

    # Calculate average word height for thresholds
    avg_height = get_average_word_height(word_boxes)
    if avg_height == 0:
        return []

    # Apply skew compensation
    skew_angle = estimate_skew_angle(word_boxes)
    if abs(skew_angle) > 0.001:
        import math
        logger.info(f"Compensating for skew: {math.degrees(skew_angle):.2f} degrees")
        deskew_coordinates(word_boxes, skew_angle)

    # Y-tolerance for line membership (after deskew)
    y_tolerance = avg_height * config.line_y_spread_multiplier

    # STEP 1: Sort all words by x0 (left to right)
    x_sorted_words = sorted(word_boxes, key=lambda w: w.x0)

    # STEP 2: Group horizontally-adjacent words into "segments"
    # A segment is a run of words that are horizontally continuous
    segments = []
    current_segment = [x_sorted_words[0]]

    for i in range(1, len(x_sorted_words)):
        word = x_sorted_words[i]
        prev_word = current_segment[-1]

        # Check if horizontally continuous (gap < 3x avg_height)
        gap = word.x0 - prev_word.x1
        is_continuous = gap < avg_height * 3.0

        # Also check vertical proximity (should be roughly on same line)
        y_close = abs(word.center_y - prev_word.center_y) < y_tolerance * 2

        if is_continuous and y_close:
            current_segment.append(word)
        else:
            segments.append(current_segment)
            current_segment = [word]

    segments.append(current_segment)  # Don't forget last segment

    logger.debug(f"Created {len(segments)} horizontal segments")

    # STEP 3: Calculate average y and x-range for each segment
    segment_info = []
    for seg in segments:
        avg_y = sum(w.center_y for w in seg) / len(seg)
        min_x = min(w.x0 for w in seg)
        max_x = max(w.x1 for w in seg)
        segment_info.append({
            'words': seg,
            'avg_y': avg_y,
            'min_x': min_x,
            'max_x': max_x
        })

    # STEP 4: Sort segments by average y (top to bottom after deskew)
    segment_info.sort(key=lambda s: s['avg_y'])

    # STEP 5: Merge segments that are on the same visual line
    # Two segments can be merged if:
    # - Their y-values are close (within tolerance)
    # - They don't horizontally overlap much
    # - They're NOT both "full-width" segments (which would be separate lines)
    lines = []
    line_id_counter = 0

    # Calculate page width from word positions for width threshold
    all_x0 = [w.x0 for w in word_boxes]
    all_x1 = [w.x1 for w in word_boxes]
    page_left = min(all_x0)
    page_right = max(all_x1)
    page_width = page_right - page_left

    # A segment is "wide" if it spans > 40% of page width
    wide_threshold = page_width * 0.4

    merged_indices = set()
    for i, seg_i in enumerate(segment_info):
        if i in merged_indices:
            continue

        # Start a line with this segment
        line_words = list(seg_i['words'])
        current_y = seg_i['avg_y']
        seg_i_width = seg_i['max_x'] - seg_i['min_x']

        # Track combined x-range for the growing line
        line_min_x = seg_i['min_x']
        line_max_x = seg_i['max_x']

        # Look for other segments that could be on the same visual line
        for j in range(i + 1, len(segment_info)):
            if j in merged_indices:
                continue

            seg_j = segment_info[j]
            seg_j_width = seg_j['max_x'] - seg_j['min_x']

            # Check if y is close enough
            if abs(seg_j['avg_y'] - current_y) > y_tolerance:
                # Too far in y - since sorted by y, no more matches
                break

            # Check if both segments are "wide" - if so, they're separate lines
            if seg_i_width > wide_threshold and seg_j_width > wide_threshold:
                continue  # Don't merge two wide segments

            # Check horizontal relationship - should not overlap much
            overlap = min(line_max_x, seg_j['max_x']) - max(line_min_x, seg_j['min_x'])

            if overlap < avg_height:  # Allow small overlap
                # Check combined width wouldn't be too wide for a single line
                combined_min = min(line_min_x, seg_j['min_x'])
                combined_max = max(line_max_x, seg_j['max_x'])
                combined_width = combined_max - combined_min

                # Only merge if combined width is reasonable (< 95% of page width per segment)
                # This prevents merging segments from adjacent lines that together span the page
                if combined_width < page_width * 0.95 or len(line_words) <= 3:
                    # Merge this segment
                    line_words.extend(seg_j['words'])
                    merged_indices.add(j)
                    # Update line's x-range
                    line_min_x = combined_min
                    line_max_x = combined_max

        # Create line from collected words (sort by x)
        sorted_line_words = sorted(line_words, key=lambda w: w.x0)
        line = _create_line(sorted_line_words, line_id_counter)
        lines.append(line)
        line_id_counter += 1

    logger.info(f"Formed {len(lines)} lines using sequential sweep")
    
    return lines


def post_process_lines(lines: List[Line], avg_height: float, y_variance_ratio: float = 0.0) -> List[Line]:
    """
    Post-process lines to fix common issues.

    1. Verify horizontal ordering within each line
    2. Split over-long lines at large gaps (DISABLED for curved documents)
    3. Validate punctuation positioning

    Args:
        lines: List of Line objects
        avg_height: Average word height for threshold calculations
        y_variance_ratio: Y-variance ratio (>0.5 = curved document, disables line splitting)

    Returns:
        Corrected list of Line objects
    """
    if not lines:
        return lines

    # For curved documents (high y-variance), DON'T split lines
    # The clustering already correctly grouped words on the same visual line
    # Splitting would break up correctly-formed lines
    disable_line_splitting = y_variance_ratio > 0.5
    if disable_line_splitting:
        logger.info(f"post_process_lines: Disabling line splitting for curved document (y_variance={y_variance_ratio:.2f})")

    corrected_lines = []
    line_id = 0

    for line in lines:
        # Step 1: Ensure words are sorted by x-position
        sorted_words = sorted(line.words, key=lambda w: w.x0)

        # Step 2: Check for large gaps that might indicate merged lines
        # SKIP this step for curved documents
        if len(sorted_words) > 1 and not disable_line_splitting:
            gaps = []
            for i in range(len(sorted_words) - 1):
                gap = sorted_words[i + 1].x0 - sorted_words[i].x1
                gaps.append((i, gap))

            # Calculate average gap
            avg_gap = sum(g[1] for g in gaps) / len(gaps) if gaps else 0

            # Find split points: gaps that are much larger than average
            # or the line has too many words
            split_points = []
            if avg_gap > 0:
                for i, gap in gaps:
                    # Split conditions:
                    # 1. Gap is 2.5x average AND line is long (>8 words)
                    # 2. Gap is 2x average AND line is very long (>15 words)
                    # 3. Gap is very large (>1.5x word height)
                    is_long_line = len(sorted_words) > 8
                    is_very_long_line = len(sorted_words) > 15

                    should_split = (
                        (gap > avg_gap * 2.5 and is_long_line) or
                        (gap > avg_gap * 2.0 and is_very_long_line) or
                        (gap > avg_height * 1.5 and len(sorted_words) > 6)
                    )

                    if should_split:
                        split_points.append(i + 1)
            
            # Split line at large gaps
            if split_points:
                prev_idx = 0
                for split_idx in split_points:
                    segment_words = sorted_words[prev_idx:split_idx]
                    if segment_words:
                        new_line = _create_line(segment_words, line_id)
                        corrected_lines.append(new_line)
                        line_id += 1
                    prev_idx = split_idx
                
                # Don't forget the last segment
                if prev_idx < len(sorted_words):
                    segment_words = sorted_words[prev_idx:]
                    new_line = _create_line(segment_words, line_id)
                    corrected_lines.append(new_line)
                    line_id += 1
            else:
                # No splits needed, just use sorted words
                new_line = _create_line(sorted_words, line_id)
                corrected_lines.append(new_line)
                line_id += 1
        else:
            # Single word line OR line splitting disabled
            new_line = _create_line(sorted_words, line_id)
            corrected_lines.append(new_line)
            line_id += 1

    if len(corrected_lines) != len(lines):
        logger.info(f"Post-processing: split {len(lines)} lines into {len(corrected_lines)}")
    
    return corrected_lines

def form_lines(word_boxes: List[WordBox],
              page_height: float,
              config: ReadingOrderConfig) -> List[Line]:
    """
    Group words into lines using sophisticated vertical proximity clustering.

    Key Insight: Words on the same line have:
    1. Significant vertical overlap
    2. Similar baseline positions
    3. Reasonable horizontal spacing

    Args:
        word_boxes: List of WordBox instances
        page_height: Page height for context
        config: Configuration with thresholds

    Returns:
        List of Line instances, each containing words on same line
    """
    if not word_boxes:
        logger.warning("No words to form lines from")
        return []

    logger.info(f"Forming lines from {len(word_boxes)} words")

    # Estimate and compensate for document skew
    skew_angle = estimate_skew_angle(word_boxes)
    if abs(skew_angle) > 0.001:  # More than 0.06 degrees
        logger.info(f"Detected document skew, compensating...")
        deskew_coordinates(word_boxes, skew_angle)

    # Sort words: top to bottom, then left to right
    sorted_words = sorted(word_boxes, key=lambda w: (w.y0, w.x0))

    # Calculate average font height for adaptive thresholds
    avg_height = get_average_word_height(sorted_words)
    if avg_height == 0:
        logger.error("Average word height is 0, cannot form lines")
        return []

    logger.debug(f"Average word height: {avg_height:.1f}px")

    # Initialize with first word
    lines = []
    current_line_words = [sorted_words[0]]
    line_id_counter = 0

    # Process remaining words
    for i in range(1, len(sorted_words)):
        current_word = sorted_words[i]
        previous_word = current_line_words[-1]

        # Calculate metrics for decision
        same_line = _is_same_line(
            previous_word,
            current_word,
            current_line_words[0],  # First word in current line
            avg_height,
            config,
            current_line_words  # Pass all words for slope calculation
        )

        if same_line:
            # Add to current line
            current_line_words.append(current_word)
        else:
            # Finalize current line
            line = _create_line(current_line_words, line_id_counter)
            lines.append(line)
            line_id_counter += 1

            # Start new line
            current_line_words = [current_word]

    # Don't forget last line
    if current_line_words:
        line = _create_line(current_line_words, line_id_counter)
        lines.append(line)

    logger.info(f"Formed {len(lines)} lines from {len(word_boxes)} words")

    # Log line statistics
    words_per_line = [len(line.words) for line in lines]
    avg_words = sum(words_per_line) / len(words_per_line) if words_per_line else 0
    logger.debug(f"Average words per line: {avg_words:.1f} "
                f"(min={min(words_per_line)}, max={max(words_per_line)})")

    return lines


def _is_same_line(previous_word: WordBox,
                  current_word: WordBox,
                  line_start_word: WordBox,
                  avg_height: float,
                  config: ReadingOrderConfig,
                  current_line_words: List[WordBox] = None) -> bool:
    """
    Determine if current word belongs to same line as previous word.

    Uses slope-aware detection for skewed documents:
    1. Calculate line slope from existing words
    2. Predict expected y-position for new word
    3. Check if actual y is close to expected

    Args:
        previous_word: Last word in current line
        current_word: Word being evaluated
        line_start_word: First word in current line
        avg_height: Average word height for threshold calculation
        config: Configuration
        current_line_words: All words in current line (for slope calculation)

    Returns:
        True if same line, False if new line
    """
    # Calculate basic metrics
    vertical_overlap = calculate_vertical_overlap(previous_word, current_word)
    vertical_distance = abs(current_word.center_y - previous_word.center_y)
    baseline_diff = abs(current_word.baseline - previous_word.baseline)
    horizontal_gap = current_word.x0 - previous_word.x1

    # Calculate dynamic thresholds based on average height
    vertical_threshold = avg_height * config.line_vertical_threshold_multiplier
    baseline_threshold = avg_height * config.line_baseline_tolerance_multiplier
    max_horizontal_gap = avg_height * config.line_horizontal_gap_multiplier

    # Slope-aware y-spread check
    # If we have multiple words in the line, calculate slope and predict expected y
    if current_line_words and len(current_line_words) >= 2:
        # Calculate line slope from first and last word
        dx = current_line_words[-1].center_x - current_line_words[0].center_x
        dy = current_line_words[-1].center_y - current_line_words[0].center_y

        if abs(dx) > 10:  # Avoid division by zero
            slope = dy / dx
            # Predict expected y for current word based on slope
            expected_y = line_start_word.center_y + slope * (current_word.center_x - line_start_word.center_x)
            # Check if actual y is close to expected
            y_deviation = abs(current_word.center_y - expected_y)
            max_y_deviation = avg_height * config.line_y_spread_multiplier
            within_line_y_spread = y_deviation < max_y_deviation
        else:
            # Not enough horizontal spread to calculate slope
            distance_from_line_start = abs(current_word.center_y - line_start_word.center_y)
            max_line_y_spread = avg_height * config.line_y_spread_multiplier
            within_line_y_spread = distance_from_line_start < max_line_y_spread
    else:
        # Only one word in line, use distance from line start
        distance_from_line_start = abs(current_word.center_y - line_start_word.center_y)
        max_line_y_spread = avg_height * config.line_y_spread_multiplier
        within_line_y_spread = distance_from_line_start < max_line_y_spread

    # Decision logic: ALL criteria must be satisfied
    has_overlap_or_close = (
        vertical_overlap > config.line_vertical_overlap_threshold or
        vertical_distance < vertical_threshold
    )
    has_similar_baseline = baseline_diff < baseline_threshold
    has_reasonable_gap = horizontal_gap < max_horizontal_gap

    same_line = (
        has_overlap_or_close and
        has_similar_baseline and
        has_reasonable_gap and
        within_line_y_spread
    )

    return same_line


def _create_line(words: List[WordBox], line_id: int) -> Line:
    """
    Create a Line object from words.

    Args:
        words: List of WordBox instances
        line_id: Unique identifier for this line

    Returns:
        Line instance with sorted words and calculated bounding box
    """
    # Sort words left to right
    sorted_words = sorted(words, key=lambda w: w.x0)

    # Create Line object (bounding box calculated in __post_init__)
    line = Line(words=sorted_words, line_id=line_id)

    # Assign line_id to constituent words
    for word in sorted_words:
        word.line_id = line_id

    return line
