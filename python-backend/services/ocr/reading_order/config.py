"""
Configuration for reading order detection.

All tunable parameters in one place for easy adjustment.
"""

from dataclasses import dataclass


@dataclass
class ReadingOrderConfig:
    """
    Configuration for reading order detection system.

    All parameters are tunable and can be adjusted based on document type
    and quality. Default values are optimized for general documents.
    """

    # ==== Layer 1: Input Normalization ====

    # Minimum OCR confidence to include a word (0.0-1.0)
    confidence_threshold: float = 0.6

    # Minimum word dimensions (pixels)
    min_word_width: float = 5.0
    min_word_height: float = 5.0

    # Maximum word width (as fraction of page width) - likely detection errors
    max_word_width_ratio: float = 0.9

    # ==== Layer 2: Line Formation ====

    # Minimum vertical overlap required for same line (0.0-1.0)
    # 0.5 means 50% of smaller word's height must overlap
    line_vertical_overlap_threshold: float = 0.5

    # Baseline tolerance as multiplier of average word height
    # Words with baselines within this tolerance are considered on same line
    # 0.6 allows more baseline variance (handles photo captures better)
    line_baseline_tolerance_multiplier: float = 0.6

    # Maximum horizontal gap between words on same line
    # Expressed as multiplier of average word height
    line_horizontal_gap_multiplier: float = 3.0

    # Vertical distance threshold for line formation
    # Expressed as multiplier of average word height
    line_vertical_threshold_multiplier: float = 0.5

    # Maximum Y-spread within a line (for skewed documents)
    # Expressed as multiplier of average word height
    # Higher values allow more vertical variation (for photos/skewed scans)
    # 1.5 allows up to 1.5x word height variation (handles photo captures better)
    line_y_spread_multiplier: float = 1.5

    # Use clustering-based line formation (better for skewed docs)
    # When False, uses faster greedy approach that works for well-aligned docs
    use_clustered_line_formation: bool = False

    # Use sequential scanning (best for skewed documents)
    use_sequential_line_formation: bool = False

    # AUTO-DETECT SKEW: If True, automatically switch to clustered line formation
    # when document skew is detected (skew angle > skew_detection_threshold)
    auto_detect_skew: bool = True

    # Threshold for skew detection (in degrees)
    # Documents with estimated skew > this value use clustered line formation
    skew_detection_threshold: float = 1.5  # Raised from 0.5 - less sensitive to photo alignment

    # Enable post-processing to fix line issues
    enable_post_processing: bool = True

    # ==== Layer 2: Block Formation ====

    # Vertical gap threshold for new block
    # Expressed as multiplier of average line height
    block_vertical_gap_multiplier: float = 2.0

    # Minimum horizontal overlap for lines in same block (0.0-1.0)
    block_horizontal_overlap_threshold: float = 0.3

    # ==== Layer 3: Layout Detection ====

    # Threshold for single-column detection
    # If this fraction of words are in center third, it's single-column
    single_column_center_ratio: float = 0.60  # Lowered from 0.7 - more lenient for curved documents

    # Threshold for multi-column detection
    # If this fraction of words are on left AND right, it's multi-column
    multi_column_side_ratio: float = 0.3

    # Minimum rows/columns for table detection
    table_min_rows: int = 3
    table_min_columns: int = 3

    # Fraction of rows that must be multi-column to be considered a table
    table_row_threshold: float = 0.7

    # ==== Layer 3: Column Detection (Projection Histogram) ====

    # Column separation threshold (fraction of page width)
    # Horizontal gap > this * page_width indicates column boundary
    column_separation_threshold: float = 0.15

    # Sigma for Gaussian smoothing of histogram
    histogram_smooth_sigma: float = 5.0

    # Valley density threshold (fraction of mean density)
    # Regions with density < this * mean are considered valleys (gutters)
    valley_density_threshold: float = 0.3

    # Minimum valley width (pixels) to be considered a column gutter
    min_valley_width: float = 20.0

    # Minimum distance between valleys (as fraction of page width)
    # Valleys closer than this are merged (prevents false column splits)
    min_inter_valley_ratio: float = 0.15

    # ==== Layer 3: Special Zones (Headers/Footers) ====

    # Header zone (top N% of page)
    header_zone_ratio: float = 0.12

    # Footer zone (bottom N% of page, as fraction from top)
    footer_zone_ratio: float = 0.88

    # ==== Layer 4: Reading Order Assignment ====

    # Threshold for detecting true two-column layout vs stacked regions
    # Vertical overlap ratio > this means true side-by-side columns
    two_column_overlap_threshold: float = 0.5

    # Table cell horizontal gap multiplier (for detecting column boundaries in tables)
    table_horizontal_gap_multiplier: float = 2.0

    # ==== Layer 5: Text Output ====

    # Whether to add structure markers (e.g., "---COLUMN END---")
    preserve_structure_markers: bool = False

    # Whether to normalize whitespace in output
    normalize_whitespace: bool = True

    # Maximum consecutive newlines in output
    max_consecutive_newlines: int = 2

    # ==== Debugging ====

    # Enable debug logging
    debug: bool = False

    # Enable visualization (requires image)
    enable_visualization: bool = False

    # Output path for debug visualizations
    visualization_output_path: str = "debug_reading_order.png"


    @classmethod
    def for_scanned_document(cls) -> 'ReadingOrderConfig':
        """
        Preset for clean scanned documents with minimal skew.
        
        Use for: High-quality scans, digital PDFs, well-aligned documents.
        """
        return cls()  # Default settings work well for clean scans

    @classmethod
    def for_photographed_document(cls) -> 'ReadingOrderConfig':
        """
        Preset for photographed documents with skew and perspective distortion.
        
        Use for: Phone photos of documents, angled captures, documents on tables.
        
        Adjustments:
        - Higher tolerance for vertical variation (skew compensation)
        - More lenient baseline matching
        - Wider horizontal gap allowance
        """
        return cls(
            line_vertical_overlap_threshold=0.4,
            line_baseline_tolerance_multiplier=1.0,
            line_horizontal_gap_multiplier=4.0,
            line_vertical_threshold_multiplier=1.0,
            line_y_spread_multiplier=1.5,
            use_sequential_line_formation=True,  # Best for skewed docs
        )

    @classmethod
    def for_multi_column(cls) -> 'ReadingOrderConfig':
        """
        Preset for multi-column documents like newspapers or magazines.

        Use for: Newspapers, magazines, academic papers with columns.

        Adjustments:
        - MORE PERMISSIVE line formation (column detection handles separation)
        - Strict column detection via projection histogram
        - Minimum inter-valley distance to avoid false columns
        """
        return cls(
            # MORE PERMISSIVE line formation (column detection handles separation)
            line_horizontal_gap_multiplier=4.0,      # Was 2.0 - allow wider gaps within columns
            line_baseline_tolerance_multiplier=0.5,  # Was 0.3 - more lenient baseline matching
            line_y_spread_multiplier=1.0,            # Was 0.6 - allow more vertical variation

            # Keep strict column detection (these work well)
            column_separation_threshold=0.10,
            multi_column_side_ratio=0.25,
            min_valley_width=20.0,
            valley_density_threshold=0.3,
            histogram_smooth_sigma=10.0,
            min_inter_valley_ratio=0.20,
        )

    @classmethod
    def for_dense_text(cls) -> 'ReadingOrderConfig':
        """
        Preset for documents with dense, tightly-spaced text.
        
        Use for: Legal documents, contracts, fine print.
        
        Adjustments:
        - Tighter vertical thresholds to avoid merging lines
        - Smaller block gaps
        """
        return cls(
            line_vertical_overlap_threshold=0.6,
            line_baseline_tolerance_multiplier=0.2,
            line_vertical_threshold_multiplier=0.3,
            line_y_spread_multiplier=0.4,
            block_vertical_gap_multiplier=1.5,
        )


def get_default_config() -> ReadingOrderConfig:
    """Get default configuration."""
    return ReadingOrderConfig()


def get_high_quality_config() -> ReadingOrderConfig:
    """
    Configuration optimized for high-quality scans.

    Stricter thresholds for cleaner documents.
    """
    config = ReadingOrderConfig()

    # Tighter thresholds for clean documents
    config.confidence_threshold = 0.8
    config.line_baseline_tolerance_multiplier = 0.2
    config.block_vertical_gap_multiplier = 1.5

    return config


def get_low_quality_config() -> ReadingOrderConfig:
    """
    Configuration optimized for low-quality scans or noisy OCR.

    More lenient thresholds to handle noise.
    """
    config = ReadingOrderConfig()

    # Looser thresholds for noisy documents
    config.confidence_threshold = 0.4
    config.line_baseline_tolerance_multiplier = 0.4
    config.line_vertical_overlap_threshold = 0.3
    config.block_vertical_gap_multiplier = 2.5

    return config


def get_newspaper_config() -> ReadingOrderConfig:
    """
    Configuration optimized for newspaper/magazine layouts.

    Emphasizes column detection.
    """
    config = ReadingOrderConfig()

    # Better column detection
    config.column_separation_threshold = 0.10
    config.min_valley_width = 30.0
    config.valley_density_threshold = 0.2

    return config


def get_form_config() -> ReadingOrderConfig:
    """
    Configuration optimized for forms and tables.

    Emphasizes table detection.
    """
    config = ReadingOrderConfig()

    # Better table detection
    config.table_min_rows = 2
    config.table_min_columns = 2
    config.table_row_threshold = 0.6

    return config
