// OCR Configuration State Management
import { writable, get } from 'svelte/store';

/**
 * OCR Configuration interface
 * Matches the Python backend OCRConfig parameters
 */
export interface OCRConfig {
  // Hardware settings
  useGpu: boolean;
  gpuId: number;

  // Engine selection
  engine: 'paddleocr' | 'tesseract' | 'auto';

  // Language settings
  languages: string[];

  // Processing settings
  batchSize: number;
  maxMemoryMb: number;
  confidenceThreshold: number;

  // Text detection settings
  enableTextDetection: boolean;
  enableAngleClassification: boolean;

  // Performance settings
  numThreads: number;

  // Hybrid GPU/CPU processing
  enableHybridMode: boolean;
  cpuBatchSize: number | null;
  enableVramMonitoring: boolean;
  enableAdaptiveBatchSizing: boolean;

  // DPI setting for rendering PDF pages
  dpi: number;

  // Verbose logging
  verbose: boolean;
}

/**
 * Default OCR configuration
 * Uses auto-detection and optimal settings for most systems
 */
export const defaultOCRConfig: OCRConfig = {
  // Hardware settings
  useGpu: true, // Auto-detect and use GPU if available
  gpuId: 0,

  // Engine selection
  engine: 'auto', // Auto-select best available engine

  // Language settings
  languages: ['en'], // English by default

  // Processing settings
  batchSize: 10, // Will be auto-tuned based on hardware
  maxMemoryMb: 2048,
  confidenceThreshold: 0.5,

  // Text detection settings
  enableTextDetection: true,
  enableAngleClassification: true, // Detect and correct rotated text

  // Performance settings
  numThreads: 4,

  // Hybrid GPU/CPU processing
  enableHybridMode: false, // Auto-enabled for systems with <=4GB VRAM and >=15GB RAM
  cpuBatchSize: null, // Auto-calculated if null
  enableVramMonitoring: true, // Monitor VRAM usage and adapt
  enableAdaptiveBatchSizing: true, // Automatically adjust batch size

  // DPI setting
  dpi: 300, // Standard quality, 300 DPI

  // Verbose logging
  verbose: false,
};

/**
 * OCR Configuration store
 * Manages the current OCR configuration settings
 */
export const ocrConfig = writable<OCRConfig>(defaultOCRConfig);

/**
 * Reset configuration to defaults
 */
export function resetOCRConfig(): void {
  ocrConfig.set({ ...defaultOCRConfig });
}

/**
 * Update a specific configuration value
 */
export function updateOCRConfig(updates: Partial<OCRConfig>): void {
  ocrConfig.update(config => ({
    ...config,
    ...updates,
  }));
}

/**
 * Get current configuration (non-reactive)
 */
export function getOCRConfig(): OCRConfig {
  return get(ocrConfig);
}

/**
 * Export configuration for backend
 * Converts camelCase to snake_case for Python backend compatibility
 */
export function exportOCRConfigForBackend(): Record<string, any> {
  const config = get(ocrConfig);

  return {
    use_gpu: config.useGpu,
    gpu_id: config.gpuId,
    engine: config.engine,
    languages: config.languages,
    batch_size: config.batchSize,
    max_memory_mb: config.maxMemoryMb,
    confidence_threshold: config.confidenceThreshold,
    enable_text_detection: config.enableTextDetection,
    enable_angle_classification: config.enableAngleClassification,
    num_threads: config.numThreads,
    enable_hybrid_mode: config.enableHybridMode,
    cpu_batch_size: config.cpuBatchSize,
    enable_vram_monitoring: config.enableVramMonitoring,
    enable_adaptive_batch_sizing: config.enableAdaptiveBatchSizing,
    dpi: config.dpi,
    verbose: config.verbose,
  };
}

/**
 * Preset configurations for common use cases
 */
export const ocrPresets = {
  /**
   * Maximum Quality - Best accuracy, slower speed
   * - High DPI (600)
   * - Larger batch sizes for efficiency
   * - All detection features enabled
   */
  maxQuality: {
    ...defaultOCRConfig,
    dpi: 600,
    batchSize: 5, // Smaller batch for high DPI
    confidenceThreshold: 0.7, // Higher confidence threshold
    enableTextDetection: true,
    enableAngleClassification: true,
  } as OCRConfig,

  /**
   * Balanced - Good quality and speed (Default)
   * - Standard DPI (300)
   * - Auto-tuned batch size
   * - All detection features enabled
   */
  balanced: defaultOCRConfig,

  /**
   * Speed Optimized - Faster processing, acceptable quality
   * - Lower DPI (200)
   * - Larger batch sizes
   * - Angle classification disabled
   */
  speedOptimized: {
    ...defaultOCRConfig,
    dpi: 200,
    batchSize: 20,
    confidenceThreshold: 0.4, // Lower threshold for speed
    enableAngleClassification: false, // Disable rotation detection for speed
  } as OCRConfig,

  /**
   * Low Memory - Minimal memory usage
   * - Standard DPI (300)
   * - Small batch size
   * - Reduced memory limit
   */
  lowMemory: {
    ...defaultOCRConfig,
    batchSize: 3,
    maxMemoryMb: 1024,
    cpuBatchSize: 2,
  } as OCRConfig,

  /**
   * GPU Intensive - Maximum GPU utilization
   * - High DPI (400)
   * - Large batch size for GPU
   * - Hybrid mode disabled (GPU only)
   */
  gpuIntensive: {
    ...defaultOCRConfig,
    useGpu: true,
    dpi: 400,
    batchSize: 30,
    enableHybridMode: false,
  } as OCRConfig,
};

/**
 * Apply a preset configuration
 */
export function applyOCRPreset(preset: keyof typeof ocrPresets): void {
  const presetConfig = ocrPresets[preset];
  ocrConfig.set({ ...presetConfig });
}
