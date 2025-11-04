// OCR Settings State Management
import { writable } from 'svelte/store';

/**
 * OCR Configuration Settings
 */
export interface OCRSettings {
  language: string;
  forceCPU: boolean;
  useSystemRecommendedDPI: boolean;
  maxDPI: number;
}

/**
 * Default OCR settings
 */
const defaultSettings: OCRSettings = {
  language: 'en',  // English by default
  forceCPU: false,
  useSystemRecommendedDPI: true,
  maxDPI: 300,  // Default if system recommendation is disabled
};

/**
 * Supported languages for PaddleOCR
 * Format: { code: display_name }
 */
export const SUPPORTED_LANGUAGES = {
  'en': 'English',
  'ch': 'Chinese (Simplified)',
  'chinese_cht': 'Chinese (Traditional)',
  'ta': 'Tamil',
  'te': 'Telugu',
  'ka': 'Kannada',
  'latin': 'Latin',
  'arabic': 'Arabic',
  'cyrillic': 'Cyrillic',
  'devanagari': 'Devanagari',
  'french': 'French',
  'german': 'German',
  'japan': 'Japanese',
  'korean': 'Korean',
  'it': 'Italian',
  'es': 'Spanish',
  'pt': 'Portuguese',
  'ru': 'Russian',
  'uk': 'Ukrainian',
  'be': 'Belarusian',
  'ur': 'Urdu',
  'fa': 'Persian',
  'ug': 'Uyghur',
  'kk': 'Kazakh',
  'rs': 'Serbian',
  'bg': 'Bulgarian',
  'hi': 'Hindi',
  'mr': 'Marathi',
  'ne': 'Nepali',
} as const;

/**
 * DPI levels for the slider (150 to 1800 in 150 intervals)
 */
export const DPI_LEVELS = [150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1800];

/**
 * Get display label for DPI slider
 */
export function getDPILabel(dpi: number): string {
  return `${dpi} DPI`;
}

/**
 * OCR settings store
 */
export const ocrSettings = writable<OCRSettings>(defaultSettings);

/**
 * Reset settings to defaults
 */
export function resetOCRSettings(): void {
  ocrSettings.set({ ...defaultSettings });
}

/**
 * Update a specific setting
 */
export function updateOCRSetting<K extends keyof OCRSettings>(
  key: K,
  value: OCRSettings[K]
): void {
  ocrSettings.update(settings => ({
    ...settings,
    [key]: value,
  }));
}

/**
 * Get current OCR settings as a plain object
 */
export function getOCRSettings(): OCRSettings {
  let currentSettings: OCRSettings = defaultSettings;
  ocrSettings.subscribe(settings => {
    currentSettings = settings;
  })();
  return currentSettings;
}
