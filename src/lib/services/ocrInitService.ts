/**
 * OCR Initialization Service
 * Handles OCR engine pre-initialization with progress tracking
 * Only runs when user enters the OCR module for the first time
 */

import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { exportOCRConfigForBackend } from '../stores/ocrConfig';
import { ocrInitialized, ocrInitializing } from '../stores/ocrInit';

export interface OcrInitProgress {
  progress: number; // 0-100
  message: string;
  currentStep: string;
  totalSteps: number;
  completedSteps: number;
}

export type OcrInitProgressCallback = (progress: OcrInitProgress) => void;

/**
 * Initialize OCR engines
 * Pre-loads PaddleOCR models and initializes CUDA (30-60 seconds)
 * This is called when user first enters the OCR module
 */
export async function initializeOCR(
  onProgress: OcrInitProgressCallback
): Promise<void> {
  const totalSteps = 3;
  let completedSteps = 0;

  try {
    // Set initializing flag
    ocrInitializing.set(true);

    // Step 1: Prepare initialization
    onProgress({
      progress: 5,
      message: 'Preparing OCR initialization...',
      currentStep: 'Loading configuration',
      totalSteps,
      completedSteps,
    });

    // Get current OCR configuration
    const ocrConfig = exportOCRConfigForBackend();
    await new Promise(resolve => setTimeout(resolve, 100));
    completedSteps++;

    // Step 2: Initialize OCR engines (SLOW - 30-60 seconds)
    onProgress({
      progress: 10,
      message: 'Initializing OCR engines...',
      currentStep: 'Loading models and initializing GPU (this may take 30-60 seconds)',
      totalSteps,
      completedSteps,
    });

    // Listen for progress events from backend
    const unlisten = await listen('python_event', (event: any) => {
      const payload = event.payload;

      if (payload.type === 'progress') {
        const data = payload.data;
        // Map backend progress to our UI progress
        // Backend progress is 0-100, we map to our 10-90 range
        const mappedProgress = 10 + (data.percent * 0.8);

        onProgress({
          progress: mappedProgress,
          message: data.message || 'Initializing OCR...',
          currentStep: data.message || 'Loading models',
          totalSteps,
          completedSteps: 1,
        });
      }
    });

    try {
      // Call the Tauri command to initialize OCR
      await invoke('initialize_ocr', { config: ocrConfig });
      completedSteps++;
    } finally {
      // Clean up event listener
      unlisten();
    }

    // Step 3: Complete
    onProgress({
      progress: 95,
      message: 'OCR initialization complete',
      currentStep: 'Ready to process documents',
      totalSteps,
      completedSteps,
    });

    await new Promise(resolve => setTimeout(resolve, 200));

    onProgress({
      progress: 100,
      message: 'OCR ready',
      currentStep: 'Complete',
      totalSteps,
      completedSteps,
    });

    // Brief delay to show completion state
    await new Promise(resolve => setTimeout(resolve, 300));

    // Set initialization complete flag
    ocrInitialized.set(true);
    ocrInitializing.set(false);

    console.log('[OcrInit] OCR engines initialized successfully');

  } catch (error) {
    console.error('OCR initialization failed:', error);
    ocrInitializing.set(false);
    throw new Error(`OCR initialization failed: ${error}`);
  }
}

/**
 * Check if OCR has been initialized
 * Can be used to skip initialization if already done
 */
export function isOcrInitialized(): boolean {
  // Could add localStorage persistence here if needed
  // For now, re-initialize on each app session
  return false;
}

/**
 * Force re-initialization of OCR engines
 * Useful if OCR settings change
 */
export async function reinitializeOCR(
  onProgress: OcrInitProgressCallback
): Promise<void> {
  ocrInitialized.set(false);
  await initializeOCR(onProgress);
}
