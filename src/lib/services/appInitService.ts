/**
 * App Initialization Service
 * Handles app startup sequence with progress tracking
 */

import { invoke } from '@tauri-apps/api/core';

export interface InitProgress {
  progress: number; // 0-100
  message: string;
  currentStep: string;
  totalSteps: number;
  completedSteps: number;
}

export type InitProgressCallback = (progress: InitProgress) => void;

/**
 * Initialize the application
 * Runs critical startup tasks with progress tracking
 * NOTE: Only blocks on essential initialization (hardware detection)
 * Language packs load in background after app is ready
 */
export async function initializeApp(
  onProgress: InitProgressCallback
): Promise<void> {
  const totalSteps = 2;
  let completedSteps = 0;

  try {
    // Step 1: Start Python backend
    onProgress({
      progress: 10,
      message: 'Starting backend services...',
      currentStep: 'Initializing Python backend',
      totalSteps,
      completedSteps,
    });

    // Python backend starts automatically via Tauri, just verify it's ready
    await new Promise(resolve => setTimeout(resolve, 500));
    completedSteps++;

    // Step 2: Detect hardware capabilities (SLOW - 15-20 seconds)
    // This is CRITICAL and must complete before app is usable
    onProgress({
      progress: 20,
      message: 'Detecting hardware capabilities...',
      currentStep: 'Initializing GPU/CPU detection (this may take 15-20 seconds)',
      totalSteps,
      completedSteps,
    });

    await invoke('get_hardware_capabilities');
    completedSteps++;

    onProgress({
      progress: 90,
      message: 'Hardware detection complete',
      currentStep: 'GPU/CPU capabilities cached',
      totalSteps,
      completedSteps,
    });

    // Complete
    onProgress({
      progress: 100,
      message: 'Initialization complete',
      currentStep: 'Ready',
      totalSteps,
      completedSteps,
    });

    // Brief delay to show completion state
    await new Promise(resolve => setTimeout(resolve, 300));

  } catch (error) {
    console.error('App initialization failed:', error);
    throw new Error(`Initialization failed: ${error}`);
  }
}

/**
 * Check if app has been initialized
 * (Can be used to skip initialization on subsequent loads if needed)
 */
export function isAppInitialized(): boolean {
  // For now, always initialize on app start
  // Could add localStorage flag for "skip on next start" if needed
  return false;
}
