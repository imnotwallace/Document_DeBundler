import { writable } from 'svelte/store';

// OCR initialization state
export const ocrInitialized = writable<boolean>(false);
export const ocrInitializing = writable<boolean>(false);

// Reset OCR initialization state (useful for re-initialization if settings change)
export function resetOcrInitState() {
  ocrInitialized.set(false);
  ocrInitializing.set(false);
}
