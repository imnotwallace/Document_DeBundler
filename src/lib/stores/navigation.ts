// Navigation state management store
import { writable } from 'svelte/store';
import type { Module } from '../types';

/**
 * Current active module
 *
 * Usage:
 * ```svelte
 * <script>
 *   import { currentModule, navigateToOCR } from '$lib/stores/navigation';
 * </script>
 *
 * {#if $currentModule === 'main_menu'}
 *   <MainMenu />
 * {/if}
 * ```
 */
export const currentModule = writable<Module>('main_menu');

/**
 * Navigate to the main menu
 */
export function navigateToMainMenu() {
  currentModule.set('main_menu');
}

/**
 * Navigate to the OCR module
 */
export function navigateToOCR() {
  currentModule.set('ocr');
}

/**
 * Navigate to the De-Bundling module
 */
export function navigateToDebundle() {
  currentModule.set('debundle');
}

/**
 * Navigate to the Bundling module
 */
export function navigateToBundle() {
  currentModule.set('bundle');
}

/**
 * Navigate to a specific module by name
 * @param module The module to navigate to
 */
export function navigateTo(module: Module) {
  currentModule.set(module);
}

/**
 * Get the current module value (non-reactive)
 * Use this for imperative checks, prefer $currentModule in components
 */
export function getCurrentModule(): Module {
  let current: Module = 'main_menu';
  currentModule.subscribe(value => {
    current = value;
  })();
  return current;
}
