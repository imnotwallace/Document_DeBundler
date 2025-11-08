/**
 * Hardware Capabilities Store
 * Stores detected GPU/CPU capabilities from system initialization
 */

import { writable } from 'svelte/store';

export interface HardwareCapabilities {
  gpu_available: boolean;
  cuda_available: boolean;
  gpu_memory_gb: number;
  system_memory_gb: number;
  cpu_count: number;
  platform: string;
  recommended_batch_size: number;
  recommended_dpi: number;
}

const defaultCapabilities: HardwareCapabilities = {
  gpu_available: false,
  cuda_available: false,
  gpu_memory_gb: 0,
  system_memory_gb: 0,
  cpu_count: 1,
  platform: 'unknown',
  recommended_batch_size: 10,
  recommended_dpi: 300,
};

/**
 * Hardware capabilities store
 * Populated during app initialization
 */
export const hardwareCapabilities = writable<HardwareCapabilities>(defaultCapabilities);

/**
 * Update hardware capabilities
 */
export function updateHardwareCapabilities(capabilities: Partial<HardwareCapabilities>): void {
  hardwareCapabilities.update(current => ({
    ...current,
    ...capabilities,
  }));
}

/**
 * Reset to defaults (for testing)
 */
export function resetHardwareCapabilities(): void {
  hardwareCapabilities.set({ ...defaultCapabilities });
}
