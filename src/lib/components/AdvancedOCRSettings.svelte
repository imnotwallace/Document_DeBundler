<script lang="ts">
  import { onMount } from 'svelte';
  import Modal from './shared/Modal.svelte';
  import Button from './shared/Button.svelte';
  import { ocrConfig, updateOCRConfig, resetOCRConfig, type OCRConfig, type ProcessingMode } from '../stores/ocrConfig';
  import { invoke } from "@tauri-apps/api/core";

  // Props
  export let isOpen: boolean = false;
  export let onClose: () => void;

  // Local state for form - Initialize from store immediately
  let localConfig: OCRConfig = $ocrConfig;
  let useDefaultDpi: boolean = false;
  let previousIsOpen: boolean = false; // Track modal state transitions
  let gpuAvailable: boolean = false;
  let gpuMemoryGb: number = 0;
  let systemMemoryGb: number = 0;
  let systemMaxDpi: number = 600; // Will be calculated
  let calculatedBatchSize: number = 10;

  // Constants
  const DEFAULT_DPI = 300;
  const MIN_DPI = 250;

  // Initialize on mount - load hardware capabilities once
  onMount(() => {
    console.log('[AdvancedOCRSettings] Component mounted, loading hardware capabilities');
    // Load hardware capabilities once when component mounts (not every time modal opens)
    loadSystemRecommendations();
  });

  // Available languages for PaddleOCR
  const availableLanguages = [
    { code: 'en', name: 'English', supported: true },
    { code: 'ch', name: 'Chinese (Simplified)', supported: true },
    { code: 'chinese_cht', name: 'Chinese (Traditional)', supported: true },
    { code: 'fr', name: 'French', supported: true },
    { code: 'german', name: 'German', supported: true },
    { code: 'es', name: 'Spanish', supported: true },
    { code: 'pt', name: 'Portuguese', supported: true },
    { code: 'ru', name: 'Russian', supported: true },
    { code: 'ar', name: 'Arabic', supported: true },
    { code: 'hi', name: 'Hindi', supported: true },
    { code: 'japan', name: 'Japanese', supported: true },
    { code: 'korean', name: 'Korean', supported: true },
    { code: 'it', name: 'Italian', supported: true },
    { code: 'nl', name: 'Dutch', supported: true },
    { code: 'vi', name: 'Vietnamese', supported: true },
    { code: 'th', name: 'Thai', supported: true },
    { code: 'tr', name: 'Turkish', supported: true },
    { code: 'pl', name: 'Polish', supported: true },
    { code: 'sv', name: 'Swedish', supported: true },
    { code: 'da', name: 'Danish', supported: true },
  ];

  // Processing mode options with detailed tooltips
  const processingModes: Array<{ value: ProcessingMode; label: string; description: string; tooltip: string }> = [
    {
      value: 'hybrid',
      label: 'Hybrid (Recommended)',
      description: 'Smart analysis: Only OCRs pages that need improvement',
      tooltip: 'Analyzes each page quality. Only re-OCRs pages with poor/missing text. Most efficient - skips good pages, fixes bad ones. Best for mixed documents.'
    },
    {
      value: 'selective',
      label: 'Selective',
      description: 'Only OCR pages with no text layer',
      tooltip: 'Only processes pages with zero text layer. Good for documents where some pages are scanned (no text) and others are digital (have text).'
    },
    {
      value: 'full',
      label: 'Full',
      description: 'Re-OCR all pages regardless',
      tooltip: 'Brute force mode: Re-OCRs every page even if it already has text. Use when you know the existing text layer is garbage. Slowest but most thorough.'
    },
  ];

  // Refresh local config from store when modal opens (transition from closed to open)
  $: if (isOpen && !previousIsOpen) {
    console.log('[AdvancedOCRSettings] Modal opened, loading config from store:', $ocrConfig);
    // IMPORTANT: Deep copy to prevent direct mutations to store
    localConfig = { 
      ...$ocrConfig, 
      languages: [...$ocrConfig.languages] // Deep copy array
    };
    useDefaultDpi = localConfig.dpi === DEFAULT_DPI;
    console.log('[AdvancedOCRSettings] Local config initialized:', localConfig);
    // Reload GPU capabilities when modal opens to ensure fresh data is displayed
    loadSystemRecommendations();
    previousIsOpen = true; // Mark that we've handled this open event
  } else if (!isOpen && previousIsOpen) {
    previousIsOpen = false; // Reset when modal closes
  }

  // Recalculate batch size when DPI or GPU settings change
  $: if (localConfig && localConfig.dpi) {
    calculatedBatchSize = calculateBatchSize(
      localConfig.dpi,
      gpuMemoryGb,
      systemMemoryGb,
      gpuAvailable && localConfig.useGpu
    );
  }

  /**
   * Calculate maximum safe DPI based on VRAM
   * Based on letter-size pages (11 inches max dimension)
   */
  function calculateSystemMaxDpi(vramGb: number): number {
    // Empirical formula from adaptive_max_side_limit.py
    // 4GB VRAM: 8800px max ~ 800 DPI for letter-size (11")
    const BASELINE_VRAM = 4.0;
    const BASELINE_MAX_PX = 8800;
    const SAFETY_MARGIN = 0.1;

    const scaling_factor = vramGb / BASELINE_VRAM;
    const theoretical_max = BASELINE_MAX_PX * scaling_factor;
    const max_dimension = Math.ceil(theoretical_max * (1 - SAFETY_MARGIN) / 1000) * 1000;
    const clamped_max = Math.max(2000, Math.min(max_dimension, 18000));

    // Convert to DPI for letter-size (11 inches)
    const maxDpi = Math.floor(clamped_max / 11);

    // Clamp to reasonable range (300-1600)
    return Math.max(300, Math.min(maxDpi, 1600));
  }

  /**
   * Calculate optimal batch size based on DPI and available memory
   * Formula-based approach (tunable parameters)
   */
  function calculateBatchSize(dpi: number, vramGb: number, ramGb: number, useGpu: boolean): number {
    // Tunable parameters
    const VRAM_UTIL_FACTOR = 0.8;  // Use 80% of VRAM
    const MODEL_OVERHEAD_GB = 0.5; // PaddleOCR model memory
    const LETTER_WIDTH_INCHES = 8.5;
    const LETTER_HEIGHT_INCHES = 11;

    if (useGpu && vramGb > 0) {
      // GPU calculation
      const width_px = LETTER_WIDTH_INCHES * dpi;
      const height_px = LETTER_HEIGHT_INCHES * dpi;
      const pixels_per_image = width_px * height_px;
      const memory_per_image_gb = (pixels_per_image * 3 * 4) / (1024 ** 3); // RGB float32

      const available_vram = (vramGb * VRAM_UTIL_FACTOR) - MODEL_OVERHEAD_GB;

      if (available_vram <= 0) return 1;

      const batch = Math.floor(available_vram / memory_per_image_gb);
      return Math.max(1, Math.min(batch, 50));
    } else {
      // CPU calculation (more conservative)
      const dpi_multiplier = (dpi / 300.0) ** 2;
      let base_batch = 10;

      if (ramGb < 6) base_batch = 3;
      else if (ramGb < 10) base_batch = 5;
      else if (ramGb < 20) base_batch = 10;
      else if (ramGb < 28) base_batch = 15;
      else base_batch = 20;

      return Math.max(1, Math.floor(base_batch / dpi_multiplier));
    }
  }

  /**
   * Load system hardware capabilities and recommendations
   */
  async function loadSystemRecommendations() {
    console.log('[AdvancedOCRSettings] Loading system recommendations...');
    try {
      const capabilities = await invoke<any>('get_hardware_capabilities');
      console.log('[AdvancedOCRSettings] Received capabilities:', capabilities);

      if (capabilities) {
        gpuAvailable = capabilities.gpu_available || false;
        gpuMemoryGb = capabilities.gpu_memory_gb || 0;
        systemMemoryGb = capabilities.system_memory_gb || 0;

        console.log('[AdvancedOCRSettings] GPU:', gpuAvailable, 'VRAM:', gpuMemoryGb, 'RAM:', systemMemoryGb);

        // Calculate system maximum DPI
        if (gpuAvailable && gpuMemoryGb > 0) {
          systemMaxDpi = calculateSystemMaxDpi(gpuMemoryGb);
        } else {
          // CPU fallback: conservative max
          systemMaxDpi = 600;
        }

        // Recalculate batch size
        calculatedBatchSize = calculateBatchSize(localConfig.dpi, gpuMemoryGb, systemMemoryGb, gpuAvailable && localConfig.useGpu);
        console.log('[AdvancedOCRSettings] Calculated - Max DPI:', systemMaxDpi, 'Batch Size:', calculatedBatchSize);
      }
    } catch (error) {
      console.error('[AdvancedOCRSettings] Failed to load hardware capabilities:', error);
      // Use defaults if detection fails
      systemMaxDpi = 600;
      calculatedBatchSize = 10;
    }
  }

  /**
   * Handle DPI change
   */
  function handleDpiChange(value: number | 'default') {
    if (value === 'default') {
      localConfig.dpi = DEFAULT_DPI;
      useDefaultDpi = true;
    } else {
      localConfig.dpi = value;
      useDefaultDpi = false;
    }
  }

  /**
   * Handle force CPU mode toggle
   */
  function handleForceCpuToggle() {
    // Use assignment to trigger Svelte reactivity
    localConfig = { ...localConfig, useGpu: !localConfig.useGpu };
  }

  /**
   * Save current settings as default
   */
  function handleSaveAsDefault() {
    try {
      localStorage.setItem('ocrDefaultConfig', JSON.stringify(localConfig));
      alert('Settings saved as default successfully!');
    } catch (error) {
      console.error('Failed to save default settings:', error);
      alert('Failed to save settings');
    }
  }

  /**
   * Load default settings from localStorage
   */
  function loadDefaultSettings() {
    try {
      const saved = localStorage.getItem('ocrDefaultConfig');
      if (saved) {
        const savedConfig = JSON.parse(saved);
        localConfig = { ...localConfig, ...savedConfig };
        useDefaultDpi = localConfig.dpi === DEFAULT_DPI;
      }
    } catch (error) {
      console.error('Failed to load default settings:', error);
    }
  }

  /**
   * Reset to factory defaults
   */
  function handleReset() {
    resetOCRConfig();
    localConfig = { 
      ...$ocrConfig, 
      languages: [...$ocrConfig.languages] // Deep copy array
    };
    useDefaultDpi = true;
  }

  /**
   * Apply changes and close modal
   */
  function handleApply() {
    console.log('[AdvancedOCRSettings] Applying config:', localConfig);

    // Create final config with angle classification enabled (hidden from UI)
    const finalConfig = {
      ...localConfig,
      enableAngleClassification: true,
      languages: [...localConfig.languages] // Ensure array is copied
    };

    // Update the global config store
    updateOCRConfig(finalConfig);
    console.log('[AdvancedOCRSettings] Config updated in store:', finalConfig);
    onClose();
  }

  /**
   * Cancel and close modal
   */
  function handleCancel() {
    onClose();
  }

  // Get DPI label text
  function getDpiLabel(): string {
    if (useDefaultDpi) {
      return `Default (${DEFAULT_DPI} DPI)`;
    }
    return `${localConfig.dpi} DPI`;
  }

  // Get selected processing mode info
  function getSelectedModeInfo() {
    return processingModes.find(m => m.value === localConfig.processingMode);
  }
</script>

<Modal {isOpen} {onClose} title="Advanced OCR Settings">
  <div class="space-y-6">
    <!-- System Information -->
    <div class="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
      <h3 class="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-2">System Hardware</h3>
      <div class="text-xs text-blue-800 dark:text-blue-200 space-y-1">
        <div>GPU: {gpuAvailable ? `Available (${gpuMemoryGb.toFixed(1)}GB VRAM)` : 'Not available'}</div>
        <div>System RAM: {systemMemoryGb.toFixed(1)}GB</div>
        <div>Maximum Safe DPI: {systemMaxDpi}</div>
        <div>Auto-calculated Batch Size: {calculatedBatchSize} pages</div>
      </div>
    </div>

    <!-- Language Selection -->
    <div>
      <label for="language" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
        OCR Language
      </label>
      <select
        id="language"
        class="w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        bind:value={localConfig.languages[0]}
      >
        {#each availableLanguages as lang}
          <option value={lang.code}>{lang.name}</option>
        {/each}
      </select>
      <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
        Language models will be automatically downloaded if not present (first use only)
      </p>
    </div>

    <!-- Force CPU Mode -->
    <div class="flex items-center justify-between">
      <div>
        <label for="forceCpu" class="text-sm font-medium text-gray-700 dark:text-gray-300">
          Force CPU Mode
        </label>
        <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Disable GPU acceleration and use CPU only
        </p>
      </div>
      <div class="flex items-center">
        <button
          type="button"
          class={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            !localConfig.useGpu ? 'bg-blue-600' : 'bg-gray-300 dark:bg-gray-600'
          }`}
          on:click={handleForceCpuToggle}
          disabled={!gpuAvailable}
        >
          <span
            class={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              !localConfig.useGpu ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>
    </div>

    {#if !gpuAvailable && localConfig.useGpu}
      <div class="text-xs text-yellow-600 dark:text-yellow-400">
        Note: GPU not available, will automatically use CPU mode
      </div>
    {/if}

    <!-- DPI Setting -->
    <div>
      <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
        DPI (Image Quality)
      </label>

      <!-- Default DPI Checkbox -->
      <div class="mb-3">
        <label class="flex items-center cursor-pointer">
          <input
            type="checkbox"
            bind:checked={useDefaultDpi}
            on:change={() => {
              if (useDefaultDpi) {
                localConfig.dpi = DEFAULT_DPI;
              }
            }}
            class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
          />
          <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">
            Use Default ({DEFAULT_DPI} DPI)
          </span>
        </label>
      </div>

      <!-- DPI Slider -->
      <div class="space-y-3">
        <!-- Large Current Value Display -->
        <div class="bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-200 dark:border-blue-700 rounded-lg p-3 text-center">
          <div class="text-3xl font-bold text-blue-600 dark:text-blue-400">
            {useDefaultDpi ? `${DEFAULT_DPI} DPI` : `${localConfig.dpi} DPI`}
          </div>
          <div class="text-xs text-gray-600 dark:text-gray-400 mt-1">
            {useDefaultDpi ? 'Default (Recommended)' : localConfig.dpi < 300 ? 'Fast Processing' : localConfig.dpi === 300 ? 'Balanced (Recommended)' : localConfig.dpi <= 450 ? 'High Quality' : localConfig.dpi <= 600 ? 'Very High Quality' : 'Maximum Quality'}
          </div>
        </div>

        <!-- Slider -->
        <input
          type="range"
          min={MIN_DPI}
          max={systemMaxDpi}
          step="50"
          bind:value={localConfig.dpi}
          on:input={() => { useDefaultDpi = false; }}
          disabled={useDefaultDpi}
          class="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
        />

        <!-- DPI Markers with Values -->
        <div class="flex justify-between text-xs text-gray-600 dark:text-gray-400 px-1">
          <div class="text-center">
            <div class="font-semibold text-gray-900 dark:text-gray-100">{MIN_DPI}</div>
            <div class="text-[10px]">Min</div>
          </div>
          <div class="text-center">
            <div class="font-semibold text-gray-900 dark:text-gray-100">300</div>
            <div class="text-[10px]">Standard</div>
          </div>
          <div class="text-center">
            <div class="text-gray-700 dark:text-gray-300">450</div>
            <div class="text-[10px]">High</div>
          </div>
          <div class="text-center">
            <div class="font-semibold text-gray-900 dark:text-gray-100">600</div>
            <div class="text-[10px]">V. High</div>
          </div>
          <div class="text-center">
            <div class="font-semibold text-gray-900 dark:text-gray-100">{systemMaxDpi}</div>
            <div class="text-[10px]">Max</div>
          </div>
        </div>
      </div>

      <p class="mt-2 text-xs text-gray-500 dark:text-gray-400">
        Higher DPI = Better quality but slower processing and more memory usage. Batch size auto-adjusts.
      </p>
    </div>

    <!-- Advanced Options Expander -->
    <details class="group">
      <summary class="cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100">
        Advanced Options
      </summary>
      <div class="mt-3 space-y-4 pl-4 border-l-2 border-gray-200 dark:border-gray-700">
        <!-- Processing Mode -->
        <div>
          <label for="processingMode" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Processing Mode
          </label>
          <select
            id="processingMode"
            bind:value={localConfig.processingMode}
            class="w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            title={getSelectedModeInfo()?.tooltip || ''}
          >
            {#each processingModes as mode}
              <option value={mode.value} title={mode.tooltip}>{mode.label}</option>
            {/each}
          </select>
          <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
            {getSelectedModeInfo()?.description || ''}
          </p>
          <p class="mt-1 text-xs text-blue-600 dark:text-blue-400">
            {getSelectedModeInfo()?.tooltip || ''}
          </p>
        </div>

        <!-- Batch Size Display (Read-only) -->
        <div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
          <div class="flex items-center justify-between">
            <div>
              <div class="text-sm font-medium text-gray-700 dark:text-gray-300">
                Batch Size (Auto-calculated)
              </div>
              <div class="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Based on {localConfig.dpi} DPI and available {gpuAvailable && localConfig.useGpu ? 'VRAM' : 'RAM'}
              </div>
            </div>
            <div class="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {calculatedBatchSize}
            </div>
          </div>
        </div>

        <!-- Confidence Threshold -->
        <div>
          <label for="confidence" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Confidence Threshold: {localConfig.confidenceThreshold.toFixed(2)}
          </label>
          <input
            type="range"
            id="confidence"
            min="0"
            max="1"
            step="0.05"
            bind:value={localConfig.confidenceThreshold}
            class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-blue-600"
          />
          <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Minimum confidence to accept OCR results. Lower = more results but less accurate.
          </p>
        </div>

        <!-- Verbose Logging -->
        <div class="flex items-center">
          <input
            type="checkbox"
            id="verbose"
            bind:checked={localConfig.verbose}
            class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
          />
          <label for="verbose" class="ml-2 text-sm text-gray-700 dark:text-gray-300">
            Verbose Logging
          </label>
        </div>
      </div>
    </details>
  </div>

  <!-- Modal Footer with Action Buttons -->
  <div slot="footer" class="flex justify-between items-center">
    <div class="flex gap-2">
      <Button variant="secondary" size="sm" on:click={handleReset}>
        Reset to Defaults
      </Button>
      <Button variant="secondary" size="sm" on:click={handleSaveAsDefault}>
        Save as Default
      </Button>
    </div>
    <div class="flex gap-2">
      <Button variant="secondary" size="md" on:click={handleCancel}>
        Cancel
      </Button>
      <Button variant="primary" size="md" on:click={handleApply}>
        Apply
      </Button>
    </div>
  </div>
</Modal>
