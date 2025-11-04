<script lang="ts">
  import { onMount } from 'svelte';
  import Modal from './shared/Modal.svelte';
  import Button from './shared/Button.svelte';
  import { ocrConfig, updateOCRConfig, resetOCRConfig, applyOCRPreset, type OCRConfig } from '../stores/ocrConfig';
  import { invoke } from "@tauri-apps/api/core";

  // Props
  export let isOpen: boolean = false;
  export let onClose: () => void;

  // Local state for form
  let localConfig: OCRConfig;
  let systemRecommendedDpi: number = 300;
  let systemRecommendedBatchSize: number = 10;
  let useSystemDpi: boolean = true;
  let gpuAvailable: boolean = false;
  let gpuMemoryGb: number = 0;
  let systemMemoryGb: number = 0;

  // Available languages for PaddleOCR
  // Common languages - PaddleOCR supports 80+ languages
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

  // DPI options (150-1800 at 150 intervals)
  const dpiOptions = [150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1800];

  // Initialize local config from store when modal opens
  $: if (isOpen) {
    localConfig = { ...$ocrConfig };
    useSystemDpi = false; // Will be set based on actual config
    loadSystemRecommendations();
  }

  /**
   * Load system hardware capabilities and recommendations
   */
  async function loadSystemRecommendations() {
    try {
      const capabilities = await invoke<any>('get_hardware_capabilities');

      if (capabilities) {
        gpuAvailable = capabilities.gpu_available || false;
        gpuMemoryGb = capabilities.gpu_memory_gb || 0;
        systemMemoryGb = capabilities.system_memory_gb || 0;

        // Calculate system recommended DPI based on memory
        if (gpuAvailable && gpuMemoryGb >= 6) {
          systemRecommendedDpi = 400; // High quality for powerful GPUs
        } else if (gpuAvailable && gpuMemoryGb >= 4) {
          systemRecommendedDpi = 300; // Balanced for mid-range GPUs
        } else if (systemMemoryGb >= 16) {
          systemRecommendedDpi = 300; // Standard for good CPU systems
        } else {
          systemRecommendedDpi = 200; // Conservative for limited systems
        }

        // Calculate recommended batch size
        systemRecommendedBatchSize = capabilities.recommended_batch_size || 10;

        // Check if current DPI matches system recommendation
        useSystemDpi = localConfig.dpi === systemRecommendedDpi;
      }
    } catch (error) {
      console.error('Failed to load hardware capabilities:', error);
      // Use defaults if detection fails
      systemRecommendedDpi = 300;
      systemRecommendedBatchSize = 10;
    }
  }

  /**
   * Handle DPI change
   */
  function handleDpiChange(value: number | 'system') {
    if (value === 'system') {
      localConfig.dpi = systemRecommendedDpi;
      useSystemDpi = true;
    } else {
      localConfig.dpi = value;
      useSystemDpi = false;
    }
  }

  /**
   * Handle language change
   */
  function handleLanguageChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    const selectedLanguage = target.value;

    // Update config with selected language
    localConfig.languages = [selectedLanguage];

    // Note: Language models will be auto-downloaded by PaddleOCR on first use
    // No need to manually download here
  }

  /**
   * Handle force CPU mode toggle
   */
  function handleForceCpuToggle() {
    localConfig.useGpu = !localConfig.useGpu;
  }

  /**
   * Apply preset configuration
   */
  function handleApplyPreset(preset: 'maxQuality' | 'balanced' | 'speedOptimized' | 'lowMemory') {
    applyOCRPreset(preset);
    localConfig = { ...$ocrConfig };
    useSystemDpi = false;
  }

  /**
   * Reset to defaults
   */
  function handleReset() {
    resetOCRConfig();
    localConfig = { ...$ocrConfig };
    useSystemDpi = true;
  }

  /**
   * Apply changes and close modal
   */
  function handleApply() {
    // Update the global config store
    updateOCRConfig(localConfig);
    onClose();
  }

  /**
   * Cancel and close modal
   */
  function handleCancel() {
    // Discard local changes
    onClose();
  }

  // Get DPI label text
  function getDpiLabel(): string {
    if (useSystemDpi) {
      return `System Recommendation (${systemRecommendedDpi} DPI)`;
    }
    return `${localConfig.dpi} DPI`;
  }

  // Get GPU status text
  function getGpuStatusText(): string {
    if (!gpuAvailable) {
      return 'No GPU detected';
    }
    return `GPU: ${gpuMemoryGb.toFixed(1)}GB VRAM`;
  }
</script>

<Modal {isOpen} {onClose} title="Advanced OCR Settings">
  <div class="space-y-6">
    <!-- System Information -->
    <div class="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
      <h3 class="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-2">System Information</h3>
      <div class="text-xs text-blue-800 dark:text-blue-200 space-y-1">
        <div>GPU: {gpuAvailable ? `Available (${gpuMemoryGb.toFixed(1)}GB VRAM)` : 'Not available'}</div>
        <div>System RAM: {systemMemoryGb.toFixed(1)}GB</div>
        <div>Recommended DPI: {systemRecommendedDpi}</div>
        <div>Recommended Batch Size: {systemRecommendedBatchSize}</div>
      </div>
    </div>

    <!-- Quick Presets -->
    <div>
      <h3 class="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">Quick Presets</h3>
      <div class="grid grid-cols-2 gap-2">
        <Button variant="secondary" size="sm" on:click={() => handleApplyPreset('maxQuality')}>
          Max Quality
        </Button>
        <Button variant="secondary" size="sm" on:click={() => handleApplyPreset('balanced')}>
          Balanced
        </Button>
        <Button variant="secondary" size="sm" on:click={() => handleApplyPreset('speedOptimized')}>
          Speed Optimized
        </Button>
        <Button variant="secondary" size="sm" on:click={() => handleApplyPreset('lowMemory')}>
          Low Memory
        </Button>
      </div>
    </div>

    <!-- Language Selection -->
    <div>
      <label for="language" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
        OCR Language
      </label>
      <select
        id="language"
        class="w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        on:change={handleLanguageChange}
        value={localConfig.languages[0]}
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
        Maximum DPI Level
      </label>

      <!-- System Recommendation Checkbox -->
      <div class="mb-3">
        <label class="flex items-center cursor-pointer">
          <input
            type="checkbox"
            checked={useSystemDpi}
            on:change={() => handleDpiChange('system')}
            class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
          />
          <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">
            Use System Recommendation ({systemRecommendedDpi} DPI)
          </span>
        </label>
      </div>

      <!-- DPI Slider -->
      <div class="space-y-2">
        <input
          type="range"
          min="0"
          max={dpiOptions.length - 1}
          step="1"
          value={dpiOptions.indexOf(localConfig.dpi)}
          on:input={(e) => handleDpiChange(dpiOptions[parseInt(e.currentTarget.value)])}
          disabled={useSystemDpi}
          class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
        />

        <!-- DPI Value Display -->
        <div class="flex justify-between text-xs text-gray-600 dark:text-gray-400">
          <span>150</span>
          <span class="font-semibold text-gray-900 dark:text-gray-100">
            {getDpiLabel()}
          </span>
          <span>1800</span>
        </div>

        <!-- DPI Markers -->
        <div class="flex justify-between text-xs text-gray-500 dark:text-gray-500 px-1">
          <span>Low</span>
          <span>Standard</span>
          <span>High</span>
          <span>Max</span>
        </div>
      </div>

      <p class="mt-2 text-xs text-gray-500 dark:text-gray-400">
        Higher DPI = Better quality but slower processing and more memory usage
      </p>
    </div>

    <!-- Advanced Options Expander (Optional) -->
    <details class="group">
      <summary class="cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100">
        Advanced Options
      </summary>
      <div class="mt-3 space-y-4 pl-4 border-l-2 border-gray-200 dark:border-gray-700">
        <!-- Batch Size -->
        <div>
          <label for="batchSize" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Batch Size: {localConfig.batchSize}
          </label>
          <input
            type="range"
            id="batchSize"
            min="1"
            max="60"
            step="1"
            bind:value={localConfig.batchSize}
            class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-blue-600"
          />
          <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Number of pages to process at once. Higher = faster but more memory.
          </p>
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

        <!-- Enable Angle Classification -->
        <div class="flex items-center">
          <input
            type="checkbox"
            id="angleClass"
            bind:checked={localConfig.enableAngleClassification}
            class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
          />
          <label for="angleClass" class="ml-2 text-sm text-gray-700 dark:text-gray-300">
            Enable Angle Classification (detect rotated text)
          </label>
        </div>

        <!-- Enable Hybrid Mode -->
        <div class="flex items-center">
          <input
            type="checkbox"
            id="hybridMode"
            bind:checked={localConfig.enableHybridMode}
            class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
          />
          <label for="hybridMode" class="ml-2 text-sm text-gray-700 dark:text-gray-300">
            Enable Hybrid GPU/CPU Mode (for limited VRAM)
          </label>
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
    <Button variant="secondary" size="sm" on:click={handleReset}>
      Reset to Defaults
    </Button>
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
