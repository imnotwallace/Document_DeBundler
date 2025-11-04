<script lang="ts">
  import Modal from './shared/Modal.svelte';
  import Button from './shared/Button.svelte';
  import { ocrSettings, SUPPORTED_LANGUAGES, DPI_LEVELS, getDPILabel } from '../stores/ocrSettings';

  export let isOpen = false;
  export let onClose: () => void;

  // Local state for the modal (apply on save)
  let localSettings = { ...$ocrSettings };

  // Watch for changes to store and update local state when modal opens
  $: if (isOpen) {
    localSettings = { ...$ocrSettings };
  }

  // Handle save
  function handleSave() {
    ocrSettings.set(localSettings);
    onClose();
  }

  // Handle cancel
  function handleCancel() {
    // Reset local state to current store values
    localSettings = { ...$ocrSettings };
    onClose();
  }

  // Get DPI slider value (index in DPI_LEVELS array)
  $: dpiSliderValue = DPI_LEVELS.indexOf(localSettings.maxDPI);

  // Update DPI when slider changes
  function handleDPISliderChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const index = parseInt(target.value);
    localSettings.maxDPI = DPI_LEVELS[index];
  }
</script>

{#if isOpen}
  <Modal title="Advanced OCR Settings" {isOpen} {onClose}>
    <div class="space-y-6">
      <!-- Language Selection -->
      <div>
        <label for="language" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Language
        </label>
        <select
          id="language"
          bind:value={localSettings.language}
          class="w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-900 dark:text-gray-100"
        >
          {#each Object.entries(SUPPORTED_LANGUAGES) as [code, name]}
            <option value={code}>{name}</option>
          {/each}
        </select>
        <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
          Missing language models will be auto-downloaded on first use.
        </p>
      </div>

      <!-- Force CPU Mode -->
      <div class="flex items-start">
        <div class="flex items-center h-5">
          <input
            id="forceCPU"
            type="checkbox"
            bind:checked={localSettings.forceCPU}
            class="w-4 h-4 text-blue-600 bg-white dark:bg-gray-700 border-gray-300 dark:border-gray-600 rounded focus:ring-blue-500 focus:ring-2"
          />
        </div>
        <div class="ml-3">
          <label for="forceCPU" class="text-sm font-medium text-gray-700 dark:text-gray-300">
            Force CPU Mode
          </label>
          <p class="text-xs text-gray-500 dark:text-gray-400">
            Disable GPU acceleration (useful for troubleshooting or when GPU is unavailable)
          </p>
        </div>
      </div>

      <!-- Max DPI Settings -->
      <div>
        <div class="flex items-start mb-3">
          <div class="flex items-center h-5">
            <input
              id="useSystemRecommendedDPI"
              type="checkbox"
              bind:checked={localSettings.useSystemRecommendedDPI}
              class="w-4 h-4 text-blue-600 bg-white dark:bg-gray-700 border-gray-300 dark:border-gray-600 rounded focus:ring-blue-500 focus:ring-2"
            />
          </div>
          <div class="ml-3">
            <label for="useSystemRecommendedDPI" class="text-sm font-medium text-gray-700 dark:text-gray-300">
              System Recommendation
            </label>
            <p class="text-xs text-gray-500 dark:text-gray-400">
              Automatically select optimal DPI based on available hardware
            </p>
          </div>
        </div>

        {#if !localSettings.useSystemRecommendedDPI}
          <div class="pl-7">
            <label for="maxDPI" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Max DPI: {getDPILabel(localSettings.maxDPI)}
            </label>
            <input
              id="maxDPI"
              type="range"
              min="0"
              max={DPI_LEVELS.length - 1}
              step="1"
              value={dpiSliderValue}
              on:input={handleDPISliderChange}
              class="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
            />
            <div class="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
              <span>150 DPI</span>
              <span>900 DPI</span>
              <span>1800 DPI</span>
            </div>
            <p class="mt-2 text-xs text-gray-500 dark:text-gray-400">
              Higher DPI provides better quality but increases processing time and memory usage.
            </p>
          </div>
        {/if}
      </div>

      <!-- Information Box -->
      <div class="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-4">
        <div class="flex">
          <div class="flex-shrink-0">
            <svg class="h-5 w-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd" />
            </svg>
          </div>
          <div class="ml-3">
            <h3 class="text-sm font-medium text-blue-800 dark:text-blue-300">
              Configuration Tips
            </h3>
            <div class="mt-2 text-sm text-blue-700 dark:text-blue-400">
              <ul class="list-disc list-inside space-y-1">
                <li>GPU mode provides 10-15x faster processing (when available)</li>
                <li>System recommendation automatically adapts to your hardware</li>
                <li>Language models (~10MB each) download automatically when needed</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Modal Footer with buttons -->
    <div slot="footer" class="flex justify-end gap-3">
      <Button variant="secondary" on:click={handleCancel}>
        Cancel
      </Button>
      <Button variant="primary" on:click={handleSave}>
        Save Settings
      </Button>
    </div>
  </Modal>
{/if}
