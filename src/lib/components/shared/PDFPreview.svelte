<script lang="ts">
  // PDFPreview component stub - will be fully implemented in Phase 4 (De-bundling Module)
  // This component will display a PDF preview panel

  export let startPage: number = 1;
  export let endPage: number = 1;
  export let onClose: () => void = () => {};

  // TODO: Implement full preview functionality in Phase 4
  // - Render PDF pages as images
  // - Display page numbers
  // - Scrollable page viewer
  // - Close button
</script>

<div
  class="bg-gray-800 dark:bg-gray-900 rounded-lg p-4 border border-gray-700 h-full"
>
  <div class="flex items-center justify-between mb-4">
    <h3 class="text-lg font-semibold text-gray-100">
      Preview: Pages {startPage}-{endPage}
    </h3>
    <button
      type="button"
      class="text-gray-400 hover:text-gray-200 transition-colors"
      on:click={onClose}
      aria-label="Close preview"
    >
      <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M6 18L18 6M6 6l12 12"
        />
      </svg>
    </button>
  </div>

  <div class="text-gray-500 italic text-center py-8">
    PDF Preview (Phase 4)
  </div>
</div>

<style>
  /* Will be enhanced in Phase 4 */
</style>
