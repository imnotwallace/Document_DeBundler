<script lang="ts">
  // FileGrid component stub - will be fully implemented in Phase 3 (OCR Module)
  // This component will display a sortable grid of files with selection

  export let files: any[] = [];
  export let columns: string[] = [];

  // TODO: Implement full grid functionality in Phase 3
  // - Sortable columns
  // - Row selection with checkboxes
  // - Header checkbox for select all
  // - Vertical scrollbar
  // - Row hover effects
</script>

<div
  class="bg-gray-800 dark:bg-gray-900 rounded-lg p-4 border border-gray-700"
>
  <div class="text-gray-400 text-sm mb-2">File Grid (Phase 3)</div>
  {#if files.length === 0}
    <div class="text-gray-500 italic text-center py-8">No files in queue</div>
  {:else}
    <div class="space-y-2">
      {#each files as file}
        <div class="bg-gray-700 p-2 rounded text-sm text-gray-200">
          {file.name || 'File'}
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  /* Will be enhanced in Phase 3 */
</style>
