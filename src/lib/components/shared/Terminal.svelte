<script lang="ts">
  // Terminal component stub - will be fully implemented in Phase 3 (OCR Module)
  // This component will display verbose log output with auto-scroll

  export let logs: string[] = [];
  export let maxLines: number = 1000;

  // TODO: Implement full terminal functionality in Phase 3
  // - Auto-scroll to bottom on new messages
  // - Color coding for INFO/WARNING/ERROR
  // - Monospace font
  // - Circular buffer (max 1000 lines)
</script>

<div
  class="bg-gray-900 text-gray-100 rounded-lg p-4 font-mono text-sm overflow-y-auto h-full"
>
  <div class="text-gray-400 text-xs mb-2">Terminal Output (Phase 3)</div>
  {#if logs.length === 0}
    <div class="text-gray-500 italic">No logs yet...</div>
  {:else}
    {#each logs as log}
      <div class="mb-1">{log}</div>
    {/each}
  {/if}
</div>

<style>
  /* Will be enhanced in Phase 3 */
</style>
