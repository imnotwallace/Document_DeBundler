<script lang="ts">
  import { onMount, afterUpdate } from 'svelte';

  // Props
  export let logs: string[] = [];
  export let maxLines: number = 1000;

  // Local state
  let terminalContainer: HTMLDivElement;
  let autoScroll = true;

  /**
   * Parse log message and extract type for color coding
   */
  function getLogClass(log: string): string {
    if (log.includes('ERROR:')) {
      return 'text-red-500';
    } else if (log.includes('✓') || log.includes('SUCCESS')) {
      return 'text-green-500';
    } else if (log.includes('WARNING:')) {
      return 'text-yellow-500';
    } else {
      return 'text-gray-300';
    }
  }

  /**
   * Scroll to bottom of terminal
   */
  function scrollToBottom() {
    if (terminalContainer && autoScroll) {
      terminalContainer.scrollTop = terminalContainer.scrollHeight;
    }
  }

  /**
   * Check if user has scrolled up (disable auto-scroll)
   */
  function handleScroll() {
    if (!terminalContainer) return;
    
    const { scrollTop, scrollHeight, clientHeight } = terminalContainer;
    const isAtBottom = Math.abs(scrollHeight - clientHeight - scrollTop) < 10;
    
    autoScroll = isAtBottom;
  }

  /**
   * Enforce circular buffer - keep only last maxLines
   */
  $: displayLogs = logs.length > maxLines ? logs.slice(-maxLines) : logs;

  /**
   * Auto-scroll when logs update
   */
  afterUpdate(() => {
    scrollToBottom();
  });

  onMount(() => {
    scrollToBottom();
  });
</script>

<div
  bind:this={terminalContainer}
  on:scroll={handleScroll}
  class="bg-gray-900 text-gray-100 rounded-lg p-4 font-mono text-sm overflow-y-auto h-full"
>
  <!-- Header -->
  <div class="flex items-center justify-between mb-3 pb-2 border-b border-gray-700">
    <div class="text-gray-400 text-xs font-semibold uppercase tracking-wide">
      Terminal Output
    </div>
    <div class="text-gray-500 text-xs">
      {displayLogs.length} lines
      {#if !autoScroll}
        <span class="ml-2 text-yellow-500">(Paused)</span>
      {/if}
    </div>
  </div>

  <!-- Log Content -->
  {#if displayLogs.length === 0}
    <div class="text-gray-500 italic">No logs yet...</div>
  {:else}
    <div class="space-y-0.5">
      {#each displayLogs as log, index (index)}
        <div class="leading-relaxed {getLogClass(log)}">
          {log}
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  /* Custom scrollbar styling for dark theme */
  .overflow-y-auto::-webkit-scrollbar {
    width: 10px;
  }

  .overflow-y-auto::-webkit-scrollbar-track {
    background: #111827; /* gray-900 */
    border-radius: 5px;
  }

  .overflow-y-auto::-webkit-scrollbar-thumb {
    background: #374151; /* gray-700 */
    border-radius: 5px;
  }

  .overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #4b5563; /* gray-600 */
  }

  /* Smooth scrolling */
  .overflow-y-auto {
    scroll-behavior: smooth;
  }
</style>
