<script lang="ts">
  import { sortedQueue, sortConfig, setSortColumn } from '../../stores/ocrQueue';
  import type { OCRQueueItem } from '../../types';

  // Props
  export let selectedIds: string[] = [];

  // Reactive statements
  $: allSelected = $sortedQueue.length > 0 && selectedIds.length === $sortedQueue.length;
  $: someSelected = selectedIds.length > 0 && !allSelected;

  /**
   * Format file size to human-readable format
   */
  function formatSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  /**
   * Get status badge class
   */
  function getStatusClass(status: OCRQueueItem['status']): string {
    switch (status) {
      case 'queued':
        return 'bg-yellow-600 text-white';
      case 'processing':
        return 'bg-blue-600 text-white';
      case 'complete':
        return 'bg-green-600 text-white';
      case 'failed':
        return 'bg-red-600 text-white';
      default:
        return 'bg-gray-600 text-gray-200';
    }
  }

  /**
   * Format elapsed time in seconds to MM:SS or HH:MM:SS
   */
  function formatElapsedTime(seconds?: number): string {
    if (!seconds || seconds < 0) return '';

    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  /**
   * Get time waiting in queue (from queuedAt to now or startedAt)
   */
  function getTimeWaiting(queuedAt?: number, startedAt?: number): string {
    if (!queuedAt) return '';
    const endTime = startedAt || Date.now();
    const waitSeconds = (endTime - queuedAt) / 1000;
    return formatElapsedTime(waitSeconds);
  }

  /**
   * Get status display text
   */
  function getStatusText(status: OCRQueueItem['status']): string {
    return status.charAt(0).toUpperCase() + status.slice(1);
  }

  /**
   * Toggle all checkboxes
   */
  function handleSelectAll() {
    if (allSelected) {
      selectedIds = [];
    } else {
      selectedIds = $sortedQueue.map(item => item.id);
    }
  }

  /**
   * Toggle individual checkbox
   */
  function handleToggle(id: string) {
    if (selectedIds.includes(id)) {
      selectedIds = selectedIds.filter(i => i !== id);
    } else {
      selectedIds = [...selectedIds, id];
    }
  }

  /**
   * Handle column header click for sorting
   */
  function handleSort(column: 'fileName' | 'pages' | 'size' | 'status') {
    setSortColumn(column);
  }

  /**
   * Get sort indicator for column
   */
  function getSortIndicator(column: string): string {
    if ($sortConfig.column !== column) return '';
    return $sortConfig.direction === 'asc' ? '↑' : '↓';
  }
</script>

<div class="bg-gray-800 dark:bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
  {#if $sortedQueue.length === 0}
    <div class="text-gray-500 italic text-center py-12">
      No files in queue
    </div>
  {:else}
    <!-- Grid Container with max height and scroll -->
    <div class="overflow-y-auto max-h-[600px]">
      <!-- Header Row -->
      <div class="grid grid-cols-[40px_minmax(200px,2fr)_80px_100px_120px] gap-2 bg-gray-700 border-b border-gray-600 p-2 sticky top-0 z-10">
        <!-- Header Checkbox -->
        <div class="flex items-center justify-center">
          <input
            type="checkbox"
            checked={allSelected}
            indeterminate={someSelected}
            on:change={handleSelectAll}
            class="w-4 h-4 rounded border-gray-500 bg-gray-600 text-blue-600 focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <!-- Column Headers (Sortable) -->
        <button
          on:click={() => handleSort('fileName')}
          class="text-left text-sm font-semibold text-gray-300 hover:text-white transition-colors cursor-pointer"
        >
          File Name {getSortIndicator('fileName')}
        </button>

        <button
          on:click={() => handleSort('pages')}
          class="text-center text-sm font-semibold text-gray-300 hover:text-white transition-colors cursor-pointer"
        >
          Pages {getSortIndicator('pages')}
        </button>

        <button
          on:click={() => handleSort('size')}
          class="text-center text-sm font-semibold text-gray-300 hover:text-white transition-colors cursor-pointer"
        >
          Size {getSortIndicator('size')}
        </button>

        <button
          on:click={() => handleSort('status')}
          class="text-center text-sm font-semibold text-gray-300 hover:text-white transition-colors cursor-pointer"
        >
          Status {getSortIndicator('status')}
        </button>
      </div>

      <!-- Data Rows -->
      {#each $sortedQueue as item (item.id)}
        <div
          class="grid grid-cols-[40px_minmax(200px,2fr)_80px_100px_120px] gap-2 p-2 border-b border-gray-700 hover:bg-gray-750 transition-colors"
        >
          <!-- Row Checkbox -->
          <div class="flex items-center justify-center">
            <input
              type="checkbox"
              checked={selectedIds.includes(item.id)}
              on:change={() => handleToggle(item.id)}
              class="w-4 h-4 rounded border-gray-500 bg-gray-600 text-blue-600 focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <!-- File Name -->
          <div class="text-sm text-gray-200 truncate" title={item.fileName}>
            {item.fileName}
          </div>

          <!-- Pages -->
          <div class="text-sm text-gray-300 text-center">
            {item.pages || '-'}
          </div>

          <!-- Size -->
          <div class="text-sm text-gray-300 text-center">
            {formatSize(item.size)}
          </div>

          <!-- Status Badge with Timing -->
          <div class="flex flex-col items-center gap-1">
            <div class="flex items-center gap-2">
              <span
                class="px-2 py-1 rounded text-xs font-medium {getStatusClass(item.status)}"
              >
                {getStatusText(item.status)}
                {#if item.status === 'processing'}
                  <span class="inline-block animate-spin ml-1">⟳</span>
                {/if}
              </span>
            </div>

            <!-- Timing information -->
            {#if item.status === 'queued' && item.queuedAt}
              <span class="text-xs text-gray-400">
                Waiting: {getTimeWaiting(item.queuedAt)}
              </span>
            {/if}

            {#if item.status === 'processing'}
              <div class="flex flex-col items-center text-xs text-gray-400">
                {#if item.elapsedTime}
                  <span>{formatElapsedTime(item.elapsedTime)}</span>
                {/if}
                {#if item.currentPage && item.totalPages}
                  <span>Page {item.currentPage}/{item.totalPages}</span>
                {/if}
              </div>
            {/if}

            {#if item.status === 'complete' && item.elapsedTime}
              <span class="text-xs text-green-400">
                {formatElapsedTime(item.elapsedTime)}
              </span>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  /* Custom scrollbar styling */
  .overflow-y-auto::-webkit-scrollbar {
    width: 8px;
  }

  .overflow-y-auto::-webkit-scrollbar-track {
    background: #1f2937; /* gray-800 */
  }

  .overflow-y-auto::-webkit-scrollbar-thumb {
    background: #4b5563; /* gray-600 */
    border-radius: 4px;
  }

  .overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #6b7280; /* gray-500 */
  }

  /* Hover effect for rows */
  .hover\:bg-gray-750:hover {
    background-color: #2d3748;
  }

  /* Indeterminate checkbox styling */
  input[type="checkbox"]:indeterminate {
    background-color: #3b82f6; /* blue-600 */
  }
</style>
