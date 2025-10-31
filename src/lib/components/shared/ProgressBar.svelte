<script lang="ts">
  import type { ProgressData } from '../../types';

  // Props
  export let current: ProgressData['current'] = 0;
  export let total: ProgressData['total'] = 100;
  export let message: ProgressData['message'] = '';
  export let percentage: ProgressData['percentage'] = undefined;

  // Calculate percentage if not provided
  $: computedPercentage =
    percentage !== undefined
      ? percentage
      : total > 0
      ? Math.round((current / total) * 100)
      : 0;

  // Clamp percentage between 0 and 100
  $: displayPercentage = Math.min(Math.max(computedPercentage, 0), 100);
</script>

<div class="w-full space-y-2">
  <!-- Progress bar -->
  <div class="relative w-full h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
    <div
      class="absolute top-0 left-0 h-full bg-blue-600 dark:bg-blue-500 transition-all duration-300 ease-out rounded-full"
      style="width: {displayPercentage}%"
      role="progressbar"
      aria-valuenow={displayPercentage}
      aria-valuemin="0"
      aria-valuemax="100"
    >
      <!-- Animated shimmer effect -->
      <div class="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
    </div>
  </div>

  <!-- Progress info -->
  <div class="flex items-center justify-between text-sm">
    <span class="text-gray-700 dark:text-gray-300">
      {#if message}
        {message}
      {:else if total > 0}
        {current} / {total}
      {:else}
        Processing...
      {/if}
    </span>
    <span class="font-semibold text-gray-900 dark:text-gray-100">
      {displayPercentage}%
    </span>
  </div>
</div>

<style>
  @keyframes shimmer {
    0% {
      transform: translateX(-100%);
    }
    100% {
      transform: translateX(100%);
    }
  }

  .animate-shimmer {
    animation: shimmer 2s infinite;
  }
</style>
