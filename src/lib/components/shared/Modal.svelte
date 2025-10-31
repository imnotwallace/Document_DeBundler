<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import type { ModalProps } from '../../types';

  export let title: ModalProps['title'];
  export let isOpen: ModalProps['isOpen'];
  export let onClose: ModalProps['onClose'];

  // Handle ESC key press
  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Escape' && isOpen) {
      onClose();
    }
  }

  // Handle click outside modal
  function handleBackdropClick(event: MouseEvent) {
    if (event.target === event.currentTarget) {
      onClose();
    }
  }

  // Add/remove event listener for ESC key
  onMount(() => {
    window.addEventListener('keydown', handleKeydown);
  });

  onDestroy(() => {
    window.removeEventListener('keydown', handleKeydown);
  });
</script>

{#if isOpen}
  <div
    class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fade-in"
    on:click={handleBackdropClick}
    role="dialog"
    aria-modal="true"
    aria-labelledby="modal-title"
  >
    <div
      class="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-hidden animate-scale-in"
    >
      <!-- Modal Header -->
      <div
        class="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700"
      >
        <h2
          id="modal-title"
          class="text-xl font-semibold text-gray-900 dark:text-gray-100"
        >
          {title}
        </h2>
        <button
          type="button"
          class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
          on:click={onClose}
          aria-label="Close modal"
        >
          <svg
            class="w-6 h-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      <!-- Modal Body -->
      <div
        class="px-6 py-4 overflow-y-auto max-h-[calc(90vh-8rem)] text-gray-700 dark:text-gray-300"
      >
        <slot />
      </div>

      <!-- Modal Footer (optional slot) -->
      {#if $$slots.footer}
        <div
          class="px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900"
        >
          <slot name="footer" />
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  /* Animation keyframes */
  @keyframes fade-in {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }

  @keyframes scale-in {
    from {
      opacity: 0;
      transform: scale(0.95);
    }
    to {
      opacity: 1;
      transform: scale(1);
    }
  }

  .animate-fade-in {
    animation: fade-in 0.2s ease-out;
  }

  .animate-scale-in {
    animation: scale-in 0.2s ease-out;
  }
</style>
