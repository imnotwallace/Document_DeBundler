// OCR Queue State Management
import { writable, get, derived } from 'svelte/store';
import type { OCRQueueItem, SortConfig } from '../types';

/**
 * OCR queue state store
 * Manages the list of files in the OCR processing queue
 */
export const ocrQueue = writable<OCRQueueItem[]>([]);

/**
 * Destination folder for OCR output
 */
export const destinationFolder = writable<string>('./outputs/ocr');

/**
 * Add a file to the OCR queue
 * @param file File information to add
 */
export function addToQueue(file: Omit<OCRQueueItem, 'id' | 'status'>): void {
  const newItem: OCRQueueItem = {
    ...file,
    id: crypto.randomUUID(),
    status: 'pending',
    progress: 0
  };
  
  ocrQueue.update(queue => [...queue, newItem]);
}

/**
 * Add multiple files to the queue at once
 * @param files Array of file information
 */
export function addMultipleToQueue(files: Omit<OCRQueueItem, 'id' | 'status'>[]): void {
  const newItems: OCRQueueItem[] = files.map(file => ({
    ...file,
    id: crypto.randomUUID(),
    status: 'pending',
    progress: 0
  }));
  
  ocrQueue.update(queue => [...queue, ...newItems]);
}

/**
 * Remove files from queue by IDs
 * @param ids Array of file IDs to remove
 */
export function removeFromQueue(ids: string[]): void {
  ocrQueue.update(queue => queue.filter(item => !ids.includes(item.id)));
}

/**
 * Update the status of a file in the queue
 * @param id File ID
 * @param status New status
 * @param progress Optional progress percentage
 * @param error Optional error message
 */
export function updateFileStatus(
  id: string,
  status: OCRQueueItem['status'],
  progress?: number,
  error?: string
): void {
  ocrQueue.update(queue => 
    queue.map(item => {
      if (item.id === id) {
        return {
          ...item,
          status,
          progress: progress ?? item.progress,
          error
        };
      }
      return item;
    })
  );
}

/**
 * Clear all files from the queue
 */
export function clearQueue(): void {
  ocrQueue.set([]);
}

/**
 * Get the current queue (non-reactive)
 */
export function getQueue(): OCRQueueItem[] {
  return get(ocrQueue);
}

/**
 * Get count of files by status
 */
export function getStatusCounts(): {
  pending: number;
  processing: number;
  complete: number;
  failed: number;
  total: number;
} {
  const queue = get(ocrQueue);
  return {
    pending: queue.filter(f => f.status === 'pending').length,
    processing: queue.filter(f => f.status === 'processing').length,
    complete: queue.filter(f => f.status === 'complete').length,
    failed: queue.filter(f => f.status === 'failed').length,
    total: queue.length
  };
}

/**
 * Check if queue is empty
 */
export function isQueueEmpty(): boolean {
  return get(ocrQueue).length === 0;
}

/**
 * Check if any file is currently processing
 */
export function isProcessing(): boolean {
  return get(ocrQueue).some(item => item.status === 'processing');
}

/**
 * Sort configuration store
 */
export const sortConfig = writable<SortConfig>({
  column: 'fileName',
  direction: 'asc'
});

/**
 * Set sort column and toggle direction if same column
 * @param column Column to sort by
 */
export function setSortColumn(column: SortConfig['column']): void {
  sortConfig.update(config => {
    if (config.column === column) {
      // Toggle direction if same column
      return {
        column,
        direction: config.direction === 'asc' ? 'desc' : 'asc'
      };
    } else {
      // New column, default to ascending
      return {
        column,
        direction: 'asc'
      };
    }
  });
}

/**
 * Derived store that returns the sorted queue
 */
export const sortedQueue = derived(
  [ocrQueue, sortConfig],
  ([$queue, $config]) => {
    const queue = [...$queue];
    
    return queue.sort((a, b) => {
      let comparison = 0;
      
      switch ($config.column) {
        case 'fileName':
          comparison = a.fileName.localeCompare(b.fileName);
          break;
        case 'pages':
          comparison = a.pages - b.pages;
          break;
        case 'size':
          comparison = a.size - b.size;
          break;
        case 'status':
          // Sort by status: pending < processing < complete < failed
          const statusOrder = { pending: 0, processing: 1, complete: 2, failed: 3 };
          comparison = statusOrder[a.status] - statusOrder[b.status];
          break;
      }
      
      // Apply direction
      return $config.direction === 'asc' ? comparison : -comparison;
    });
  }
);
