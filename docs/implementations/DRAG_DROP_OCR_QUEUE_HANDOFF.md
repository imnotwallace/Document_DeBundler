# Drag & Drop for OCR Queue - Implementation Handoff

**Date**: 2025-11-07
**Feature**: Drag-and-drop interface for adding PDF documents to OCR queue
**Target**: Center panel of OCR Module only
**Status**: Ready for implementation

## Table of Contents
1. [Context & Objectives](#context--objectives)
2. [User Requirements](#user-requirements)
3. [Current System Analysis](#current-system-analysis)
4. [Technical Approach](#technical-approach)
5. [Implementation Checklist](#implementation-checklist)
6. [Code Examples & Patterns](#code-examples--patterns)
7. [Testing Strategy](#testing-strategy)
8. [Potential Gotchas](#potential-gotchas)
9. [Resources](#resources)

---

## Context & Objectives

### What We're Building
A drag-and-drop interface that allows users to add PDF documents to the OCR processing queue by dragging files or folders onto the center panel of the OCR Module.

### Why It Matters
- **UX Improvement**: Streamlines the file addition workflow
- **Batch Operations**: Makes it easier to add large numbers of files
- **Modern UX Pattern**: Meets user expectations for desktop applications
- **Folder Support**: Enables bulk import by dropping entire folders

### Success Criteria
- Users can drag single/multiple PDF files onto center panel
- Users can drag folders (recursively scanned for PDFs)
- Visual feedback clearly indicates drop zone and drag-over state
- Duplicate files are silently skipped
- Drops are blocked during active OCR processing
- No impact on existing "Add Files" button functionality

---

## User Requirements

Based on Q&A with product owner:

| Requirement | Decision |
|-------------|----------|
| **Drop Zone Visibility** | Always visible - overlay on FileGrid |
| **Folder Handling** | Recursively scan folders for PDF files |
| **Duplicate Handling** | Skip files already in queue silently (no notification) |
| **Drop During Processing** | Block drops when OCR is actively running |

---

## Current System Analysis

### UI Structure

**OCRModule.svelte** - Three-panel layout:
- **Left Panel (20%)**: Controls (Add Files button, settings, Start/Cancel)
- **Center Panel (50%)**: FileGrid component showing queue
- **Right Panel (30%)**: Terminal for logs

**Center Panel Structure** (lines 480-502):
```svelte
<!-- Center Panel (50%) - Hero Section -->
<div class="w-[50%] border-r border-gray-300 dark:border-gray-700 p-4 flex flex-col">
  {#if hasSelection}
    <div class="mb-4">
      <Button variant="danger" size="sm" on:click={handleRemoveSelected}>
        Remove Selected ({selectedFileIds.length})
      </Button>
    </div>
  {/if}

  <div class="flex-1 overflow-auto">
    <FileGrid bind:selectedIds={selectedFileIds} />
  </div>
</div>
```

### Queue Management

**Store**: `src/lib/stores/ocrQueue.ts`

**Queue Item Structure**:
```typescript
interface OCRQueueItem {
  id: string;              // crypto.randomUUID()
  fileName: string;
  filePath: string;
  pages: number;
  size: number;
  status: 'pending' | 'processing' | 'complete' | 'failed';
  progress?: number;
  error?: string;
}
```

**Key Store Functions**:
- `addToQueue(file)` - Add single file
- `addMultipleToQueue(files)` - Add batch of files
- `removeFromQueue(ids)` - Remove by ID array
- `updateFileStatus(id, status, progress, error)`
- `clearQueue()` - Remove all files
- `sortedQueue` - Derived store with sorting

### Current File Selection Flow

**1. User clicks "Add Files" button** (OCRModule.svelte:166-205):
```typescript
async function handleAddFiles() {
  // Open file dialog
  const filePaths = await invoke<string[]>("select_multiple_pdf_files");

  // Get metadata for each file
  const fileInfos: FileInfo[] = [];
  for (const filePath of filePaths) {
    const info = await invoke<FileInfo>("get_file_info", { filePath });
    fileInfos.push(info);
  }

  // Add to queue store
  addMultipleToQueue(fileInfos.map(info => ({
    fileName: info.name,
    filePath: info.path,
    pages: info.page_count || 0,
    size: info.size
  })));

  // Load page counts asynchronously
  loadPageCounts();
}
```

**2. Tauri Commands** (src-tauri/src/commands.rs):

```rust
// Lines 103-117: File dialog
#[tauri::command]
pub async fn select_multiple_pdf_files(app: tauri::AppHandle)
    -> Result<Vec<String>, String>

// Lines 133-158: Get file metadata
#[tauri::command]
pub async fn get_file_info(file_path: String, ...)
    -> Result<FileInfo, String>

// Lines 161-244: Get page count via Python
#[tauri::command]
pub async fn get_pdf_page_count(file_path: String, ...)
    -> Result<Option<u32>, String>
```

**3. Python Backend** (python-backend/main.py):
```python
# "analyze" command returns page count and structure
{
  "command": "analyze",
  "file_path": "/path/to/file.pdf",
  "options": {}
}
```

### File Validation

**Rust Side** (commands.rs:323-331):
```rust
// Validates file exists and is PDF
for file in &files {
    let path = PathBuf::from(file);
    if !path.exists() {
        return Err(format!("File not found: {}", file));
    }
    if !file.to_lowercase().ends_with(".pdf") {
        return Err(format!("File must be a PDF: {}", file));
    }
}
```

### Processing State

**OCRModule.svelte** tracks processing state:
```typescript
let isProcessing = false;  // Set to true during OCR batch processing
```

Used to disable controls during processing (Start button becomes Cancel, Add Files disabled).

---

## Technical Approach

### Architecture Overview

```
User Drag Files/Folders
    ↓
DropZone Component (HTML5 Drag/Drop API)
    ↓
Extract Files + Scan Folders (FileSystemEntry API)
    ↓
Filter: PDFs only, Skip Duplicates
    ↓
Get File Info (Tauri commands)
    ↓
Add to Queue Store (addMultipleToQueue)
    ↓
Load Page Counts Async (existing flow)
    ↓
UI Updates (FileGrid re-renders)
```

### Component Design

**New Component**: `src/lib/components/shared/DropZone.svelte`

**Responsibilities**:
1. Handle HTML5 drag/drop events
2. Manage visual states (idle, drag-over, disabled, empty)
3. Extract files from DataTransfer
4. Recursively scan folders for PDFs
5. Emit custom event with file list
6. Show empty state when queue is empty
7. Prevent default browser behavior

**Props**:
```typescript
export let disabled = false;      // Blocks drops (during processing)
export let isEmpty = false;       // Shows empty state hero section
export let acceptedTypes = ['.pdf'];  // File type filter
```

**Events**:
```typescript
// Dispatched when files are dropped
createEventDispatcher<{
  filesDropped: File[];
}>();
```

**Visual States**:
- **Idle**: Transparent, no visual indicator
- **Drag-Over**: Dashed border, blue accent, overlay with text
- **Disabled**: Gray overlay with "Processing..." message
- **Empty**: Large hero section with icon and instructions

### HTML5 Drag & Drop Events

**Event Flow**:
```typescript
1. dragenter  → Set isDragging = true, dragCounter++
2. dragover   → Maintain drag state, preventDefault() [REQUIRED]
3. dragleave  → dragCounter--, if 0 set isDragging = false
4. drop       → Process files, preventDefault(), isDragging = false
```

**Why dragCounter?**
Nested elements trigger multiple dragenter/dragleave events. Counter ensures we only clear isDragging when leaving the entire drop zone.

### Folder Scanning Algorithm

**FileSystemEntry API** (webkit):

```typescript
async function scanFolder(
  entry: FileSystemDirectoryEntry,
  depth: number = 0
): Promise<File[]> {
  const MAX_DEPTH = 10;
  if (depth > MAX_DEPTH) return [];

  const files: File[] = [];
  const reader = entry.createReader();

  // Read all entries in directory
  const entries = await new Promise<FileSystemEntry[]>((resolve, reject) => {
    reader.readEntries(resolve, reject);
  });

  // Process each entry
  for (const entry of entries) {
    if (entry.isFile) {
      const file = await getFileFromEntry(entry as FileSystemFileEntry);
      if (file.name.toLowerCase().endsWith('.pdf')) {
        files.push(file);
      }
    } else if (entry.isDirectory) {
      // Recursive scan
      const subFiles = await scanFolder(
        entry as FileSystemDirectoryEntry,
        depth + 1
      );
      files.push(...subFiles);
    }
  }

  return files;
}

function getFileFromEntry(
  entry: FileSystemFileEntry
): Promise<File> {
  return new Promise((resolve, reject) => {
    entry.file(resolve, reject);
  });
}
```

### Duplicate Detection

**Strategy**: Check file paths against current queue

```typescript
function filterDuplicates(files: File[], queue: OCRQueueItem[]): File[] {
  const queuePaths = new Set(queue.map(item => item.filePath));
  return files.filter(file => !queuePaths.has(file.path));
}
```

**Note**: Files in browser don't have full paths by default. We'll need to use the file name or get the path after processing starts. Consider using `file.name` for duplicate detection or accepting that same-named files from different directories might be treated as duplicates.

**Alternative**: Use Tauri's file dialog result paths for comparison, which provides full paths.

### Integration with Existing Flow

**handleFilesDropped** (to be added to OCRModule.svelte):

```typescript
async function handleFilesDropped(event: CustomEvent<File[]>) {
  const droppedFiles = event.detail;

  // Filter duplicates
  const existingPaths = new Set($ocrQueue.map(item => item.filePath));
  const newFiles = droppedFiles.filter(file => !existingPaths.has(file.path));

  if (newFiles.length === 0) {
    return; // All duplicates, skip silently
  }

  // Show loading indicator
  isLoadingFiles = true;

  try {
    // Get file info for each new file
    const fileInfos: FileInfo[] = [];
    for (const file of newFiles) {
      // Convert File to path (may need special handling)
      const filePath = file.path || file.name;
      const info = await invoke<FileInfo>("get_file_info", { filePath });
      fileInfos.push(info);
    }

    // Add to queue
    addMultipleToQueue(fileInfos.map(info => ({
      fileName: info.name,
      filePath: info.path,
      pages: info.page_count || 0,
      size: info.size
    })));

    // Load page counts asynchronously
    loadPageCounts();

  } catch (error) {
    console.error("Error processing dropped files:", error);
    // Show error notification
  } finally {
    isLoadingFiles = false;
  }
}
```

**IMPORTANT**: File API in browsers doesn't expose full file paths for security reasons. We may need to use Tauri's file system API or handle files differently. Research needed on how Tauri handles dropped files.

---

## Implementation Checklist

### Phase 1: Create DropZone Component
- [ ] Create `src/lib/components/shared/DropZone.svelte`
- [ ] Add HTML5 drag/drop event handlers (dragenter, dragover, dragleave, drop)
- [ ] Implement drag counter for nested element handling
- [ ] Add preventDefault() calls to prevent browser default behavior
- [ ] Create visual state variables (isDragging, isDisabled)
- [ ] Add TypeScript types for props and events
- [ ] Test basic drag-over visual feedback

### Phase 2: Implement File/Folder Processing
- [ ] Add FileSystemEntry API types
- [ ] Implement `scanFolder()` recursive function
- [ ] Implement `getFileFromEntry()` helper
- [ ] Add maximum depth limit (10 levels)
- [ ] Filter for .pdf extensions (case-insensitive)
- [ ] Flatten collected files into single array
- [ ] Add error handling for permission issues
- [ ] Test with various folder structures

### Phase 3: Visual States & UI
- [ ] Implement idle state (transparent overlay)
- [ ] Implement drag-over state (dashed border + overlay)
- [ ] Implement disabled state (gray overlay + message)
- [ ] Implement empty state (hero section with icon)
- [ ] Add CSS transitions (200ms smooth)
- [ ] Add proper z-index layering
- [ ] Ensure dark mode support
- [ ] Add upload icon to empty state
- [ ] Test all states visually

### Phase 4: Integration with OCRModule
- [ ] Import DropZone component into OCRModule.svelte
- [ ] Wrap center panel content with DropZone
- [ ] Pass `disabled={isProcessing}` prop
- [ ] Pass `isEmpty={$ocrQueue.length === 0}` prop
- [ ] Add `handleFilesDropped` event handler
- [ ] Research Tauri file path handling for dropped files
- [ ] Implement duplicate detection logic
- [ ] Wire up to existing `get_file_info` flow
- [ ] Maintain existing FileGrid display
- [ ] Test integration with existing "Add Files" button

### Phase 5: Error Handling & Edge Cases
- [ ] Handle non-PDF files (skip silently)
- [ ] Handle unreadable files (show notification)
- [ ] Handle empty folders (no action)
- [ ] Handle permission errors (show specific error)
- [ ] Handle large batches (show progress indicator)
- [ ] Add loading state during file info gathering
- [ ] Add timeout for folder scanning (30s max)
- [ ] Test all edge cases

### Phase 6: Testing & Polish
- [ ] Test single file drop
- [ ] Test multiple files drop
- [ ] Test folder drop (shallow)
- [ ] Test folder drop (nested)
- [ ] Test mixed PDF/non-PDF files
- [ ] Test duplicate files
- [ ] Test drop during processing (should be blocked)
- [ ] Test empty queue state
- [ ] Test visual feedback in light mode
- [ ] Test visual feedback in dark mode
- [ ] Test with very large batches (100+ files)
- [ ] Check accessibility (keyboard support)
- [ ] Verify no console errors

### Phase 7: Documentation
- [ ] Add JSDoc comments to DropZone component
- [ ] Update CLAUDE.md with drag-and-drop usage
- [ ] Add example to DEVELOPER_QUICK_START.md
- [ ] Create user-facing documentation if needed

---

## Code Examples & Patterns

### DropZone Component Structure

```svelte
<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  // Props
  export let disabled = false;
  export let isEmpty = false;
  export let acceptedTypes = ['.pdf'];

  // State
  let isDragging = false;
  let dragCounter = 0;
  let isScanning = false;

  // Events
  const dispatch = createEventDispatcher<{
    filesDropped: File[];
  }>();

  // Drag handlers
  function handleDragEnter(e: DragEvent) {
    if (disabled) return;
    e.preventDefault();
    dragCounter++;
    isDragging = true;
  }

  function handleDragOver(e: DragEvent) {
    if (disabled) return;
    e.preventDefault(); // REQUIRED for drop to work
    e.dataTransfer!.dropEffect = 'copy';
  }

  function handleDragLeave(e: DragEvent) {
    if (disabled) return;
    dragCounter--;
    if (dragCounter === 0) {
      isDragging = false;
    }
  }

  async function handleDrop(e: DragEvent) {
    if (disabled) return;
    e.preventDefault();
    isDragging = false;
    dragCounter = 0;

    isScanning = true;
    try {
      const files = await extractFilesFromDrop(e.dataTransfer!);
      const pdfFiles = files.filter(f =>
        f.name.toLowerCase().endsWith('.pdf')
      );

      if (pdfFiles.length > 0) {
        dispatch('filesDropped', pdfFiles);
      }
    } catch (error) {
      console.error('Error processing drop:', error);
    } finally {
      isScanning = false;
    }
  }

  async function extractFilesFromDrop(
    dataTransfer: DataTransfer
  ): Promise<File[]> {
    const items = Array.from(dataTransfer.items);
    const files: File[] = [];

    for (const item of items) {
      if (item.kind === 'file') {
        const entry = item.webkitGetAsEntry();
        if (entry) {
          if (entry.isFile) {
            const file = await getFileFromEntry(entry as FileSystemFileEntry);
            files.push(file);
          } else if (entry.isDirectory) {
            const folderFiles = await scanFolder(entry as FileSystemDirectoryEntry);
            files.push(...folderFiles);
          }
        }
      }
    }

    return files;
  }

  // Helper functions (scanFolder, getFileFromEntry) here...
</script>

<div
  class="relative h-full"
  on:dragenter={handleDragEnter}
  on:dragover={handleDragOver}
  on:dragleave={handleDragLeave}
  on:drop={handleDrop}
>
  <!-- Child content (FileGrid) -->
  <slot />

  <!-- Drag overlay -->
  {#if isDragging && !disabled}
    <div class="absolute inset-0 bg-blue-100/50 dark:bg-blue-900/50
                border-2 border-dashed border-blue-500 rounded-lg
                flex items-center justify-center z-10 pointer-events-none">
      <div class="text-center">
        <p class="text-lg font-semibold text-blue-700 dark:text-blue-300">
          Drop files to add to queue
        </p>
      </div>
    </div>
  {/if}

  <!-- Disabled overlay -->
  {#if disabled}
    <div class="absolute inset-0 bg-gray-500/50
                flex items-center justify-center z-10">
      <p class="text-white font-semibold">
        Processing... cannot add files
      </p>
    </div>
  {/if}

  <!-- Empty state -->
  {#if isEmpty && !isDragging}
    <div class="absolute inset-0 flex items-center justify-center">
      <div class="text-center">
        <!-- Upload icon here -->
        <p class="text-xl font-semibold mb-2">
          Drag and drop PDF files here
        </p>
        <p class="text-gray-500">
          or click "Add Files" to browse
        </p>
      </div>
    </div>
  {/if}

  <!-- Scanning indicator -->
  {#if isScanning}
    <div class="absolute inset-0 bg-white/80 dark:bg-gray-900/80
                flex items-center justify-center z-20">
      <p class="text-lg">Scanning folders...</p>
    </div>
  {/if}
</div>
```

### OCRModule Integration

```svelte
<script lang="ts">
  import DropZone from '$lib/components/shared/DropZone.svelte';

  // ... existing code ...

  async function handleFilesDropped(event: CustomEvent<File[]>) {
    const droppedFiles = event.detail;

    // Filter duplicates
    const existingPaths = new Set($ocrQueue.map(item => item.filePath));
    const newFiles = droppedFiles.filter(file =>
      !existingPaths.has(file.path || file.name)
    );

    if (newFiles.length === 0) {
      return; // All duplicates, skip silently
    }

    isLoadingFiles = true;

    try {
      const fileInfos: FileInfo[] = [];

      for (const file of newFiles) {
        // TODO: Research how to get full path from dropped File
        // May need to use Tauri's API or read file as blob
        const filePath = (file as any).path || file.name;

        const info = await invoke<FileInfo>("get_file_info", {
          filePath
        });
        fileInfos.push(info);
      }

      addMultipleToQueue(fileInfos.map(info => ({
        fileName: info.name,
        filePath: info.path,
        pages: info.page_count || 0,
        size: info.size
      })));

      loadPageCounts();

    } catch (error) {
      console.error("Error processing dropped files:", error);
      terminal.error(`Failed to add files: ${error}`);
    } finally {
      isLoadingFiles = false;
    }
  }
</script>

<!-- Center Panel -->
<div class="w-[50%] border-r border-gray-300 dark:border-gray-700 p-4 flex flex-col">
  {#if hasSelection}
    <div class="mb-4">
      <Button variant="danger" size="sm" on:click={handleRemoveSelected}>
        Remove Selected ({selectedFileIds.length})
      </Button>
    </div>
  {/if}

  <div class="flex-1 overflow-auto">
    <DropZone
      disabled={isProcessing}
      isEmpty={$ocrQueue.length === 0}
      on:filesDropped={handleFilesDropped}
    >
      <FileGrid bind:selectedIds={selectedFileIds} />
    </DropZone>
  </div>
</div>
```

---

## Testing Strategy

### Manual Testing Checklist

**Basic Functionality**:
- [ ] Drag single PDF file → Added to queue
- [ ] Drag multiple PDF files → All added to queue
- [ ] Drag folder with PDFs → All PDFs found and added
- [ ] Drag nested folders → Recursively scans, all PDFs added
- [ ] Drag mixed files → Only PDFs added, others skipped

**Duplicate Handling**:
- [ ] Drag file already in queue → Skipped silently
- [ ] Drag mix of new and duplicate files → Only new ones added

**State Management**:
- [ ] Drag during processing → Blocked, shows disabled overlay
- [ ] Empty queue → Shows hero section
- [ ] Non-empty queue → Shows FileGrid with overlay on drag

**Visual Feedback**:
- [ ] Drag over center panel → Highlights correctly
- [ ] Drag over other panels → No highlight
- [ ] Drag leave → Highlight clears
- [ ] Drop animation → Smooth transition
- [ ] Light mode → All states visible
- [ ] Dark mode → All states visible

**Edge Cases**:
- [ ] Drag 100+ files → Shows loading, all added
- [ ] Drag very deep folder structure → Max depth respected
- [ ] Drag empty folder → No error, no files added
- [ ] Drag folder with no PDFs → No error, no files added
- [ ] Drag file without permissions → Error shown
- [ ] Drag corrupted PDF → Error shown
- [ ] Drag while page counts loading → No conflicts

**Existing Features**:
- [ ] "Add Files" button still works → Opens dialog correctly
- [ ] File removal still works → Selected files removed
- [ ] Queue sorting still works → Columns sortable
- [ ] OCR processing still works → Files process correctly

### Automated Testing

**Unit Tests** (if implementing):
```typescript
// Test file extraction
describe('DropZone', () => {
  test('extracts PDF files from drop', async () => {
    // Mock DataTransfer with PDF files
    // Test extractFilesFromDrop function
    // Assert correct files extracted
  });

  test('filters out non-PDF files', async () => {
    // Mock DataTransfer with mixed files
    // Assert only PDFs remain
  });

  test('scans folders recursively', async () => {
    // Mock folder structure
    // Assert all nested PDFs found
  });

  test('respects max depth limit', async () => {
    // Mock very deep folder structure
    // Assert stops at depth 10
  });
});
```

---

## Potential Gotchas

### 1. File Path Access in Browser

**Problem**: Browsers don't expose full file paths for security reasons.

**Solution Options**:
- Use `file.name` for duplicate detection (may cause false positives)
- Use Tauri's file system API to handle dropped files
- Research if Tauri exposes file paths in drop events
- Consider using file size + name combination for better duplicate detection

**Research Needed**: How does Tauri handle drag-and-drop file paths?

### 2. FileSystemEntry API Browser Support

**Problem**: `webkitGetAsEntry()` is webkit-specific.

**Status**:
- Supported in Chrome, Edge, Safari
- Not in Firefox (uses different API)

**Solution**:
- Document browser requirements
- Add fallback for non-supporting browsers
- Tauri uses Chromium, so should be safe

### 3. Drag/Drop Event Bubbling

**Problem**: Nested elements trigger multiple dragenter/dragleave events.

**Solution**: Use `dragCounter` pattern (implemented in examples above).

### 4. Large Folder Scanning Performance

**Problem**: Scanning a folder with thousands of files could freeze UI.

**Solutions**:
- Show "Scanning..." indicator
- Add timeout (30s max)
- Consider web worker for scanning (more complex)
- Limit max files (e.g., 1000 PDFs max)

### 5. Memory Issues with Large Batches

**Problem**: Processing 500+ files could consume significant memory.

**Solutions**:
- Process file info in batches of 50
- Show progress indicator
- Add "Are you sure?" confirmation for 100+ files

### 6. Duplicate Detection Accuracy

**Problem**: Same filename from different directories might be treated as duplicate.

**Solutions**:
- Use full path when available
- Fallback to name + size combination
- Document limitation

### 7. Drop Zone Z-Index Conflicts

**Problem**: Overlay might interfere with FileGrid interactions.

**Solutions**:
- Use `pointer-events: none` on overlay
- Ensure FileGrid remains interactive when not dragging
- Test click/select functionality thoroughly

### 8. Dark Mode Styling

**Problem**: Overlay colors need to work in both light and dark modes.

**Solution**: Use Tailwind's dark: prefix consistently, test both modes.

---

## Resources

### Documentation
- [HTML5 Drag and Drop API](https://developer.mozilla.org/en-US/docs/Web/API/HTML_Drag_and_Drop_API)
- [FileSystemEntry API](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemEntry)
- [Tauri File System](https://tauri.app/v1/api/js/fs/)
- [Svelte Custom Events](https://svelte.dev/docs#run-time-svelte-createeventdispatcher)

### Existing Codebase References
- `src/lib/components/OCRModule.svelte` - Main OCR UI (lines 480-502 for center panel)
- `src/lib/components/shared/FileGrid.svelte` - Queue display component
- `src/lib/stores/ocrQueue.ts` - Queue state management
- `src-tauri/src/commands.rs` - File handling commands (lines 103-244)
- `CLAUDE.md` - Project overview and patterns

### Similar Implementations
- Look for drag-and-drop examples in Tauri showcase
- Reference Svelte component libraries (Skeleton UI, etc.)

---

## Next Steps

1. **Research Tauri File Paths**: Determine how to get full file paths from dropped files in Tauri
2. **Create DropZone Component**: Start with Phase 1 (basic drag/drop events)
3. **Test Early, Test Often**: Validate each phase before moving to next
4. **Document As You Go**: Update this handoff with learnings and solutions

---

## Questions & Decisions Log

### Open Questions
- [ ] How does Tauri expose file paths in drag/drop events?
- [ ] Should we limit max files in a single drop? (Recommend: 500)
- [ ] Should we show notification for skipped non-PDF files? (Decision: No, per requirements)
- [ ] Should we add confirmation for large batches? (Recommend: Yes, for 100+)

### Decisions Made
- ✅ Always show drop zone (overlay when dragging)
- ✅ Recursively scan folders
- ✅ Skip duplicates silently
- ✅ Block drops during processing
- ✅ Use dragCounter pattern for nested elements
- ✅ Max folder depth: 10 levels
- ✅ Scanning timeout: 30 seconds

---

## Implementation Notes

_Use this section to record discoveries, challenges, and solutions as you implement._

### Date: YYYY-MM-DD - Implementer Name
- **Discovery**: [What you learned]
- **Challenge**: [Problem encountered]
- **Solution**: [How you solved it]

---

**Last Updated**: 2025-11-07
**Status**: Ready for Implementation
**Estimated Effort**: 8-12 hours (depending on Tauri file path research)
