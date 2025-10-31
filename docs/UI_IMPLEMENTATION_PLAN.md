# UI Implementation Plan: Sam's PDF OCR and (De)Bundling Tool

**Version:** 1.1
**Date:** 2025-10-31
**Last Updated:** 2025-10-31 (Phi-3-mini & PaddleOCR clarifications)
**Status:** Approved - Ready for Implementation

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Decisions](#architecture-decisions)
3. [Project Structure](#project-structure)
4. [Main Menu](#main-menu)
5. [OCR Module](#ocr-module)
6. [De-Bundling Module](#de-bundling-module)
7. [Bundling Module](#bundling-module)
8. [Backend Extensions](#backend-extensions)
9. [Implementation Phases](#implementation-phases)
10. [Dependencies](#dependencies)
11. [Testing Strategy](#testing-strategy)

---

## Overview

Transform the single-screen PDF processing application into a comprehensive multi-module desktop tool with three distinct functional areas:

- **OCR Module**: Batch OCR processing with queue management
- **De-Bundling Module**: Intelligent PDF splitting with LLM-assisted document detection
- **Bundling Module**: Future functionality (placeholder for now)

### Key Features

- Multi-module navigation from central main menu
- Self-contained temporary processing folders
- Real-time progress tracking and verbose logging
- System theme integration (light/dark mode)
- Local LLM integration for document analysis
- Fully offline operation

---

## Architecture Decisions

Based on user requirements and preferences:

| Decision Point | Choice | Rationale |
|---------------|--------|-----------|
| **LLM Backend** | llama.cpp with Phi-3-mini-4k-instruct | Fully offline, 2.3GB model fits in 4GB VRAM with room for concurrent PaddleOCR, excellent instruction-following for metadata extraction |
| **OCR Engine** | PaddleOCR 3.x (library, not server) | Standard library import, models persist in memory during Python subprocess lifetime, no separate server process needed |
| **Theme System** | System preference | Auto-detects OS theme (light/dark), respects user's system settings |
| **OCR Logging** | Combined approach | Structured user messages + Python stderr logs for debugging |
| **PDF Preview** | Side panel | Grid shrinks to 60%, panel slides from right taking 40% width |
| **Navigation** | Component-based routing | Svelte stores for state, conditional rendering in App.svelte |

---

## Project Structure

### New Directory Layout

```
Document-De-Bundler/
├── docs/
│   └── UI_IMPLEMENTATION_PLAN.md          (this file)
├── src/
│   ├── lib/
│   │   ├── components/
│   │   │   ├── MainMenu.svelte            # Landing screen with module buttons
│   │   │   ├── OCRModule.svelte           # OCR batch processing interface
│   │   │   ├── DebundleModule.svelte      # De-bundling wizard
│   │   │   ├── BundleModule.svelte        # Placeholder for future bundling
│   │   │   └── shared/
│   │   │       ├── FileGrid.svelte        # Reusable sortable grid component
│   │   │       ├── Terminal.svelte        # Verbose log output component
│   │   │       ├── PDFPreview.svelte      # PDF viewer panel
│   │   │       ├── ProgressBar.svelte     # Progress indicators
│   │   │       ├── Modal.svelte           # Modal dialog component
│   │   │       └── Button.svelte          # Consistent button styling
│   │   ├── stores/
│   │   │   ├── theme.ts                   # System theme detection & management
│   │   │   ├── navigation.ts              # Current module/route state
│   │   │   ├── ocrQueue.ts                # OCR file queue state management
│   │   │   └── debundleState.ts           # De-bundling wizard state
│   │   └── types/
│   │       └── index.ts                   # TypeScript interfaces
│   ├── App.svelte                          # Root component with routing
│   ├── main.ts
│   └── app.css
├── src-tauri/
│   └── src/
│       ├── main.rs
│       ├── commands.rs                     # Extended with new commands
│       └── python_bridge.rs                # Enhanced IPC handling
├── python-backend/
│   ├── services/
│   │   ├── llm_service.py                 # NEW: llama.cpp integration
│   │   ├── database_service.py            # NEW: SQLite operations
│   │   ├── ocr_batch_service.py           # NEW: Batch OCR processing
│   │   └── (existing services)
│   └── main.py                             # Extended with new commands
├── ocrtempprocessing/                      # NEW: OCR queue temp folder
├── debundletempprocessing/                 # NEW: De-bundle temp folder
├── outputs/
│   ├── ocr/                                # NEW: OCR output folder
│   └── debundle/                           # NEW: De-bundle output folder
└── models/
    └── llm/                                # NEW: LLM model storage
```

### Key Folders

| Folder | Purpose | Lifecycle |
|--------|---------|-----------|
| `ocrtempprocessing/` | Stores PDFs queued for OCR | Files persist between sessions |
| `debundletempprocessing/` | Stores PDF being de-bundled + SQLite DB | Cleared on reset or completion |
| `outputs/ocr/` | Default destination for OCR'd files | User-configurable |
| `outputs/debundle/` | Default destination for split documents | User-configurable |
| `models/llm/` | Stores downloaded LLM models | Persistent, auto-download on first use |

---

## Main Menu

### UI Specification

**Layout:**
```
┌─────────────────────────────────────────────────────┐
│                                                     │
│         Sam's PDF OCR and (De)Bundling Tool        │
│                                                     │
│   ┌───────────────────────────────────────────┐   │
│   │                                           │   │
│   │              OCR Module                   │   │
│   │    (Batch OCR processing & queue)         │   │
│   │                                           │   │
│   └───────────────────────────────────────────┘   │
│                                                     │
│   ┌───────────────────────────────────────────┐   │
│   │                                           │   │
│   │          De-Bundling Module               │   │
│   │    (Split & organize bundled PDFs)        │   │
│   │                                           │   │
│   └───────────────────────────────────────────┘   │
│                                                     │
│   ┌───────────────────────────────────────────┐   │
│   │                                           │   │
│   │          Bundling Module                  │   │
│   │         (Coming Soon - Disabled)          │   │
│   │                                           │   │
│   └───────────────────────────────────────────┘   │
│                                                     │
│                                      [Quit]         │
└─────────────────────────────────────────────────────┘
```

**Component:** `src/lib/components/MainMenu.svelte`

**Elements:**
- **Title**: Large heading at top center
- **Module Buttons**: 3 large cards/buttons (200px height minimum)
  - OCR: Blue theme, enabled
  - De-Bundling: Green theme, enabled
  - Bundling: Gray theme, disabled (opacity 50%, no click)
- **Quit Button**: Bottom right, red/destructive styling

**Navigation:**
- Clicking module button updates `navigationStore` with selected module
- App.svelte conditionally renders corresponding component

**Tauri Commands:**
```rust
#[tauri::command]
pub fn quit_app() -> Result<(), String>
```

---

## OCR Module

### UI Specification

**Layout:**
```
┌─────────────────────────────────────────────────────────────────────────┐
│ [← Return to Main Menu]                                                 │
├──────────────┬────────────────────────────────┬─────────────────────────┤
│ LEFT PANEL   │      CENTER PANEL              │    RIGHT PANEL          │
│ (30%)        │      (40%)                     │    (30%)                │
│              │                                │                         │
│ [Add Files]  │ [Remove from Queue] (if selected) │ ┌─────────────────┐ │
│              │                                │ │ VERBOSE OUTPUT  │ │
│ Destination: │ ┌────────────────────────────┐│ │                 │ │
│ [outputs/ocr]│ │☐│Name│Pages│Size│Status    ││ │ > Processing... │ │
│ [Browse...]  │ │☐│doc1│ 100 │5MB │Pending   ││ │ > Page 1/100... │ │
│              │ │☐│doc2│  50 │2MB │Processing││ │ > OCR complete  │ │
│ [Start OCR]  │ │☐│doc3│ 200 │8MB │Complete  ││ │   for page 1    │ │
│              │ │                            ││ │ ERROR: ...      │ │
│              │ │  (Sortable, scrollable)    ││ │                 │ │
│              │ └────────────────────────────┘│ │ (Auto-scroll)   │ │
│              │                                │ └─────────────────┘ │
└──────────────┴────────────────────────────────┴─────────────────────────┘
```

**Component:** `src/lib/components/OCRModule.svelte`

### Left Panel

**Elements:**

1. **Add Files Button**
   - Opens multi-select file dialog (PDF only)
   - Copies selected files to `./ocrtempprocessing/`
   - Adds files to queue store
   - Extracts metadata (page count, file size)

2. **Destination Folder**
   - Text input showing current destination (default: `./outputs/ocr/`)
   - Browse button opens folder selection dialog
   - Path stored in component state

3. **Start OCR Button**
   - Disabled if queue is empty
   - Triggers batch OCR processing
   - Processes files sequentially
   - Updates status in real-time

4. **Return to Main Menu Button**
   - Top left corner
   - Returns to MainMenu component

### Center Panel - File Queue Grid

**Component:** `src/lib/components/shared/FileGrid.svelte`

**Columns:**

| Column | Width | Sortable | Description |
|--------|-------|----------|-------------|
| Checkbox | 40px | No | Select/deselect for removal |
| File Name | Auto | Yes | Display name only (not full path) |
| Pages | 80px | Yes | Total page count |
| File Size | 100px | Yes | Auto-formatted (KB/MB/GB) |
| Status | 120px | Yes | Pending/Processing/Complete/Failed |

**Features:**
- Header checkbox: Select/deselect all
- Click column header to sort (ascending/descending toggle)
- Vertical scrollbar appears when > 10 rows
- Row hover effect
- "Remove from Queue" button appears when ≥1 item selected
  - Deletes files from `./ocrtempprocessing/`
  - Removes from queue store

**Grid State Management:**
- Svelte store: `ocrQueue.ts`
- Persisted to: `./ocrtempprocessing/queue_state.json`
- On app launch: Scan folder + load state file to populate grid

### Right Panel - Terminal Output

**Component:** `src/lib/components/shared/Terminal.svelte`

**Display:**
- Monospace font (Consolas, Monaco, 'Courier New')
- Dark background with light text
- Auto-scroll to bottom on new messages
- Max 1000 lines (circular buffer)

**Content Types:**
1. **Structured Messages** (user-friendly)
   - "Processing file: document.pdf (1/5)"
   - "OCR page 10/100 - 10% complete"
   - "✓ File complete: document.pdf"

2. **Python Logs** (technical, from stderr)
   - INFO/WARNING/ERROR logs
   - Stack traces on failures
   - Debug information

**Color Coding:**
- INFO: White/Light gray
- WARNING: Yellow
- ERROR: Red
- SUCCESS: Green

### OCR Processing Flow

1. User clicks "Start OCR"
2. Frontend calls Tauri command: `start_batch_ocr(files, destination)`
3. Rust spawns Python process, sends JSON command
4. Python processes files sequentially:
   - For each file:
     - Load PDF with PyMuPDF
     - For each page:
       - Check text layer
       - If no text: Run OCR (PaddleOCR/Tesseract)
       - Extract/OCR'd text
       - Emit progress event
     - Save OCR'd PDF to destination
     - Emit completion event
5. Frontend updates grid status in real-time
6. Terminal displays logs

### PaddleOCR Integration Architecture

**Library Mode (Not Server):**
- PaddleOCR imported as standard Python library: `from paddleocr import PaddleOCR`
- Models load once when OCRService initializes
- Models persist in memory for entire Python subprocess lifetime
- No separate server process required
- Self-contained within desktop application

**Version:** PaddleOCR 3.x
- Breaking changes from 2.x (see CLAUDE.md for details)
- `show_log` parameter removed (use PaddleOCR's logging system)
- `use_mp` parameter not supported in 3.x

**Model Lifecycle:**
```
Tauri starts Python subprocess
    → OCRService.__init__() called
    → PaddleOCR loads models (~10MB, one-time)
    → Models stay in GPU/RAM
    → Process 100s-1000s of pages (no reload)
    → Python subprocess exits
    → Models unloaded from memory
```

**Why Not PaddleOCR Server:**
- Server = separate HTTP/gRPC service process
- Unnecessary complexity for desktop app
- Library mode is simpler and equally performant
- Models still persist in memory during batch processing

### Tauri Commands

```rust
// File selection
#[tauri::command]
pub fn select_multiple_pdf_files() -> Result<Vec<String>, String>

// Folder operations
#[tauri::command]
pub fn select_folder() -> Result<Option<String>, String>

#[tauri::command]
pub fn copy_to_temp_folder(file_path: String, dest_folder: String) -> Result<String, String>

#[tauri::command]
pub fn remove_from_temp_folder(file_path: String) -> Result<(), String>

#[tauri::command]
pub fn scan_temp_folder(folder: String) -> Result<Vec<FileInfo>, String>

// OCR processing
#[tauri::command]
pub async fn start_batch_ocr(
    files: Vec<String>,
    destination: String
) -> Result<(), String>

#[tauri::command]
pub fn cancel_batch_ocr() -> Result<(), String>
```

### Python Backend - OCR Batch Service

**New File:** `python-backend/services/ocr_batch_service.py`

```python
class OCRBatchService:
    def process_batch(self, files: List[str], output_dir: str, progress_callback):
        """Process multiple PDFs with OCR"""
        total_files = len(files)
        logger.info(f"Starting batch OCR for {total_files} files")

        # Initialize OCR service ONCE - models persist in memory
        # This is the key advantage: no reload between files
        ocr = OCRService(gpu=True)

        if not ocr.is_available():
            raise Exception("OCR service not available")

        # NOTE: PaddleOCR models now loaded and stay in memory
        # for the entire batch processing session

        for i, file_path in enumerate(files):
            self.emit_progress(f"Processing file {i+1}/{total_files}: {file_path}")

            with PDFProcessor(file_path) as pdf:
                total_pages = pdf.get_page_count()

                for page_num in range(total_pages):
                    # Check text layer
                    if pdf.has_text_layer(page_num):
                        text = pdf.extract_text(page_num)
                    else:
                        # Models already loaded - fast processing!
                        text = ocr.process_pdf_page(file_path, page_num)

                    # Update page with OCR'd text
                    pdf.update_page_text(page_num, text)

                    self.emit_progress(
                        current=page_num + 1,
                        total=total_pages,
                        message=f"OCR page {page_num+1}/{total_pages}"
                    )

                # Save to output directory
                output_path = os.path.join(output_dir, os.path.basename(file_path))
                pdf.save(output_path)

            self.emit_result("complete", {"file": file_path})

        # Cleanup only after ALL files processed
        ocr.cleanup()
```

**Integration with main.py:**

```python
def handle_ocr_batch(self, command: Dict):
    files = command.get("files", [])
    output_dir = command.get("output_dir")

    service = OCRBatchService()
    service.process_batch(
        files,
        output_dir,
        progress_callback=self.send_progress
    )
```

---

## De-Bundling Module

### Overview

Multi-step wizard for intelligent PDF splitting with LLM assistance.

**Workflow:**
1. File Selection (drag-drop or browse)
2. Boundary Detection (automated analysis)
3. Document Grid Editor (review & adjust)
4. De-bundling Execution (split & save)

### Step 1: File Selection

**Component:** `DebundleModule.svelte` (step 1 state)

**Layout:**
```
┌─────────────────────────────────────────────────────┐
│ [← Return to Main Menu]  [Reset]                    │
│                                                     │
│       Select PDF File for De-Bundling              │
│                                                     │
│   ┌───────────────────────────────────────────┐   │
│   │                                           │   │
│   │         Drag & Drop PDF Here              │   │
│   │               or                          │   │
│   │          [Find File]                      │   │
│   │                                           │   │
│   │   ✓ Selected: large_bundle.pdf (500 MB)   │   │
│   │                                           │   │
│   └───────────────────────────────────────────┘   │
│                                                     │
│          [Detect Individual Document Boundaries]    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Features:**
- Drag-drop zone validates file type (reject non-PDFs)
- "Find File" button opens file picker (PDF filter)
- Selected file copied to `./debundletempprocessing/`
- Display file name and size
- "Detect Boundaries" button enabled after selection

**Tauri Commands:**
```rust
#[tauri::command]
pub fn handle_debundle_file_drop(file_path: String) -> Result<FileInfo, String>
```

### Step 2: Boundary Detection

**UI During Processing:**
```
┌─────────────────────────────────────────────────────┐
│                                                     │
│         Detecting Document Boundaries...           │
│                                                     │
│   ┌─────────────────────────────────────────┐     │
│   │ ████████░░░░░░░░░░░░░░░░░░░░  35%       │     │
│   └─────────────────────────────────────────┘     │
│                                                     │
│   Current Step: Running OCR on page 150/500       │
│                                                     │
│   ┌─────────────────────────────────────────┐     │
│   │ VERBOSE LOG OUTPUT                      │     │
│   │                                         │     │
│   │ > Checking for text layer...            │     │
│   │ > No text layer detected                │     │
│   │ > Starting OCR process...               │     │
│   │ > OCR page 1 complete                   │     │
│   │ > Generating embeddings for page 1      │     │
│   │ > Embedding vector saved to SQLite      │     │
│   │ ...                                     │     │
│   │ (Auto-scrolling terminal)               │     │
│   └─────────────────────────────────────────┘     │
│                                                     │
│                                    [Cancel]         │
└─────────────────────────────────────────────────────┘
```

**Processing Pipeline:**

```
1. Clear Memory/GPU Cache
   ↓
2. Check for Text Layer (PyMuPDF)
   ↓
3. ┌─ Has text layer? ───┐
   │ NO                YES│
   ↓                     ↓
   Run OCR          Skip OCR
   (page-by-page)
   ↓                     ↓
   └─────────────────────┘
   ↓
4. Generate Embeddings (sentence-transformers)
   - Process each page
   - Store in SQLite (embeddings table)
   ↓
5. Heuristic Boundary Detection
   - Calculate embedding similarities
   - Identify large semantic shifts
   - Mark potential boundaries
   ↓
6. LLM Validation (llama.cpp)
   - For uncertain boundaries:
     - Query LLM with surrounding pages
     - "Are pages X-Y a single document?"
   - Update boundary confidence
   ↓
7. Document Naming (LLM)
   - For each detected document:
     - Extract pages [start:end]
     - Query LLM: "Identify: date, type, name"
     - Parse response
     - Store in SQLite (documents table)
   ↓
8. Clear LLM from Memory
   ↓
9. Navigate to Document Grid Editor
```

**SQLite Schema:**

**File:** `./debundletempprocessing/debundle_temp.db`

```sql
-- Store embeddings for similarity analysis
CREATE TABLE embeddings (
    page_number INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    text_preview TEXT
);

-- Store detected documents
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_number INTEGER NOT NULL,
    start_page INTEGER NOT NULL,
    end_page INTEGER NOT NULL,
    document_date TEXT DEFAULT 'undated',  -- ISO: YYYY-MM-DD or 'undated'
    document_type TEXT DEFAULT '',
    document_name TEXT NOT NULL,
    confidence_score REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ensure no overlapping pages
CREATE UNIQUE INDEX idx_page_range ON documents(start_page, end_page);
```

**Embedding Strategy:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim, fast

for page_num in range(total_pages):
    text = extract_text(page_num)
    embedding = model.encode(text)

    db.store_embedding(page_num, embedding, text[:200])
```

**Boundary Detection Algorithm:**

```python
def detect_boundaries(embeddings):
    """Find document boundaries using cosine similarity"""
    boundaries = [0]  # First page always a boundary

    for i in range(1, len(embeddings)):
        # Calculate similarity with previous page
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])

        # Large drop in similarity = boundary
        if similarity < THRESHOLD:  # e.g., 0.7
            boundaries.append(i)

    boundaries.append(len(embeddings))  # Last page
    return boundaries
```

**LLM Integration (llama.cpp):**

**Model:** Llama 3.2 3B (or similar efficient model)

**Prompts:**

1. **Boundary Validation:**
```
System: You are a document analysis assistant.
User: I have pages {start}-{end} from a scanned document bundle.
Based on this text excerpt, are these pages part of the same document?

[Text from pages]

Answer with YES or NO, followed by a brief reason.
```

2. **Document Metadata Extraction:**
```
System: You are a document classification assistant.
User: Analyze this document and provide:
1. Document date (ISO format YYYY-MM-DD, or "undated")
2. Document type (e.g., letter, invoice, report)
3. Suggested file name (concise, descriptive)

[Text from document]

Respond in JSON format:
{
  "date": "YYYY-MM-DD",
  "type": "document_type",
  "name": "Suggested Name"
}
```

**Python Service:** `python-backend/services/llm_service.py`

```python
from llama_cpp import Llama

class LLMService:
    def __init__(self, model_path: str = "./models/llm/Phi-3-mini-4k-instruct-q4_k_m.gguf"):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,  # 4K context window (Phi-3-mini)
            n_gpu_layers=-1  # Use GPU if available
        )

    def validate_boundary(self, text: str, start_page: int, end_page: int) -> bool:
        prompt = f"""Are pages {start_page}-{end_page} a single document?

Text: {text[:1000]}

Answer YES or NO."""

        response = self.llm(prompt, max_tokens=50)
        return "YES" in response["choices"][0]["text"].upper()

    def extract_metadata(self, text: str) -> dict:
        prompt = f"""Analyze this document and respond in JSON:

Text: {text[:2000]}

Format:
{{
  "date": "YYYY-MM-DD or undated",
  "type": "document type",
  "name": "suggested name"
}}"""

        response = self.llm(prompt, max_tokens=200)
        return json.loads(response["choices"][0]["text"])

    def cleanup(self):
        """Free memory"""
        del self.llm
        import gc
        gc.collect()
```

### Step 3: Document Grid Editor

**Layout:**
```
┌──────────────────────────────────────────────────────────────────────────┐
│ [← Return to Main Menu]  [Reset]                                         │
│                                                                          │
│  Destination: [outputs/debundle] [Browse...]  ☐ Retain original order? │
│                                                                          │
│  [Remove Selected] (when items checked)                                 │
│ ┌────────────────────────────────────────────────┐ ┌──────────────────┐│
│ │☐│#│Date      │Type  │Name        │Start│End │  │ │ PDF PREVIEW     ││
│ │─┼─┼──────────┼──────┼────────────┼─────┼────┼──│ │                 ││
│ │☐│1│2025-10-31│Letter│To Daniel   │  1  │ 5  │▶│ │ [Page 1-5]      ││
│ │☐│2│2024-12-20│Report│Annual 2024 │  6  │ 12 │▶│ │                 ││
│ │☐│3│undated   │Memo  │Staff Memo  │ 13  │ 15 │▶│ │ (Shows PDF when ││
│ │               ...                              │ │  row selected)  ││
│ │                                                │ │                 ││
│ │ (Scrollable grid)                             │ │ [Close Preview] ││
│ └────────────────────────────────────────────────┘ └──────────────────┘│
│                                                                          │
│                                                            [De-bundle]   │
└──────────────────────────────────────────────────────────────────────────┘
```

**Grid Columns:**

| Column | Width | Type | Validation | Behavior |
|--------|-------|------|------------|----------|
| ☐ | 40px | Checkbox | N/A | Select for bulk actions |
| # | 60px | Integer | Read-only | Item number (auto-assigned) |
| Date | 120px | Date | ISO YYYY-MM-DD or "undated" | Calendar picker modal |
| Type | 150px | Text | No invalid filename chars | Free text input |
| Name | 250px | Text | No invalid filename chars, required | Free text input |
| Start | 80px | Integer | Must be ≤ End | Adjusts previous row's End |
| End | 80px | Integer | Must be ≥ Start | Adjusts next row's Start |
| Actions | 120px | Buttons | N/A | Preview, Add, Delete, Generate |

**Action Buttons (per row):**

1. **▶ Preview**
   - Opens PDF preview side panel
   - Shows pages [Start:End] for this document
   - Grid shrinks to 60% width
   - Panel takes 40% width on right

2. **➕ Add Row**
   - Inserts new row after current row
   - Default values:
     - Date: "undated"
     - Type: ""
     - Name: "New Document"
     - Start: `previous_end + 1`
     - End: `next_start - 1` (or total pages if last)
   - Renumbers subsequent items

3. **🗑️ Delete Row**
   - Expands next row's Start to `current_start`
   - Removes current row
   - Renumbers subsequent items
   - Confirmation modal if document has many pages

4. **🤖 Generate**
   - Calls LLM with pages [Start:End]
   - Auto-fills Date, Type, Name
   - Shows loading spinner during generation

**Validation Rules:**

1. **Document Date:**
   - ISO format: YYYY-MM-DD
   - Special value: "undated"
   - UI: Calendar modal with "Undated" checkbox

2. **Document Type:**
   - Alphanumeric + spaces, hyphens, underscores
   - Blocked: `< > : " / \ | ? *` (Windows filename restrictions)
   - Can be empty

3. **Document Name:**
   - Same restrictions as Type
   - **Cannot be empty**
   - Max length: 200 characters

4. **Start/End Pages:**
   - Positive integers
   - Start ≤ End
   - First row: Start = 1
   - Last row: End = total_pages
   - No gaps between documents
   - No overlaps

**Auto-Adjustment Logic:**

When user edits **Start Page**:
```javascript
function updateStartPage(rowIndex, newStart) {
    // Update previous row's end
    if (rowIndex > 0) {
        rows[rowIndex - 1].end = newStart - 1;
    }

    // Update current row
    rows[rowIndex].start = newStart;

    // Ensure end is still valid
    if (rows[rowIndex].end < newStart) {
        rows[rowIndex].end = newStart;
    }
}
```

When user edits **End Page**:
```javascript
function updateEndPage(rowIndex, newEnd) {
    // Update current row
    rows[rowIndex].end = newEnd;

    // Update next row's start
    if (rowIndex < rows.length - 1) {
        rows[rowIndex + 1].start = newEnd + 1;
    }

    // Ensure start is still valid
    if (rows[rowIndex].start > newEnd) {
        rows[rowIndex].start = newEnd;
    }
}
```

**PDF Preview Component:**

**File:** `src/lib/components/shared/PDFPreview.svelte`

```svelte
<script>
    export let startPage = 1;
    export let endPage = 1;
    export let onClose;

    let pages = [];

    async function loadPages() {
        pages = await invoke('preview_document_pages', {
            startPage,
            endPage
        });
    }

    $: if (startPage || endPage) loadPages();
</script>

<div class="preview-panel">
    <div class="preview-header">
        <h3>Preview: Pages {startPage}-{endPage}</h3>
        <button on:click={onClose}>✕ Close</button>
    </div>

    <div class="preview-content">
        {#each pages as pageImage, i}
            <div class="page">
                <div class="page-number">Page {startPage + i}</div>
                <img src="data:image/png;base64,{pageImage}" alt="Page {startPage + i}" />
            </div>
        {/each}
    </div>
</div>
```

**Tauri Command for Preview:**
```rust
#[tauri::command]
pub async fn preview_document_pages(
    start_page: i32,
    end_page: i32
) -> Result<Vec<String>, String> {
    // Call Python to render pages as Base64 images
    // Returns vector of Base64 PNG images
}
```

**State Management:**

**Store:** `src/lib/stores/debundleState.ts`

```typescript
import { writable } from 'svelte/store';

export interface DocumentRow {
    id: number;
    itemNumber: number;
    startPage: number;
    endPage: number;
    documentDate: string;
    documentType: string;
    documentName: string;
    confidenceScore: number;
}

export const debundleState = writable({
    currentStep: 'file_selection',  // file_selection | boundary_detection | grid_editor | executing
    selectedFile: null,
    totalPages: 0,
    documents: [] as DocumentRow[],
    destinationFolder: './outputs/debundle',
    retainOrder: true
});
```

### Step 4: De-bundling Execution

**Process Flow:**

1. User clicks "De-bundle" button
2. Validate all rows:
   - All names present
   - No gaps in pages
   - Valid date formats
3. Show confirmation modal with summary
4. Execute splitting:
   ```
   For each row in grid:
       1. Extract pages [start:end] from source PDF
       2. Generate filename based on naming convention
       3. Save to destination folder
       4. Emit progress event
   ```
5. Verify all files written
6. Clear `./debundletempprocessing/` folder
7. Show success modal with "Open Destination Folder" button

**File Naming Convention:**

**With "Retain Order" = TRUE:**
```
{item_number_padded}_{date}_{type}_{name}.pdf

Examples:
001_2025-10-31_letter_Letter to Daniel.pdf
002_2024-12-20_report_Annual Report 2024.pdf
003_undated_memo_Staff Memo.pdf
010_2025-01-15_invoice_Invoice #12345.pdf
```

Padding: Zero-pad to max digits (e.g., 001 for 1-999, 0001 for 1000+)

**With "Retain Order" = FALSE:**
```
{date}_{type}_{name}.pdf

Examples:
2025-10-31_letter_Letter to Daniel.pdf
2024-12-20_report_Annual Report 2024.pdf
undated_memo_Staff Memo.pdf
2025-01-15_invoice_Invoice #12345.pdf
```

**Python Service:** `python-backend/services/debundler.py`

```python
class DebundlerService:
    def execute_split(
        self,
        source_pdf: str,
        documents: List[DocumentRow],
        output_dir: str,
        retain_order: bool
    ):
        """Split PDF into separate documents"""

        with PDFProcessor(source_pdf) as pdf:
            for doc in documents:
                # Extract pages
                pages = pdf.extract_pages(doc.start_page - 1, doc.end_page - 1)

                # Generate filename
                if retain_order:
                    padding = len(str(len(documents)))
                    item_num = str(doc.item_number).zfill(padding)
                    filename = f"{item_num}_{doc.document_date}_{doc.document_type}_{doc.document_name}.pdf"
                else:
                    filename = f"{doc.document_date}_{doc.document_type}_{doc.document_name}.pdf"

                # Sanitize filename
                filename = sanitize_filename(filename)

                # Save
                output_path = os.path.join(output_dir, filename)
                pages.save(output_path)

                # Emit progress
                self.emit_progress(
                    current=doc.item_number,
                    total=len(documents),
                    message=f"Saved: {filename}"
                )

        # Verify all files
        expected_files = len(documents)
        actual_files = len(os.listdir(output_dir))

        if expected_files != actual_files:
            raise Exception(f"File count mismatch: expected {expected_files}, got {actual_files}")

        # Cleanup temp folder
        shutil.rmtree('./debundletempprocessing')
        os.makedirs('./debundletempprocessing')

        return {"status": "success", "files_created": expected_files}
```

**Tauri Commands:**

```rust
#[tauri::command]
pub async fn execute_debundle(
    documents: Vec<DocumentRow>,
    destination: String,
    retain_order: bool
) -> Result<DebundleResult, String>

#[tauri::command]
pub fn open_folder(path: String) -> Result<(), String>

#[tauri::command]
pub fn reset_debundle() -> Result<(), String>
```

---

## Bundling Module

**Component:** `src/lib/components/BundleModule.svelte`

**UI:**
```
┌─────────────────────────────────────────────────────┐
│ [← Return to Main Menu]                             │
│                                                     │
│                                                     │
│            Bundling Module                         │
│                                                     │
│              Coming Soon                           │
│                                                     │
│   This feature will allow you to combine          │
│   multiple PDF documents into a single file.       │
│                                                     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Main Menu Button:**
- Greyed out (opacity 50%)
- Not clickable
- "Coming Soon" badge

---

## Backend Extensions

### Rust Backend (Tauri)

**File:** `src-tauri/src/commands.rs`

**New Commands Summary:**

```rust
// Navigation
pub fn quit_app() -> Result<(), String>

// File/Folder Selection
pub fn select_multiple_pdf_files() -> Result<Vec<String>, String>
pub fn select_folder() -> Result<Option<String>, String>
pub fn open_folder(path: String) -> Result<(), String>

// File Operations
pub fn copy_to_temp_folder(file_path: String, dest: String) -> Result<String, String>
pub fn remove_from_temp_folder(file_path: String) -> Result<(), String>
pub fn scan_temp_folder(folder: String) -> Result<Vec<FileInfo>, String>
pub fn get_pdf_page_count(file_path: String) -> Result<u32, String>

// OCR Module
pub async fn start_batch_ocr(files: Vec<String>, destination: String) -> Result<(), String>
pub fn cancel_batch_ocr() -> Result<(), String>

// De-bundle Module
pub async fn start_debundle_analysis(file_path: String) -> Result<(), String>
pub fn get_debundle_documents() -> Result<Vec<DocumentRow>, String>
pub fn update_document_row(id: i32, data: DocumentRow) -> Result<(), String>
pub fn add_document_row(after_id: i32) -> Result<DocumentRow, String>
pub fn delete_document_row(id: i32) -> Result<(), String>
pub async fn generate_document_metadata(start_page: i32, end_page: i32) -> Result<DocumentMetadata, String>
pub async fn preview_document_pages(start_page: i32, end_page: i32) -> Result<Vec<String>, String>
pub async fn execute_debundle(documents: Vec<DocumentRow>, destination: String, retain_order: bool) -> Result<DebundleResult, String>
pub fn reset_debundle() -> Result<(), String>
```

**Structs:**

```rust
#[derive(Serialize, Deserialize, Clone)]
pub struct FileInfo {
    pub path: String,
    pub name: String,
    pub size: u64,
    pub page_count: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DocumentRow {
    pub id: i32,
    pub item_number: i32,
    pub start_page: i32,
    pub end_page: i32,
    pub document_date: String,
    pub document_type: String,
    pub document_name: String,
    pub confidence_score: f32,
}

#[derive(Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub date: String,
    pub doc_type: String,
    pub name: String,
}

#[derive(Serialize, Deserialize)]
pub struct DebundleResult {
    pub status: String,
    pub files_created: u32,
}
```

**File:** `src-tauri/src/python_bridge.rs`

**Enhancements:**

1. **Separate stdout/stderr handling:**
   ```rust
   pub struct PythonProcess {
       child: Child,
       stdout_handle: JoinHandle<()>,
       stderr_handle: JoinHandle<()>,
   }
   ```

2. **Event streaming:**
   ```rust
   pub fn send_command(&mut self, command: PythonCommand) -> Result<()> {
       let json = serde_json::to_string(&command)?;
       writeln!(self.child.stdin.as_mut().unwrap(), "{}", json)?;
       Ok(())
   }

   pub fn read_event(&mut self) -> Result<PythonEvent> {
       // Read from stdout, parse JSON
   }
   ```

3. **Process lifecycle:**
   ```rust
   pub fn spawn() -> Result<Self>
   pub fn kill(&mut self) -> Result<()>
   pub fn is_alive(&self) -> bool
   ```

### Python Backend

**File:** `python-backend/services/llm_service.py` (NEW)

```python
from llama_cpp import Llama
import json
import logging

logger = logging.getLogger(__name__)

class LLMService:
    """Local LLM integration using llama.cpp"""

    def __init__(self, model_path: str = "./models/llm/Phi-3-mini-4k-instruct-q4_k_m.gguf"):
        self.model_path = model_path
        self.llm = None

    def initialize(self):
        """Load LLM model"""
        logger.info(f"Loading Phi-3-mini model: {self.model_path}")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=4096,  # 4K context window
            n_threads=8,  # CPU threads
            n_gpu_layers=-1,  # Use all GPU layers if available
            verbose=False
        )
        logger.info("Phi-3-mini model loaded successfully")

    def validate_boundary(self, text: str, start_page: int, end_page: int) -> dict:
        """Validate if pages form a single document"""
        prompt = f"""Analyze pages {start_page} to {end_page}.

Are these pages part of the same document?

Text excerpt:
{text[:1500]}

Answer with YES or NO, followed by a one-sentence reason.
Format: YES/NO - reason"""

        response = self.llm(
            prompt,
            max_tokens=100,
            temperature=0.1,
            stop=["\n\n"]
        )

        answer = response["choices"][0]["text"].strip()
        is_same_doc = "YES" in answer.upper()

        return {
            "is_same_document": is_same_doc,
            "reason": answer
        }

    def extract_metadata(self, text: str) -> dict:
        """Extract document metadata using Phi-3-mini"""
        prompt = f"""<|system|>
You are a document analysis assistant. Extract metadata from documents accurately.<|end|>
<|user|>
Analyze this document and provide:
1. Document date (ISO format YYYY-MM-DD, or "undated")
2. Document type (letter, invoice, report, memo, etc.)
3. Concise filename (3-8 words)

Document text:
{text[:2500]}

Respond ONLY with valid JSON:
{{
  "date": "YYYY-MM-DD or undated",
  "type": "document_type",
  "name": "Suggested Filename"
}}<|end|>
<|assistant|>"""

        response = self.llm(
            prompt,
            max_tokens=250,
            temperature=0.1,  # Low temp for structured output
            stop=["<|end|>", "<|user|>"]
        )

        response_text = response["choices"][0]["text"].strip()

        try:
            # Extract JSON from response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_str = response_text[start:end]
            metadata = json.loads(json_str)

            # Validate and clean
            metadata["date"] = metadata.get("date", "undated")
            metadata["type"] = metadata.get("type", "")
            metadata["name"] = metadata.get("name", "Untitled Document")

            return metadata
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM JSON response: {response_text}")
            return {
                "date": "undated",
                "type": "",
                "name": "Document"
            }

    def cleanup(self):
        """Free LLM from memory"""
        logger.info("Unloading LLM model")
        del self.llm
        self.llm = None

        # Force garbage collection
        import gc
        gc.collect()

        # Clear GPU cache if using CUDA
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
```

**File:** `python-backend/services/database_service.py` (NEW)

```python
import sqlite3
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseService:
    """SQLite operations for de-bundling workflow"""

    def __init__(self, db_path: str = "./debundletempprocessing/debundle_temp.db"):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Connect to SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create database schema"""
        cursor = self.conn.cursor()

        # Embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                page_number INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                text_preview TEXT
            )
        """)

        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_number INTEGER NOT NULL,
                start_page INTEGER NOT NULL,
                end_page INTEGER NOT NULL,
                document_date TEXT DEFAULT 'undated',
                document_type TEXT DEFAULT '',
                document_name TEXT NOT NULL,
                confidence_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()

    def store_embedding(self, page_number: int, embedding: np.ndarray, text_preview: str):
        """Store page embedding"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO embeddings (page_number, embedding, text_preview) VALUES (?, ?, ?)",
            (page_number, embedding.tobytes(), text_preview)
        )
        self.conn.commit()

    def get_embedding(self, page_number: int) -> Optional[np.ndarray]:
        """Retrieve page embedding"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT embedding FROM embeddings WHERE page_number = ?", (page_number,))
        row = cursor.fetchone()

        if row:
            return np.frombuffer(row["embedding"], dtype=np.float32)
        return None

    def get_all_embeddings(self) -> List[np.ndarray]:
        """Get all embeddings in order"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT embedding FROM embeddings ORDER BY page_number")
        rows = cursor.fetchall()

        return [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]

    def insert_document(self, doc: Dict) -> int:
        """Insert document boundary"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO documents (item_number, start_page, end_page, document_date, document_type, document_name, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            doc["item_number"],
            doc["start_page"],
            doc["end_page"],
            doc["document_date"],
            doc["document_type"],
            doc["document_name"],
            doc.get("confidence_score", 0.0)
        ))
        self.conn.commit()
        return cursor.lastrowid

    def get_all_documents(self) -> List[Dict]:
        """Get all documents"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY item_number")
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def update_document(self, doc_id: int, updates: Dict):
        """Update document fields"""
        fields = []
        values = []

        for key, value in updates.items():
            fields.append(f"{key} = ?")
            values.append(value)

        values.append(doc_id)

        cursor = self.conn.cursor()
        cursor.execute(
            f"UPDATE documents SET {', '.join(fields)} WHERE id = ?",
            values
        )
        self.conn.commit()

    def delete_document(self, doc_id: int):
        """Delete document"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
```

**File:** `python-backend/services/ocr_batch_service.py` (NEW)

```python
from typing import List, Callable
import os
from .pdf_processor import PDFProcessor
from .ocr_service import OCRService
import logging

logger = logging.getLogger(__name__)

class OCRBatchService:
    """Batch OCR processing for multiple PDFs"""

    def __init__(self, progress_callback: Callable):
        self.progress_callback = progress_callback
        self.cancelled = False

    def process_batch(self, files: List[str], output_dir: str):
        """Process multiple PDF files with OCR"""

        total_files = len(files)
        logger.info(f"Starting batch OCR for {total_files} files")

        # Initialize OCR service once
        ocr = OCRService(gpu=True)

        if not ocr.is_available():
            raise Exception("OCR service not available")

        results = []

        for file_idx, file_path in enumerate(files):
            if self.cancelled:
                logger.info("Batch processing cancelled")
                break

            logger.info(f"Processing file {file_idx + 1}/{total_files}: {file_path}")

            try:
                result = self._process_single_file(
                    file_path,
                    output_dir,
                    ocr,
                    file_idx + 1,
                    total_files
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
                results.append({
                    "file": file_path,
                    "status": "failed",
                    "error": str(e)
                })

        # Cleanup OCR service
        ocr.cleanup()

        return {
            "total": total_files,
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "results": results
        }

    def _process_single_file(
        self,
        file_path: str,
        output_dir: str,
        ocr: OCRService,
        file_num: int,
        total_files: int
    ) -> dict:
        """Process a single PDF file"""

        filename = os.path.basename(file_path)

        self.progress_callback(
            current=file_num,
            total=total_files,
            message=f"Processing: {filename}"
        )

        with PDFProcessor(file_path) as pdf:
            total_pages = pdf.get_page_count()
            logger.info(f"{filename}: {total_pages} pages")

            # Process pages with OCR
            for page_num in range(total_pages):
                if self.cancelled:
                    break

                # Check if page has text layer
                if pdf.has_text_layer(page_num):
                    logger.debug(f"Page {page_num + 1} has text layer, skipping OCR")
                    continue

                # Run OCR
                logger.debug(f"Running OCR on page {page_num + 1}")
                text = ocr.process_pdf_page(file_path, page_num)

                # Update page with OCR'd text
                pdf.update_page_text(page_num, text)

                # Progress update
                self.progress_callback(
                    current=file_num,
                    total=total_files,
                    message=f"{filename}: OCR page {page_num + 1}/{total_pages}"
                )

            # Save OCR'd PDF
            output_path = os.path.join(output_dir, filename)
            pdf.save(output_path)

            logger.info(f"Saved: {output_path}")

        return {
            "file": filename,
            "status": "success",
            "pages": total_pages,
            "output": output_path
        }

    def cancel(self):
        """Cancel batch processing"""
        self.cancelled = True
```

**File:** `python-backend/main.py` (MODIFIED)

**New Command Handlers:**

```python
def handle_ocr_batch(self, command: Dict):
    """Handle batch OCR command"""
    files = command.get("files", [])
    output_dir = command.get("output_dir")

    logger.info(f"OCR batch: {len(files)} files -> {output_dir}")

    try:
        service = OCRBatchService(progress_callback=self.send_progress)
        result = service.process_batch(files, output_dir)

        self.send_result(result)
    except Exception as e:
        logger.error(f"OCR batch failed: {e}", exc_info=True)
        self.send_error(str(e))

def handle_debundle_analyze(self, command: Dict):
    """Handle de-bundle analysis command"""
    file_path = command.get("file_path")

    logger.info(f"De-bundle analysis: {file_path}")

    try:
        # Import services
        from services.embedding_service import EmbeddingService
        from services.split_detection import SplitDetectionService
        from services.llm_service import LLMService
        from services.database_service import DatabaseService

        # Initialize
        db = DatabaseService()
        db.connect()

        embedding_svc = EmbeddingService()
        llm_svc = LLMService()
        split_svc = SplitDetectionService(db, llm_svc)

        # Process
        self.send_progress(0, 100, "Checking text layer...")

        with PDFProcessor(file_path) as pdf:
            total_pages = pdf.get_page_count()

            # OCR if needed
            if not pdf.has_text_layer_all():
                self.send_progress(10, 100, "Running OCR...")
                ocr = OCRService(gpu=True)

                for page_num in range(total_pages):
                    if not pdf.has_text_layer(page_num):
                        text = ocr.process_pdf_page(file_path, page_num)
                        pdf.update_page_text(page_num, text)

                    self.send_progress(
                        10 + int((page_num / total_pages) * 30),
                        100,
                        f"OCR page {page_num + 1}/{total_pages}"
                    )

                ocr.cleanup()

            # Generate embeddings
            self.send_progress(40, 100, "Generating embeddings...")

            for page_num in range(total_pages):
                text = pdf.extract_text(page_num)
                embedding = embedding_svc.encode(text)
                db.store_embedding(page_num, embedding, text[:200])

                self.send_progress(
                    40 + int((page_num / total_pages) * 20),
                    100,
                    f"Embedding page {page_num + 1}/{total_pages}"
                )

            # Detect boundaries
            self.send_progress(60, 100, "Detecting boundaries...")
            llm_svc.initialize()

            boundaries = split_svc.detect_boundaries(total_pages)

            self.send_progress(80, 100, "Generating document metadata...")

            # Generate metadata for each document
            documents = []
            for i, (start, end) in enumerate(boundaries):
                text = ""
                for page in range(start, end + 1):
                    text += pdf.extract_text(page) + "\n"

                metadata = llm_svc.extract_metadata(text[:3000])

                doc = {
                    "item_number": i + 1,
                    "start_page": start + 1,  # 1-indexed
                    "end_page": end + 1,
                    "document_date": metadata["date"],
                    "document_type": metadata["type"],
                    "document_name": metadata["name"],
                    "confidence_score": 0.8
                }

                db.insert_document(doc)
                documents.append(doc)

                self.send_progress(
                    80 + int((i / len(boundaries)) * 20),
                    100,
                    f"Naming document {i + 1}/{len(boundaries)}"
                )

            # Cleanup
            llm_svc.cleanup()
            db.close()

            self.send_progress(100, 100, "Analysis complete")
            self.send_result({
                "status": "success",
                "total_documents": len(documents),
                "documents": documents
            })

    except Exception as e:
        logger.error(f"De-bundle analysis failed: {e}", exc_info=True)
        self.send_error(str(e))

def handle_debundle_execute(self, command: Dict):
    """Handle de-bundle execution command"""
    documents = command.get("documents", [])
    output_dir = command.get("output_dir")
    retain_order = command.get("retain_order", True)
    source_pdf = command.get("source_pdf")

    logger.info(f"De-bundle execute: {len(documents)} documents")

    try:
        from services.bundler import DebundlerService

        service = DebundlerService(progress_callback=self.send_progress)
        result = service.execute_split(
            source_pdf,
            documents,
            output_dir,
            retain_order
        )

        self.send_result(result)
    except Exception as e:
        logger.error(f"De-bundle execution failed: {e}", exc_info=True)
        self.send_error(str(e))
```

---

## Implementation Phases

### Phase 1: Project Structure & Routing (3 hours)

**Tasks:**
1. Create folder structure for components and stores
2. Implement theme detection system
3. Setup navigation store
4. Create routing logic in App.svelte
5. Create shared components (Button, Modal, ProgressBar)

**Deliverables:**
- `src/lib/` folder structure
- `src/lib/stores/theme.ts`
- `src/lib/stores/navigation.ts`
- `src/lib/components/shared/*.svelte`
- Updated `src/App.svelte` with routing

### Phase 2: Main Menu (2 hours)

**Tasks:**
1. Create MainMenu component
2. Style module buttons
3. Implement quit functionality
4. Test navigation between modules

**Deliverables:**
- `src/lib/components/MainMenu.svelte`
- Tauri command: `quit_app()`
- Working navigation

### Phase 3: OCR Module (6 hours)

**Tasks:**
1. Create OCRModule component
2. Implement FileGrid component
3. Implement Terminal component
4. Create OCR queue store
5. Add Tauri commands for file operations
6. Create Python OCRBatchService
7. Integrate Python IPC
8. Test batch processing

**Deliverables:**
- `src/lib/components/OCRModule.svelte`
- `src/lib/components/shared/FileGrid.svelte`
- `src/lib/components/shared/Terminal.svelte`
- `src/lib/stores/ocrQueue.ts`
- Rust commands for OCR module
- `python-backend/services/ocr_batch_service.py`
- Working OCR batch processing

### Phase 4: De-Bundling Module (12 hours)

**Tasks:**

**Step 1-2: File Selection & Analysis (4 hours)**
1. Create DebundleModule component (wizard structure)
2. Implement file drop zone
3. Add Python LLMService
4. Add Python DatabaseService
5. Implement boundary detection pipeline
6. Test embedding generation
7. Test LLM integration

**Step 3: Grid Editor (5 hours)**
1. Implement document grid with inline editing
2. Add validation logic
3. Implement auto-adjustment for page ranges
4. Create PDFPreview component
5. Implement side panel layout
6. Add row action buttons
7. Test grid operations

**Step 4: Execution (3 hours)**
1. Implement file naming logic
2. Create DebundlerService in Python
3. Implement page extraction
4. Add file verification
5. Implement cleanup
6. Test end-to-end workflow

**Deliverables:**
- `src/lib/components/DebundleModule.svelte`
- `src/lib/components/shared/PDFPreview.svelte`
- `src/lib/stores/debundleState.ts`
- `python-backend/services/llm_service.py`
- `python-backend/services/database_service.py`
- Rust commands for de-bundle module
- Updated `python-backend/main.py`
- Working de-bundling workflow

### Phase 5: Bundling Placeholder (30 mins)

**Tasks:**
1. Create BundleModule component
2. Grey out button on main menu
3. Display "Coming Soon" message

**Deliverables:**
- `src/lib/components/BundleModule.svelte`

### Phase 6: Rust Backend Extensions (4 hours)

**Tasks:**
1. Implement all Tauri commands
2. Enhance Python bridge for streaming
3. Add process lifecycle management
4. Implement SQLite wrapper commands
5. Add file system utility commands
6. Test IPC communication

**Deliverables:**
- Updated `src-tauri/src/commands.rs`
- Enhanced `src-tauri/src/python_bridge.rs`
- Registered commands in `main.rs`

### Phase 7: Python Backend Extensions (4 hours)

**Tasks:**
1. Implement LLMService
2. Implement DatabaseService
3. Implement OCRBatchService
4. Add command handlers to main.py
5. Test each service independently
6. Test IPC integration

**Deliverables:**
- `python-backend/services/llm_service.py`
- `python-backend/services/database_service.py`
- `python-backend/services/ocr_batch_service.py`
- Updated `python-backend/main.py`

### Phase 8: Testing & Polish (4 hours)

**Tasks:**
1. Create test PDF files
2. Test OCR module end-to-end
3. Test de-bundling module end-to-end
4. Test error handling
5. Test memory management
6. UI/UX refinements
7. Performance optimization
8. Documentation updates

**Deliverables:**
- Test PDFs
- Bug fixes
- Performance improvements
- Updated documentation

---

## Dependencies

### Python Dependencies

**Add to `python-backend/requirements.txt`:**

```
# Existing dependencies
PyMuPDF>=1.23.0
paddleocr>=3.0.0  # Version 3.x (breaking changes from 2.x)
paddlepaddle>=2.5.0

# New dependencies for de-bundling
llama-cpp-python>=0.2.0
sentence-transformers>=2.2.0
transformers>=4.30.0
torch>=2.0.0
numpy>=1.24.0
```

**Installation:**
```bash
cd python-backend
venv\Scripts\activate
pip install llama-cpp-python sentence-transformers
pip freeze > requirements.txt
```

**PaddleOCR 3.x Note:**
- Breaking changes from 2.x (see CLAUDE.md for details)
- `show_log` parameter removed
- `use_mp` parameter not supported

### Node.js Dependencies

No new dependencies required - using existing Svelte + Tauri ecosystem.

### Model Downloads

**LLM Model (llama.cpp):**
- Model: **Microsoft Phi-3-mini-4k-instruct**
- Format: GGUF (Q4_K_M quantization)
- Size: ~2.3GB
- Download: `https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf`
- File: `Phi-3-mini-4k-instruct-q4_k_m.gguf`
- Download location: `./models/llm/`
- Auto-download on first de-bundle operation or manual download

**PaddleOCR Models:**
- Auto-downloads on first run (~10MB for English)
- Cached in user directory (Windows: `%LOCALAPPDATA%\PaddleOCR\`)
- No manual download required

**Embedding Model (sentence-transformers):**
- Model: `all-MiniLM-L6-v2`
- Size: ~80MB
- Auto-downloads to `~/.cache/huggingface/`

---

## Testing Strategy

### Unit Tests

**Python Services:**
- `test_llm_service.py`: LLM initialization, prompts, cleanup
- `test_database_service.py`: CRUD operations, embedding storage
- `test_ocr_batch_service.py`: Batch processing, cancellation
- `test_debundler_service.py`: File splitting, naming

**Run:**
```bash
cd python-backend
pytest tests/
```

### Integration Tests

**OCR Module:**
1. Add single PDF to queue
2. Add multiple PDFs
3. Remove from queue
4. Start batch processing
5. Cancel processing
6. Verify output files

**De-Bundling Module:**
1. Upload bundled PDF
2. Run analysis
3. Verify detected boundaries
4. Edit document metadata
5. Adjust page ranges
6. Preview documents
7. Execute de-bundling
8. Verify output files

### Manual Testing

**Test PDFs:**
1. **Small (1-10 pages)**: Quick testing
2. **Medium (50-100 pages)**: Realistic bundled documents
3. **Large (500+ pages)**: Performance testing
4. **Text layer**: PDFs with existing text
5. **Scanned**: PDFs requiring OCR
6. **Mixed**: Combination of text and scanned pages

**Scenarios:**
- Empty queue handling
- Invalid file types
- Corrupted PDFs
- Out of memory conditions
- LLM unavailable
- Disk space errors
- Permission errors
- Network offline (for model downloads)

### Performance Benchmarks

**Target Metrics:**
- OCR: < 1 second per page (GPU)
- Embedding: < 0.5 seconds per page
- LLM inference: < 5 seconds per query
- Boundary detection: < 30 seconds for 100-page PDF
- De-bundling execution: < 10 seconds for 10 documents

---

## Error Handling

### User-Facing Errors

**Error Categories:**
1. **File errors**: Invalid file, corrupted PDF, access denied
2. **Processing errors**: OCR failed, LLM unavailable, out of memory
3. **Validation errors**: Invalid metadata, overlapping pages
4. **System errors**: Disk full, permission denied

**Error Display:**
- Modal dialogs for critical errors
- Inline messages for validation errors
- Terminal output for technical details
- Toast notifications for minor warnings

### Recovery Strategies

**OCR Module:**
- Skip failed files, continue with rest
- Retry logic for transient errors
- Fallback to CPU if GPU fails

**De-Bundling Module:**
- Save state to SQLite before LLM calls
- Allow resume after crashes
- Reset button to clear corrupted state

---

## Deployment Considerations

### First-Run Setup

**Auto-created folders:**
```
Document-De-Bundler/
├── ocrtempprocessing/
├── debundletempprocessing/
├── outputs/
│   ├── ocr/
│   └── debundle/
└── models/
    └── llm/
```

**Model downloads:**
- Llama model: Download on first de-bundle operation
- Embedding model: Auto-downloads via sentence-transformers
- Show progress dialog during downloads

### System Requirements

**Minimum:**
- RAM: 8GB system RAM
- Storage: 5GB free (3GB for Phi-3-mini, 2GB for other models/temp files)
- GPU: Optional, 4GB VRAM recommended
- CPU: 4+ cores for decent CPU-only performance

**Recommended (Target Hardware):**
- RAM: 16GB system RAM
- Storage: 10GB free
- GPU: NVIDIA/AMD with 4GB VRAM (perfect for Phi-3-mini + PaddleOCR)
- CPU: 8+ cores

**GPU Notes:**
- 4GB VRAM is optimal for this application
- Phi-3-mini (2.8GB) + PaddleOCR (1GB) = 3.8GB total
- Works well with NVIDIA GTX 1650, RTX 3050, or AMD equivalents
- Sequential processing (OCR first, then LLM) allows safe operation within 4GB

**Memory Budget Breakdown (4GB VRAM):**

**Concurrent Usage During De-Bundling:**
```
Total VRAM: 4GB
├── Phi-3-mini model: ~2.3GB
├── llama.cpp runtime overhead: ~0.5GB
├── Total LLM: ~2.8GB
└── Available for other tasks: ~1.2GB
    └── PaddleOCR (if concurrent): ~1GB ✓ Fits!
```

**Sequential Usage (Recommended):**
```
Phase 1 - OCR (if needed):
├── PaddleOCR models: ~25-75MB base
├── Processing overhead: ~1-1.5GB
└── Total: ~1.5-2GB VRAM

Phase 2 - LLM Analysis:
├── Unload PaddleOCR (optional, or keep loaded)
├── Load Phi-3-mini: ~2.8GB
└── Total: ~2.8GB VRAM
```

---

## Future Enhancements

**Bundling Module:**
- Combine multiple PDFs
- Reorder pages
- Add separators/bookmarks
- Compress output

**OCR Module:**
- Language selection
- OCR quality settings
- Batch rename options
- PDF compression

**De-Bundling Module:**
- Confidence visualization
- Manual boundary drawing
- Batch metadata editing
- Export metadata to CSV

**General:**
- Settings panel (theme, LLM model, OCR engine)
- Command history
- Export logs
- Cloud sync (optional)

---

## Conclusion

This implementation plan provides a comprehensive roadmap for transforming the Document De-Bundler application into a full-featured, multi-module PDF processing tool. The phased approach ensures systematic development with clear milestones and deliverables.

**Key Success Factors:**
1. Self-contained architecture with local processing
2. Robust error handling and recovery
3. Real-time progress feedback
4. Intuitive wizard-based workflows
5. Performance optimization for large files

**Next Steps:**
1. Review and approve this plan
2. Begin Phase 1: Project Structure & Routing
3. Iterate through phases sequentially
4. Test thoroughly after each phase
5. Deploy and gather user feedback

---

**Document Version:** 1.1
**Last Updated:** 2025-10-31 (Phi-3-mini LLM & PaddleOCR 3.x clarifications)
**Status:** ✅ Approved - Ready for Implementation

**Version 1.1 Changes:**
- Updated LLM model from Llama 3.2 3B to **Phi-3-mini-4k-instruct** (2.3GB, optimized for 4GB VRAM)
- Clarified using **PaddleOCR library mode** (not server) for self-contained desktop app
- Added **PaddleOCR 3.x** version notes and breaking changes from 2.x
- Updated **memory budget breakdown** for 4GB VRAM constraint
- Enhanced **Phi-3-mini chat format** prompts for better structured output
- Documented **model persistence** during Python subprocess lifetime
