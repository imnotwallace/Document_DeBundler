"""
Prompt Templates for LLM-based De-bundling
"""


SPLIT_REFINEMENT_PROMPT = """You are analyzing a potential document boundary in a bundled PDF.

Context: Pages {start_page}-{end_page} are being considered for a split at page {split_page}.

Pages BEFORE split point (pages {before_start}-{before_end}):
{before_text}

Pages AFTER split point (pages {after_start}-{after_end}):
{after_text}

Heuristic signals detected:
{heuristic_signals}

Question: Should there be a document boundary at page {split_page}?

Consider:
- Page numbering patterns
- Header/footer consistency
- Content topic/subject
- Document structure
- Semantic continuity

Answer with ONLY "YES" or "NO", followed by a brief reason (one sentence).

Example: "YES - Page numbering resets and header changes indicate new document"
Example: "NO - Content continues same topic with consistent formatting"

Answer:"""


DOCUMENT_NAMING_PROMPT = """You are generating a filename for a document extracted from a bundled PDF.

Document Segment: Pages {start_page} to {end_page}

First page content:
{first_page_text}

Second page content (if available):
{second_page_text}

Task: Generate a filename in this EXACT format:
{{DATE}}_{{DOCTYPE}}_{{DESCRIPTION}}

Instructions:

1. DATE (YYYY-MM-DD):
   - Find the DOCUMENT DATE (when written/issued), NOT today's date
   - Look for: "Date:", "Dated:", "Issued on:", "Effective Date:", "As of:"
   - If multiple dates, use the earliest/primary date
   - If no date found, use: "UNDATED"

2. DOCTYPE (one word):
   - Choose ONE type: Invoice, Contract, Agreement, Letter, Report, Receipt,
     Form, Certificate, Statement, Memo, Proposal, Notice, Policy, Other
   - Be specific (e.g., "Agreement" not "Document")

3. DESCRIPTION (2-5 words, no special characters):
   - Include: parties, subject matter, identifiers
   - Examples: "Acme Corp Service Agreement", "Q4 Financial Report", "Employee John Smith"
   - Avoid: generic words like "document", "pdf", articles like "the", "a"
   - Keep concise but descriptive

Output ONLY the filename (no quotes, no .pdf extension, no explanation):

Examples:
2024-12-01_Letter_Employment Offer John Smith
2023-06-15_Invoice_Acme Corp June Services
UNDATED_Report_Annual Financial Summary
2024-03-20_Contract_Software License Agreement

Filename:"""


SPLIT_REASONING_SYSTEM_PROMPT = """You are an expert document analyst specializing in identifying document boundaries in bundled PDFs. You analyze page content, structure, and metadata to make accurate split decisions."""


NAMING_SYSTEM_PROMPT = """You are a document classification expert. You extract metadata from documents and generate descriptive, standardized filenames following specific naming conventions."""


def format_split_prompt(
    split_page: int,
    before_pages: list,
    after_pages: list,
    heuristic_signals: list
) -> str:
    """
    Format split refinement prompt with context.

    Args:
        split_page: Page number where split is being considered
        before_pages: List of page dicts before the split (typically last 3)
        after_pages: List of page dicts after the split (typically first 3)
        heuristic_signals: List of signal descriptions

    Returns:
        Formatted prompt string
    """
    # Extract text from last 3 pages before split
    if before_pages:
        before_text = "\n---\n".join([
            f"Page {p['page_num']}: {p['text'][:300]}..."
            for p in before_pages[-3:]  # Last 3 pages before split
        ])
    else:
        before_text = "(No pages before split point)"

    # Extract text from first 3 pages after split
    if after_pages:
        after_text = "\n---\n".join([
            f"Page {p['page_num']}: {p['text'][:300]}..."
            for p in after_pages[:3]  # First 3 pages after split
        ])
    else:
        after_text = "(No pages after split point)"

    # Format signals as bullet list
    signals_text = "\n- ".join(heuristic_signals)

    # Determine page ranges with guards for empty lists
    if before_pages:
        before_start = before_pages[-3]['page_num'] if len(before_pages) >= 3 else before_pages[0]['page_num']
        before_end = before_pages[-1]['page_num']
        start_page = before_pages[0]['page_num']
    else:
        before_start = split_page - 1
        before_end = split_page - 1
        start_page = split_page - 1
    
    if after_pages:
        after_start = after_pages[0]['page_num']
        after_end = after_pages[2]['page_num'] if len(after_pages) >= 3 else after_pages[-1]['page_num']
        end_page = after_pages[-1]['page_num']
    else:
        after_start = split_page
        after_end = split_page
        end_page = split_page

    return SPLIT_REFINEMENT_PROMPT.format(
        start_page=start_page,
        end_page=end_page,
        split_page=split_page,
        before_start=before_start,
        before_end=before_end,
        after_start=after_start,
        after_end=after_end,
        before_text=before_text,
        after_text=after_text,
        heuristic_signals=signals_text
    )


def format_naming_prompt(
    start_page: int,
    end_page: int,
    first_page_text: str,
    second_page_text: str = ""
) -> str:
    """
    Format document naming prompt.

    Args:
        start_page: Starting page number of document
        end_page: Ending page number of document
        first_page_text: Text content of first page
        second_page_text: Text content of second page (optional)

    Returns:
        Formatted prompt string
    """
    return DOCUMENT_NAMING_PROMPT.format(
        start_page=start_page,
        end_page=end_page,
        first_page_text=first_page_text[:2000],  # Limit context
        second_page_text=second_page_text[:1000] if second_page_text else "(Not available)"
    )


def parse_split_decision(response: str) -> tuple[bool, str]:
    """
    Parse LLM response for split decision.

    Args:
        response: Raw LLM response

    Returns:
        Tuple of (should_split, reasoning)
    """
    response = response.strip()

    # Check for YES/NO at start
    upper_response = response.upper()

    if upper_response.startswith('YES'):
        # Extract reasoning (everything after YES and optional dash/hyphen)
        reasoning = response[3:].strip()
        if reasoning.startswith('-') or reasoning.startswith(':'):
            reasoning = reasoning[1:].strip()
        return True, reasoning

    elif upper_response.startswith('NO'):
        # Extract reasoning (everything after NO and optional dash/hyphen)
        reasoning = response[2:].strip()
        if reasoning.startswith('-') or reasoning.startswith(':'):
            reasoning = reasoning[1:].strip()
        return False, reasoning

    else:
        # Unclear response - default to no split
        return False, f"Unclear response: {response}"


def parse_filename(response: str) -> str:
    """
    Parse LLM response for filename.

    Args:
        response: Raw LLM response

    Returns:
        Cleaned filename
    """
    # Get first non-empty line
    filename = response.strip().split('\n')[0].strip()

    # Remove quotes if present
    filename = filename.strip('"\'')

    # Remove .pdf extension if present
    if filename.lower().endswith('.pdf'):
        filename = filename[:-4]

    # Clean up any special characters that shouldn't be in filenames
    # Keep: letters, numbers, spaces, underscores, hyphens
    import re
    filename = re.sub(r'[^\w\s\-]', '', filename)

    # Collapse multiple spaces
    filename = re.sub(r'\s+', ' ', filename)

    return filename.strip()


def validate_filename(filename: str) -> bool:
    """
    Validate that a filename follows the expected format.

    Args:
        filename: Filename to validate

    Returns:
        True if valid format
    """
    import re

    # Expected format: {DATE}_{DOCTYPE}_{DESCRIPTION}
    # DATE: YYYY-MM-DD or UNDATED
    # DOCTYPE: Single word
    # DESCRIPTION: 2-5 words

    parts = filename.split('_', 2)

    if len(parts) != 3:
        return False

    date_part, doctype_part, description_part = parts

    # Validate date
    date_valid = (
        date_part == "UNDATED" or
        re.match(r'^\d{4}-\d{2}-\d{2}$', date_part)
    )

    # Validate doctype (single word)
    doctype_valid = re.match(r'^\w+$', doctype_part)

    # Validate description (2-5 words)
    # Split on both spaces and underscores to count words
    description_words = description_part.strip().replace('_', ' ').split()
    description_valid = 2 <= len(description_words) <= 5

    return date_valid and doctype_valid and description_valid
