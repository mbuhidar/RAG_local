# Document Processing Deep Dive

## Table of Contents
1. [Overview](#overview)
2. [Data Models](#data-models)
3. [Document Loader](#document-loader)
4. [Document Chunker](#document-chunker)
5. [Supported Formats](#supported-formats)
6. [Best Practices](#best-practices)

---

## Overview

The document processing module (`src/document_processing/`) is responsible for:
1. **Loading** documents from various file formats
2. **Extracting** text content and metadata
3. **Chunking** documents into optimal sizes for retrieval
4. **Preserving** context through overlapping chunks

### Module Structure
```
src/document_processing/
├── __init__.py       # Data models (Document, DocumentChunk)
├── loader.py         # DocumentLoader class
└── chunker.py        # DocumentChunker class
```

---

## Data Models

### Document Class

**Location**: `src/document_processing/__init__.py`

```python
@dataclass
class Document:
    """Document representation with content and metadata."""
    content: str                    # Full text content
    metadata: Dict[str, Any]        # File metadata
    doc_id: Optional[str] = None    # Unique identifier
```

**Purpose**: Represents a complete document loaded from disk.

**Fields**:
- **`content`**: The full text extracted from the file
- **`metadata`**: Dictionary containing:
  - `filename`: Original file name
  - `filepath`: Absolute path to file
  - `file_size`: Size in bytes
  - `created_at`: Creation timestamp (ISO format)
  - `modified_at`: Last modification timestamp
  - `extension`: File format (pdf, docx, txt, md)
  - `num_pages`: Page count (PDF only)
  - `num_paragraphs`: Paragraph count (DOCX only)
- **`doc_id`**: Auto-generated unique ID (if not provided)

**Example**:
```python
Document(
    content="The Commodore 64 is an 8-bit home computer...",
    metadata={
        'filename': 'c64_manual.pdf',
        'filepath': '/home/user/docs/c64_manual.pdf',
        'file_size': 1024576,
        'created_at': '2024-01-15T10:30:00',
        'modified_at': '2024-01-15T10:30:00',
        'extension': 'pdf',
        'num_pages': 150
    },
    doc_id='c64_manual_pdf_abc123'
)
```

---

### DocumentChunk Class

**Location**: `src/document_processing/__init__.py`

```python
@dataclass
class DocumentChunk:
    """Chunk of a document with metadata."""
    content: str                        # Chunk text
    metadata: Dict[str, Any]            # Chunk metadata
    chunk_id: Optional[str] = None      # Unique chunk ID
    parent_doc_id: Optional[str] = None # Parent document ID
```

**Purpose**: Represents a portion of a document optimized for embedding and retrieval.

**Fields**:
- **`content`**: Text content of this chunk (~1000 characters)
- **`metadata`**: Inherits parent document metadata plus:
  - `chunk_index`: Position in document (0, 1, 2, ...)
  - `total_chunks`: Total number of chunks in parent document
  - `chunk_size`: Character count of this chunk
- **`chunk_id`**: Unique identifier (MD5 hash)
- **`parent_doc_id`**: Links back to parent Document

**Example**:
```python
DocumentChunk(
    content="The Commodore 64 has a MOS Technology 6510 CPU...",
    metadata={
        'filename': 'c64_manual.pdf',
        'chunk_index': 5,
        'total_chunks': 147,
        'chunk_size': 987,
        'num_pages': 150,
        # ... other parent metadata
    },
    chunk_id='abc123def456',
    parent_doc_id='c64_manual_pdf_abc123'
)
```

---

## Document Loader

**Location**: `src/document_processing/loader.py`

### Class: DocumentLoader

**Purpose**: Load documents from local filesystem with multi-format support.

### Initialization

```python
def __init__(self, supported_formats: Optional[List[str]] = None):
    """
    Args:
        supported_formats: File extensions to process
                          Default: ['txt', 'pdf', 'md', 'docx']
    """
```

**Usage**:
```python
loader = DocumentLoader(supported_formats=['pdf', 'txt'])
```

---

### Method: `load_file(file_path: Path) -> Optional[Document]`

**Purpose**: Load a single file and return a Document object.

**Process**:
1. Check if file exists
2. Validate file extension
3. Route to format-specific loader
4. Extract content and metadata
5. Return Document object

**Example**:
```python
from pathlib import Path
loader = DocumentLoader()
doc = loader.load_file(Path('data/documents/manual.pdf'))
print(f"Loaded: {doc.metadata['filename']}")
print(f"Pages: {doc.metadata.get('num_pages', 'N/A')}")
print(f"Content length: {len(doc.content)} characters")
```

**Error Handling**:
- Returns `None` if file doesn't exist
- Returns `None` if format unsupported
- Logs errors but doesn't crash
- Catches format-specific parsing errors

---

### Method: `load_directory(directory_path: Path, recursive: bool = True) -> List[Document]`

**Purpose**: Load all supported documents from a directory.

**Parameters**:
- **`directory_path`**: Path to directory
- **`recursive`**: If `True`, searches subdirectories

**Example**:
```python
loader = DocumentLoader()
documents = loader.load_directory(
    Path('data/documents'),
    recursive=True
)
print(f"Loaded {len(documents)} documents")
```

**Behavior**:
- Skips unsupported formats (logs warning)
- Continues on individual file errors
- Returns successfully loaded documents
- Uses glob pattern matching:
  - `recursive=True`: `**/*` (all files in tree)
  - `recursive=False`: `*` (only top level)

---

### Format-Specific Loaders

#### 1. Text Files (`.txt`, `.md`)

**Method**: `_load_text_file(file_path: Path) -> Document`

**Process**:
```python
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
```

**Characteristics**:
- Simple UTF-8 text reading
- Preserves line breaks and formatting
- Works for both plain text and Markdown
- Fast and lightweight

**Error Cases**:
- Encoding errors → falls back to Latin-1
- Binary files → raises exception

---

#### 2. PDF Files (`.pdf`)

**Method**: `_load_pdf_file(file_path: Path) -> Document`

**Dependencies**: Requires `pypdf` library

**Process**:
```python
import pypdf

with open(file_path, 'rb') as f:
    pdf_reader = pypdf.PdfReader(f)
    for page in pdf_reader.pages:
        text = page.extract_text()
        content_parts.append(text)
```

**Features**:
- Page-by-page extraction
- Preserves page breaks (`\n\n` separator)
- Extracts `num_pages` metadata
- Handles multi-page documents efficiently

**Limitations**:
- Cannot extract images or tables perfectly
- Scanned PDFs (image-only) return empty text
- Complex layouts may have text order issues
- Encrypted PDFs may fail

**Metadata Added**:
```python
{
    'num_pages': 150,
    'format': 'pdf'
}
```

---

#### 3. Word Documents (`.docx`)

**Method**: `_load_docx_file(file_path: Path) -> Document`

**Dependencies**: Requires `python-docx` library

**Process**:
```python
from docx import Document as DocxDocument

doc = DocxDocument(file_path)
content_parts = [
    paragraph.text 
    for paragraph in doc.paragraphs 
    if paragraph.text.strip()
]
content = "\n\n".join(content_parts)
```

**Features**:
- Paragraph-by-paragraph extraction
- Skips empty paragraphs
- Preserves paragraph structure
- Extracts `num_paragraphs` metadata

**Limitations**:
- Cannot extract embedded images
- Tables extracted as plain text
- Headers/footers not included
- Comments and tracked changes ignored

**Metadata Added**:
```python
{
    'num_paragraphs': 42,
    'format': 'docx'
}
```

---

### Method: `_extract_file_metadata(file_path: Path) -> Dict`

**Purpose**: Extract filesystem metadata from any file.

**Extracted Information**:
```python
{
    'filename': 'document.pdf',              # File name
    'filepath': '/absolute/path/to/document.pdf',  # Full path
    'file_size': 1048576,                    # Bytes
    'created_at': '2024-01-15T10:30:00',     # ISO timestamp
    'modified_at': '2024-01-16T14:22:00',    # ISO timestamp
    'extension': 'pdf'                       # File type
}
```

**Uses**:
- Source attribution in chat responses
- Filtering by date or size
- Debugging document issues
- Provenance tracking

---

## Document Chunker

**Location**: `src/document_processing/chunker.py`

### Class: DocumentChunker

**Purpose**: Split documents into optimal-sized chunks for embedding and retrieval.

### Why Chunking is Critical

1. **Embedding Limits**: Models have max token limits (~512 tokens)
2. **Retrieval Precision**: Smaller chunks = more specific results
3. **Context Quality**: Right-sized chunks balance detail vs. context
4. **Memory Efficiency**: Smaller embeddings = faster search

**The Chunking Challenge**:
- Too small → loses context
- Too large → imprecise retrieval
- Bad splits → broken sentences

---

### Initialization

```python
def __init__(
    self,
    chunk_size: int = 1000,        # Target characters per chunk
    chunk_overlap: int = 200,       # Overlap between chunks
    separators: Optional[List[str]] = None  # Split priorities
):
```

**Parameters**:

1. **`chunk_size`**: Target maximum characters per chunk
   - Default: `1000` (~250 tokens, ~200 words)
   - Rationale: 
     - Large enough for context
     - Small enough for specificity
     - Fits embedding model limits
   
2. **`chunk_overlap`**: Characters shared between consecutive chunks
   - Default: `200` (20% overlap)
   - Purpose:
     - Prevents sentence splitting
     - Maintains continuity
     - Ensures important context isn't lost at boundaries
   
3. **`separators`**: Priority-ordered list of split points
   - Default: `["\n\n", "\n", " ", ""]`
   - Strategy: Try each separator in order:
     1. `\n\n` - Split on paragraph breaks (ideal)
     2. `\n` - Split on line breaks (good)
     3. ` ` - Split on spaces (acceptable)
     4. `""` - Character-level split (last resort)

**Example Configuration**:
```python
# Default - balanced
chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)

# Smaller chunks - more precise retrieval
chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)

# Larger chunks - more context
chunker = DocumentChunker(chunk_size=2000, chunk_overlap=400)

# Custom separators for code
chunker = DocumentChunker(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\ndef ", "\nclass ", "\n", " "]
)
```

---

### Method: `chunk_document(document: Document) -> List[DocumentChunk]`

**Purpose**: Chunk a single document into multiple DocumentChunk objects.

**Process Flow**:
```
Document
   ↓
Recursive Split (respects separators)
   ↓
Merge Small Chunks (add overlap)
   ↓
Generate Unique IDs
   ↓
Add Chunk Metadata
   ↓
List[DocumentChunk]
```

**Code Example**:
```python
from src.document_processing import Document, DocumentChunker

doc = Document(
    content="Long document text here...",
    metadata={'filename': 'manual.pdf'}
)

chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk_document(doc)

print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk.content)} chars")
```

**Metadata Added to Each Chunk**:
```python
{
    'chunk_index': 0,        # Position in document
    'total_chunks': 17,      # Total chunks in document
    'chunk_size': 987,       # Actual character count
    # ... plus all parent document metadata
}
```

---

### Method: `_recursive_split(text: str) -> List[str]`

**Purpose**: Intelligently split text using separator hierarchy.

**Algorithm**:
```python
def _recursive_split(text):
    1. Try to split with current separator (e.g., "\n\n")
    2. For each split piece:
       a. If < chunk_size → keep it
       b. If >= chunk_size:
          - Try next separator in list
          - If no more separators → force character split
    3. Merge small chunks with overlap
    4. Return final chunks
```

**Example Walkthrough**:

**Input Text** (1500 chars):
```
Paragraph 1 (500 chars)

Paragraph 2 (800 chars)

Paragraph 3 (200 chars)
```

**Step 1**: Split on `\n\n` (paragraph breaks)
- Chunk 1: 500 chars ✓ (under limit)
- Chunk 2: 800 chars ✓ (under limit)
- Chunk 3: 200 chars ✓ (under limit)

**Step 2**: Merge and overlap
- Final Chunk 1: Para 1 + overlap from Para 2 (700 chars)
- Final Chunk 2: Para 2 + overlap from Para 3 (1000 chars)
- Final Chunk 3: Para 3 + overlap from Para 2 (400 chars)

**Why This Works**:
- Respects natural document structure
- Maintains semantic coherence
- Falls back gracefully for difficult text

---

### Method: `_generate_chunk_id(document: Document, index: int) -> str`

**Purpose**: Create unique, deterministic chunk identifiers.

**Implementation**:
```python
def _generate_chunk_id(self, document: Document, index: int) -> str:
    # Combine document ID and chunk index
    unique_string = f"{document.doc_id}_{index}"
    # Generate MD5 hash
    chunk_id = hashlib.md5(unique_string.encode()).hexdigest()
    return chunk_id
```

**Properties**:
- **Unique**: Hash ensures no collisions
- **Deterministic**: Same input = same ID
- **Traceable**: Links back to parent document
- **Short**: 32-character hex string

**Example**:
```python
doc_id = "manual_pdf_abc123"
index = 5
→ "manual_pdf_abc123_5"
→ MD5 hash
→ "7f3e9c1a2b4d8e6f5a9c3b7d1e4f8a2c"
```

**Usage**: Links chunks to parent document for source attribution.

---

### Method: `_merge_chunks(chunks: List[str], separator: str) -> List[str]`

**Purpose**: Combine small chunks and add overlap between chunks.

**Algorithm**:
```python
current_chunk = []
current_length = 0

for chunk in chunks:
    if current_length + len(chunk) <= chunk_size:
        # Add to current chunk
        current_chunk.append(chunk)
        current_length += len(chunk)
    else:
        # Finalize current chunk
        merged.append(separator.join(current_chunk))
        
        # Start new chunk with overlap
        overlap_text = get_last_n_chars(current_chunk, overlap_size)
        current_chunk = [overlap_text, chunk]
        current_length = len(overlap_text) + len(chunk)
```

**Overlap Strategy**:
```
Chunk 1: "The Commodore 64 has a MOS Technology 6510..."
         [............overlap region............]
Chunk 2:              "...6510 CPU running at 1MHz. The VIC-II..."
```

**Benefits**:
- Prevents information loss at boundaries
- Maintains context across chunks
- Improves retrieval quality
- Handles questions that span chunk boundaries

---

### Method: `_force_split(text: str) -> List[str]`

**Purpose**: Last resort - split by character count when no separators work.

**Use Case**: Dense text with no natural break points (e.g., minified code, long URLs)

**Implementation**:
```python
chunks = []
for i in range(0, len(text), chunk_size - chunk_overlap):
    chunks.append(text[i:i + chunk_size])
```

**Example**:
```python
text = "a" * 3000  # 3000 character string with no spaces

# chunk_size=1000, chunk_overlap=200
# Result:
# Chunk 1: chars 0-1000
# Chunk 2: chars 800-1800   (200 char overlap)
# Chunk 3: chars 1600-2600
# Chunk 4: chars 2400-3000
```

---

## Supported Formats

### Format Comparison Table

| Format | Loader | Dependencies | Text Quality | Tables | Images | Metadata |
|--------|--------|--------------|--------------|--------|--------|----------|
| **TXT** | Native | None | Perfect | N/A | N/A | Basic |
| **MD** | Native | None | Perfect | As text | No | Basic |
| **PDF** | pypdf | `pypdf` | Good | Poor | No | Pages |
| **DOCX** | python-docx | `python-docx` | Good | As text | No | Paragraphs |

### When to Use Each Format

**Plain Text (`.txt`, `.md`)**:
- ✅ Simple documents
- ✅ Code files
- ✅ Markdown documentation
- ✅ Fastest processing

**PDF (`.pdf`)**:
- ✅ Published documents
- ✅ Research papers
- ✅ Manuals and guides
- ⚠️ Requires OCR for scanned documents

**Word (`.docx`)**:
- ✅ Office documents
- ✅ Reports and proposals
- ⚠️ Complex formatting may be lost

---

## Best Practices

### 1. **Choosing Chunk Size**

**Small Chunks (500 chars)**:
- More precise retrieval
- Better for fact-finding
- Higher storage overhead
- More chunks to search

**Medium Chunks (1000 chars)** - ✅ Recommended:
- Good balance
- Works for most content types
- Efficient retrieval

**Large Chunks (2000 chars)**:
- More context per chunk
- Better for narrative documents
- Less precise retrieval
- Fewer chunks overall

**Formula**: 
```
Optimal chunk size ≈ 3-4 sentences OR 1-2 paragraphs
```

---

### 2. **Choosing Overlap**

**Overlap Guidelines**:
- 10-20% of chunk_size is typical
- `chunk_overlap = chunk_size * 0.2` (20%)
- Minimum: 50 characters
- Maximum: 50% of chunk_size

**Example**:
```python
# Good balance
chunk_size=1000, chunk_overlap=200  # 20%

# Minimal overlap
chunk_size=1000, chunk_overlap=100  # 10%

# Generous overlap
chunk_size=1000, chunk_overlap=300  # 30%
```

---

### 3. **Custom Separators**

**For Code**:
```python
separators=[
    "\n\n",        # Blank lines between functions
    "\ndef ",      # Function definitions
    "\nclass ",    # Class definitions
    "\n",          # Line breaks
    " "            # Spaces
]
```

**For Technical Docs**:
```python
separators=[
    "\n## ",       # Markdown H2 headers
    "\n### ",      # Markdown H3 headers
    "\n\n",        # Paragraph breaks
    "\n",          # Line breaks
    " "            # Spaces
]
```

**For Q&A / FAQ**:
```python
separators=[
    "\n\nQ:",      # Question markers
    "\n\n",        # Blank lines
    "\n",          # Line breaks
    " "            # Spaces
]
```

---

### 4. **Handling Special Cases**

**Very Large Documents** (>100 pages):
```python
# Use smaller chunks for better precision
chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
```

**Short Documents** (<5 pages):
```python
# Use larger chunks to preserve context
chunker = DocumentChunker(chunk_size=2000, chunk_overlap=400)
```

**Mixed Content** (text + code):
```python
# Use custom separators
chunker = DocumentChunker(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "```", "\ndef ", "\n", " "]
)
```

---

### 5. **Testing Your Chunking Strategy**

```python
# Test script
from src.document_processing import DocumentLoader, DocumentChunker

loader = DocumentLoader()
doc = loader.load_file(Path('test_document.pdf'))

# Try different configurations
configs = [
    (500, 100),
    (1000, 200),
    (2000, 400)
]

for size, overlap in configs:
    chunker = DocumentChunker(chunk_size=size, chunk_overlap=overlap)
    chunks = chunker.chunk_document(doc)
    
    avg_size = sum(len(c.content) for c in chunks) / len(chunks)
    
    print(f"Config: size={size}, overlap={overlap}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Avg size: {avg_size:.0f} chars")
    print(f"  Size range: {min(len(c.content) for c in chunks)} - {max(len(c.content) for c in chunks)}")
    print()
```

---

### 6. **Metadata Best Practices**

**Always Include**:
- `filename` - For source attribution
- `filepath` - For re-reading if needed
- `chunk_index` - For ordering results
- `total_chunks` - For completeness

**Optional but Useful**:
- `page_number` - For PDFs
- `section` - For structured docs
- `author` - For attribution
- `date` - For temporal filtering

**Custom Metadata Example**:
```python
doc = Document(
    content=text,
    metadata={
        'filename': 'c64_manual.pdf',
        'author': 'Commodore',
        'year': 1982,
        'category': 'hardware',
        'language': 'en'
    }
)
```

---

## Performance Considerations

### Loading Speed

| Format | Speed | Notes |
|--------|-------|-------|
| TXT | ⚡️⚡️⚡️ Very Fast | Direct file read |
| MD | ⚡️⚡️⚡️ Very Fast | Direct file read |
| PDF | ⚡️⚡️ Fast | ~1-2 seconds per 100 pages |
| DOCX | ⚡️⚡️ Fast | ~0.5 seconds per document |

### Chunking Speed

- ~1000 chunks/second (typical)
- Linear with document size
- Negligible compared to embedding generation

### Memory Usage

- Each Document: ~1-2x file size in memory
- Each Chunk: ~1KB overhead per chunk
- Large corpus: Consider batch processing

---

## Complete Usage Example

```python
from pathlib import Path
from src.document_processing import DocumentLoader, DocumentChunker

# Initialize
loader = DocumentLoader(supported_formats=['pdf', 'txt', 'md'])
chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)

# Load documents
docs = loader.load_directory(Path('data/documents'), recursive=True)
print(f"Loaded {len(docs)} documents")

# Chunk all documents
all_chunks = []
for doc in docs:
    chunks = chunker.chunk_document(doc)
    all_chunks.extend(chunks)

print(f"Created {len(all_chunks)} total chunks")

# Inspect first chunk
chunk = all_chunks[0]
print(f"\nFirst chunk:")
print(f"  ID: {chunk.chunk_id}")
print(f"  From: {chunk.metadata['filename']}")
print(f"  Index: {chunk.metadata['chunk_index']}/{chunk.metadata['total_chunks']}")
print(f"  Size: {chunk.metadata['chunk_size']} chars")
print(f"  Content preview: {chunk.content[:100]}...")
```

---

## Next Steps

Continue to:
- **[03-EMBEDDINGS-VECTORSTORE.md](03-EMBEDDINGS-VECTORSTORE.md)** - How chunks are converted to embeddings and stored
- **[04-LLM-RAG-PIPELINE.md](04-LLM-RAG-PIPELINE.md)** - How the RAG pipeline uses these chunks
- **[01-ARCHITECTURE-OVERVIEW.md](01-ARCHITECTURE-OVERVIEW.md)** - Return to architecture overview
