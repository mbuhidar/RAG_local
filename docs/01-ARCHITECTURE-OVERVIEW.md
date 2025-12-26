# RAG System Architecture Overview

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Data Flow](#data-flow)
4. [Technology Stack](#technology-stack)
5. [Design Decisions](#design-decisions)

---

## System Architecture

This RAG (Retrieval-Augmented Generation) system is designed to enable intelligent question-answering over your documents using a local LLM, ensuring complete data privacy and no cloud dependencies.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│              (Gradio Web Chat Interface)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Retrieval  │─▶│   Context    │─▶│  Generation  │     │
│  │    Engine    │  │  Aggregation │  │    (LLM)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                  ▼
┌──────────────────┐            ┌──────────────────┐
│  VECTOR STORE    │            │   LOCAL LLM      │
│   (ChromaDB)     │            │    (Ollama)      │
│                  │            │                  │
│  - Embeddings    │            │  - llama3.2      │
│  - Metadata      │            │  - deepseek-r1   │
│  - Similarity    │            │  - No API costs  │
└──────────────────┘            └──────────────────┘
        ▲
        │
┌──────────────────┐
│  DOCUMENT        │
│  PROCESSING      │
│                  │
│  - PDF           │
│  - DOCX          │
│  - TXT/MD        │
└──────────────────┘
```

---

## Core Components

### 1. **Document Processing Module** (`src/document_processing/`)

**Purpose**: Load and prepare documents for indexing.

**Key Components**:
- **`loader.py`**: Multi-format document loader
  - Supports PDF, DOCX, TXT, Markdown
  - Extracts text and metadata (filename, page numbers, dates)
  - Handles encoding and error recovery
  
- **`chunker.py`**: Intelligent text splitting
  - Recursive splitting strategy (paragraphs → sentences → words)
  - Configurable chunk size (default: 1000 characters)
  - Overlap between chunks (default: 200 characters) to preserve context
  - Generates unique chunk IDs using MD5 hashing

**Why This Design**:
- Chunking is critical for RAG - too large and retrieval is imprecise, too small and context is lost
- Overlap prevents important information from being split across boundaries
- Metadata preservation enables source attribution and filtering

---

### 2. **Embedding Generation** (`src/embeddings/`)

**Purpose**: Convert text into numerical vectors for semantic search.

**Key Components**:
- **`generator.py`**: Embedding model wrapper
  - Uses `sentence-transformers/all-MiniLM-L6-v2` (default)
  - 384-dimensional embeddings
  - Auto-detects GPU availability, falls back to CPU
  - Batch processing for efficiency
  - Normalized embeddings for cosine similarity

**Model Choice**:
- **all-MiniLM-L6-v2**: 
  - Fast (~80MB, 14M parameters)
  - Good quality for general text
  - Optimized for semantic similarity
  - Works well on CPU

**Device Management**:
```python
device: "auto"  # Tries GPU, falls back to CPU
```
- Checks `torch.cuda.is_available()`
- Logs which device is being used
- Graceful fallback prevents crashes

---

### 3. **Vector Store** (`src/vector_store/`)

**Purpose**: Store and retrieve document embeddings efficiently.

**Key Components**:
- **`__init__.py`**: Abstract base class defining vector store interface
  - `add_documents()`: Store text with embeddings
  - `similarity_search()`: Find relevant documents
  - `delete_documents()`: Remove by ID
  - `get_collection_stats()`: Metadata about collection
  - `clear_collection()`: Reset database

- **`chroma_store.py`**: ChromaDB implementation
  - Persistent storage (survives restarts)
  - Metadata filtering support
  - Cosine similarity search
  - Efficient for <1M documents

**Why ChromaDB**:
- Easy setup (no complex configuration)
- Persistent by default (data saved to disk)
- Built-in metadata filtering
- Python-native (no separate server required)
- Good performance for typical RAG use cases

**Alternative**: FAISS would be used for >1M documents or when maximum speed is critical.

---

### 4. **LLM Integration** (`src/llm/`)

**Purpose**: Generate natural language responses using local models.

**Key Components**:
- **`__init__.py`**: Abstract LLM interface
  - `generate()`: Single response generation
  - `stream_generate()`: Streaming responses
  - `get_model_info()`: Model metadata

- **`ollama_llm.py`**: Ollama client implementation
  - HTTP API client for Ollama server
  - Support for multiple models (llama3.2, deepseek-r1, etc.)
  - Streaming and non-streaming modes
  - Configurable temperature, max tokens, timeout
  - Connection verification on initialization

**Why Ollama**:
- Easiest setup for local LLMs
- Model management built-in (`ollama pull llama3.2`)
- Efficient inference engine
- Supports many open-source models
- REST API for easy integration

**Configuration**:
```yaml
llm:
  model_name: "llama3.2:latest"  # Your installed model
  temperature: 0.7               # Creativity (0=deterministic, 1=creative)
  max_tokens: 512                # Response length limit
  timeout: 300                   # 5 minute timeout for generation
```

---

### 5. **RAG Pipeline** (`src/rag/`)

**Purpose**: Orchestrate the complete RAG workflow.

**Key Components**:
- **`pipeline.py`**: Main RAG coordinator
  - Document ingestion workflow
  - Query processing pipeline
  - Context retrieval and ranking
  - Response generation with sources
  - Statistics and monitoring

**RAG Workflow**:

#### A. Document Ingestion
```
Documents → Load → Chunk → Embed → Store
```
1. Load documents from directory
2. Split into chunks with overlap
3. Generate embeddings for each chunk
4. Store in vector database with metadata

#### B. Query Processing
```
Query → Embed → Search → Retrieve → Rank → Generate
```
1. Convert query to embedding
2. Search vector store for similar chunks
3. Filter by similarity threshold (default: 0.3)
4. Rank by relevance score
5. Construct prompt with top-k context
6. Generate response using LLM
7. Return answer + source documents

**Key Parameters**:
- `top_k: 5` - Retrieve 5 most relevant chunks
- `score_threshold: 0.3` - Minimum similarity (0-1 scale)
- `max_context_length: 3000` - Token limit for context

**Prompt Template**:
```
Use the following pieces of context to answer the question.
If you don't know the answer, say so.

Context:
{retrieved_chunks}

Question: {user_question}

Answer:
```

---

### 6. **Chat Interface** (`src/chat/`)

**Purpose**: Provide user-friendly web interface for interactions.

**Key Components**:
- **`interface.py`**: Gradio web UI
  - Chat tab with conversation history
  - System info tab showing statistics
  - Instructions tab with usage guide
  - Message processing with error handling
  - Source document display

**Features**:
- **Message Format**: Uses Gradio 6.x message format
  ```python
  {"role": "user", "content": "question"}
  {"role": "assistant", "content": "answer"}
  ```
- **Source Attribution**: Shows which documents were used
- **Real-time Stats**: Document count, model info, configuration
- **Error Handling**: Graceful degradation on failures

**Interface Tabs**:
1. **Chat**: Main conversation interface
2. **System Info**: Statistics (document count, model, device)
3. **Instructions**: How to use the system

---

## Data Flow

### Complete Query Flow

```
1. User enters question in Gradio interface
   ↓
2. Chat interface calls RAGPipeline.query()
   ↓
3. Pipeline generates query embedding
   ↓
4. Vector store performs similarity search
   ↓
5. Results filtered by score threshold
   ↓
6. Top-k most relevant chunks selected
   ↓
7. Prompt constructed with context + question
   ↓
8. LLM generates response
   ↓
9. Response + sources returned to interface
   ↓
10. User sees answer with source attribution
```

### Timing Breakdown (Typical)
- Embedding generation: ~100-200ms (CPU)
- Vector search: ~50-100ms (729 documents)
- LLM generation: 10-60 seconds (depends on response length)
- Total: ~10-60 seconds per query

---

## Technology Stack

### Core Technologies

| Component | Technology | Why This Choice |
|-----------|-----------|-----------------|
| **Language** | Python 3.10+ | Rich ML ecosystem, type hints |
| **Embeddings** | sentence-transformers | Fast, local, good quality |
| **Vector DB** | ChromaDB | Easy setup, persistent, metadata support |
| **LLM** | Ollama | Easiest local LLM setup |
| **Web UI** | Gradio 6.2 | Rapid prototyping, built-in chat components |
| **Document Processing** | PyPDF, python-docx | Format-specific extractors |
| **Config** | YAML | Human-readable, easy to edit |

### Dependencies
```
chromadb==1.3.7           # Vector database
sentence-transformers==5.2.0  # Embeddings
torch==2.5.1              # Deep learning (CPU/CUDA)
transformers==4.57.3      # Transformer models
ollama==0.6.1             # LLM client
gradio==6.2.0             # Web interface
pypdf==6.5.0              # PDF parsing
python-docx==1.2.0        # Word document parsing
pyyaml==6.0.3             # Configuration
```

---

## Design Decisions

### 1. **Why Local-First?**
- **Privacy**: No data sent to cloud APIs
- **Cost**: No per-token API charges
- **Control**: Choose models, update anytime
- **Reliability**: Works offline
- **Speed**: No network latency for LLM calls

### 2. **Why ChromaDB over FAISS?**
- **Persistence**: Built-in disk storage
- **Metadata**: Native metadata filtering
- **Ease**: No separate server needed
- **Scale**: Perfect for <1M documents (typical RAG use case)

**When to use FAISS**: >1M documents, maximum speed critical, willing to manage metadata separately

### 3. **Why sentence-transformers/all-MiniLM-L6-v2?**
- **Size**: Only 80MB (fast download/load)
- **Speed**: Works well on CPU
- **Quality**: Good for general text
- **Proven**: Widely used in production RAG systems

**Alternative**: `all-mpnet-base-v2` for higher quality (420MB, slower)

### 4. **Why Ollama?**
- **Setup**: `ollama pull llama3.2` - that's it!
- **Management**: Built-in model manager
- **Performance**: Optimized inference
- **Variety**: Supports many open-source models

**Alternatives**: 
- llama.cpp for more control
- GPT4All for GUI
- vLLM for maximum throughput

### 5. **Why Gradio over Streamlit?**
- **Chat Components**: Built-in chat UI
- **Speed**: Faster initial load
- **Sharing**: Easy public URLs
- **Async**: Better async support

### 6. **Chunking Strategy**
- **Size**: 1000 chars (~250 tokens)
  - Large enough for context
  - Small enough for specificity
- **Overlap**: 200 chars
  - Prevents split sentences
  - Maintains continuity
- **Separators**: `["\n\n", "\n", " ", ""]`
  - Respects paragraph boundaries
  - Falls back to sentences, then words

### 7. **Retrieval Threshold: 0.3**
- **0.7**: Too strict - misses relevant docs
- **0.5**: Moderate - good balance
- **0.3**: Permissive - catches more relevant content
- **0.0**: Everything - too much noise

**Current choice (0.3)**: Errs on side of recall over precision.

---

## Performance Considerations

### Memory Usage
- **Embedding Model**: ~200MB RAM
- **Vector Store**: ~1MB per 1000 documents
- **LLM**: 2-8GB RAM (depending on model)
- **Total**: ~4-10GB recommended

### Speed Optimization
1. **Batch Embeddings**: Process 32 texts at once
2. **GPU Acceleration**: Auto-detects CUDA
3. **Persistent Cache**: Models cached on disk
4. **Lazy Loading**: Components loaded on-demand

### Scalability
- **Current**: Tested with 729 chunks (~100 pages)
- **Recommended**: Up to 100K chunks (10K pages)
- **Maximum**: 1M chunks with ChromaDB

**Beyond 1M**: Migrate to FAISS or specialized vector DB.

---

## Security & Privacy

### Data Privacy
- ✅ All processing happens locally
- ✅ No external API calls
- ✅ No telemetry by default (ChromaDB telemetry can be disabled)
- ✅ Documents stored only on your machine

### Access Control
- Default: localhost only (`127.0.0.1`)
- Optional: Public sharing via Gradio (`share=true`)
- Recommended: Use behind authentication proxy for production

---

## Next Steps

Continue to:
- **[02-DOCUMENT-PROCESSING.md](02-DOCUMENT-PROCESSING.md)** - Deep dive into document loading and chunking
- **[03-EMBEDDINGS-VECTORSTORE.md](03-EMBEDDINGS-VECTORSTORE.md)** - Embedding generation and vector storage
- **[04-LLM-RAG-PIPELINE.md](04-LLM-RAG-PIPELINE.md)** - LLM integration and RAG orchestration
- **[05-CONFIGURATION-CLI.md](05-CONFIGURATION-CLI.md)** - Configuration system and command-line interface
