# Configuration and CLI Deep Dive

## Table of Contents
1. [Overview](#overview)
2. [Configuration System](#configuration-system)
3. [CLI Commands](#cli-commands)
4. [Environment Variables](#environment-variables)
5. [Configuration Examples](#configuration-examples)
6. [Advanced Usage](#advanced-usage)
7. [Best Practices](#best-practices)

---

## Overview

The RAG system uses YAML-based configuration and a comprehensive CLI for all operations:

```
config/config.yaml  →  RAGPipeline  →  main.py CLI
       ↓                    ↓              ↓
  Settings loaded    Components init   User commands
```

### Key Features

- **YAML Configuration**: Centralized settings in `config/config.yaml`
- **Environment Variables**: Override settings via `.env` file
- **CLI Interface**: Five main commands (ingest, query, chat, stats, clear)
- **Default Values**: Sensible defaults for quick start
- **Validation**: Automatic checking of configuration values

---

## Configuration System

**Location**: `config/config.yaml`

### Configuration Structure

```yaml
vector_db:           # Vector database settings
embeddings:          # Embedding model configuration
document_processing: # Document loading and chunking
llm:                 # Local LLM settings
retrieval:           # Search and retrieval parameters
rag:                 # RAG pipeline configuration
chat:                # Chat interface settings
logging:             # Logging configuration
performance:         # Performance optimizations
```

---

### Vector Database Configuration

```yaml
vector_db:
  provider: "chromadb"
  persist_directory: "./data/vectorstore"
  collection_name: "rag_documents"
  distance_metric: "cosine"
```

**Parameters**:

1. **`provider`**: Vector database backend
   - `"chromadb"` - ChromaDB (✅ Recommended for <1M docs)
   - `"faiss"` - FAISS (for large datasets >1M docs)
   
   **When to Choose**:
   ```yaml
   # Small to medium datasets (<1M documents)
   provider: "chromadb"
   
   # Large datasets (>1M documents, need speed)
   provider: "faiss"
   ```

2. **`persist_directory`**: Where to store vector database
   - Default: `"./data/vectorstore"`
   - Relative or absolute paths supported
   - Created automatically if doesn't exist
   
   **Examples**:
   ```yaml
   persist_directory: "./data/vectorstore"  # Relative
   persist_directory: "/var/rag/vectors"    # Absolute
   persist_directory: "~/Documents/rag_db"  # Home directory
   ```

3. **`collection_name`**: Name of the vector collection
   - Default: `"rag_documents"`
   - Use descriptive names for multiple collections
   
   **Examples**:
   ```yaml
   collection_name: "rag_documents"      # General
   collection_name: "c64_documentation"  # Specific
   collection_name: "company_knowledge"  # Domain-specific
   ```

4. **`distance_metric`**: Similarity calculation method
   - `"cosine"` - Cosine similarity (✅ Recommended)
   - `"l2"` - Euclidean distance
   - `"ip"` - Inner product
   
   **Comparison**:
   ```yaml
   # Best for semantic similarity (text)
   distance_metric: "cosine"
   
   # Best for exact matches (coordinates)
   distance_metric: "l2"
   
   # Best when magnitude matters
   distance_metric: "ip"
   ```

---

### Embeddings Configuration

```yaml
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  model_kwargs:
    device: "auto"
  encode_kwargs:
    normalize_embeddings: true
    batch_size: 32
  cache_folder: "./data/models"
```

**Parameters**:

1. **`model_name`**: Which embedding model to use
   
   **Available Models**:
   ```yaml
   # Fast, general purpose (✅ Recommended)
   model_name: "sentence-transformers/all-MiniLM-L6-v2"
   
   # Better quality, slower
   model_name: "sentence-transformers/all-mpnet-base-v2"
   
   # Multilingual support
   model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
   
   # Domain-specific (code)
   model_name: "microsoft/codebert-base"
   ```
   
   **Selection Guide**:
   | Model | Size | Speed | Quality | Use Case |
   |-------|------|-------|---------|----------|
   | all-MiniLM-L6-v2 | 80MB | Fast | Good | General (✅) |
   | all-mpnet-base-v2 | 420MB | Slower | Better | High quality |
   | paraphrase-multilingual | 420MB | Slower | Good | Multiple languages |

2. **`device`**: Hardware acceleration
   ```yaml
   device: "auto"  # Try GPU, fall back to CPU (✅ Recommended)
   device: "cpu"   # Force CPU
   device: "cuda"  # Force NVIDIA GPU
   device: "mps"   # Force Apple Silicon GPU
   ```
   
   **Performance Impact**:
   ```
   CPU: 200 embeddings/sec
   GPU: 2000 embeddings/sec (10x faster)
   ```

3. **`normalize_embeddings`**: L2 normalization
   ```yaml
   normalize_embeddings: true   # ✅ Recommended (standard practice)
   normalize_embeddings: false  # Only if you have specific reasons
   ```
   
   **Why normalize?**
   - Makes cosine similarity = dot product (faster)
   - Standardizes vector magnitudes
   - Industry standard for semantic search

4. **`batch_size`**: Texts processed per batch
   ```yaml
   # CPU
   batch_size: 16-32  # ✅ Recommended
   
   # GPU (4GB VRAM)
   batch_size: 32-64
   
   # GPU (8GB+ VRAM)
   batch_size: 64-128
   ```

5. **`cache_folder`**: Where to store downloaded models
   ```yaml
   cache_folder: "./data/models"           # Local (✅ Recommended)
   cache_folder: "~/.cache/huggingface"    # User cache (default)
   cache_folder: "/shared/models"          # Shared across users
   ```

---

### Document Processing Configuration

```yaml
document_processing:
  chunk_size: 1000
  chunk_overlap: 200
  separators: ["\n\n", "\n", " ", ""]
  supported_formats: ["txt", "pdf", "md", "docx"]
  extract_metadata: true
```

**Parameters**:

1. **`chunk_size`**: Maximum characters per chunk
   
   **Guidelines**:
   ```yaml
   # Short chunks (precise matching, more chunks)
   chunk_size: 500
   
   # Medium chunks (balanced) ✅ Recommended
   chunk_size: 1000
   
   # Large chunks (more context, fewer chunks)
   chunk_size: 2000
   ```
   
   **Trade-offs**:
   | Size | Precision | Context | Chunks | Best For |
   |------|-----------|---------|--------|----------|
   | 500 | High | Low | Many | Specific facts |
   | 1000 | Medium | Medium | Medium | General use ✅ |
   | 2000 | Low | High | Few | Narrative text |

2. **`chunk_overlap`**: Characters overlapping between chunks
   
   **Recommended Ratios**:
   ```yaml
   # 20% overlap (standard)
   chunk_size: 1000
   chunk_overlap: 200
   
   # 15% overlap (minimum)
   chunk_size: 1000
   chunk_overlap: 150
   
   # 30% overlap (maximum, avoids duplicates)
   chunk_size: 1000
   chunk_overlap: 300
   ```
   
   **Why overlap?**
   - Prevents information split across chunks
   - Ensures context continuity
   - Improves retrieval quality

3. **`separators`**: Splitting priority order
   ```yaml
   # Try to split on paragraphs, then lines, then words
   separators: ["\n\n", "\n", " ", ""]
   
   # More aggressive (sentences)
   separators: ["\n\n", "\n", ". ", " ", ""]
   
   # Code-friendly
   separators: ["\n\n", "\nclass ", "\ndef ", "\n", " "]
   ```

4. **`supported_formats`**: File types to process
   ```yaml
   # Common formats ✅ Recommended
   supported_formats: ["txt", "pdf", "md", "docx"]
   
   # Text only (fastest)
   supported_formats: ["txt", "md"]
   
   # All formats (requires all libraries)
   supported_formats: ["txt", "pdf", "md", "docx", "html", "csv"]
   ```

5. **`extract_metadata`**: Whether to extract file metadata
   ```yaml
   extract_metadata: true   # ✅ Recommended (enables filtering)
   extract_metadata: false  # Faster, no filtering by metadata
   ```

---

### LLM Configuration

```yaml
llm:
  provider: "ollama"
  model_name: "llama3.2:latest"
  base_url: "http://localhost:11434"
  temperature: 0.7
  max_tokens: 512
  streaming: true
  context_window: 4096
```

**Parameters**:

1. **`provider`**: LLM backend
   ```yaml
   provider: "ollama"    # ✅ Recommended (easiest setup)
   provider: "llamacpp"  # For llama.cpp integration
   provider: "gpt4all"   # For GPT4All models
   ```

2. **`model_name`**: Which LLM to use
   
   **Ollama Models**:
   ```yaml
   # Fast, small (2GB) ✅ Recommended for testing
   model_name: "llama3.2:latest"
   
   # Better quality (7B, 4GB RAM)
   model_name: "llama2"
   
   # Best quality (47B, 32GB RAM)
   model_name: "mixtral"
   
   # Specialized for code
   model_name: "codellama"
   
   # Instruction-tuned
   model_name: "mistral"
   ```
   
   **Model Selection**:
   | Model | Size | RAM | Speed | Quality | Use Case |
   |-------|------|-----|-------|---------|----------|
   | llama3.2:latest | 2GB | 4GB | Fast | Good | Testing ✅ |
   | llama2 | 7B | 8GB | Medium | Better | Production |
   | mistral | 7B | 8GB | Medium | Better | Instructions |
   | mixtral | 47B | 32GB | Slow | Best | High quality |
   | codellama | 7B | 8GB | Medium | Good | Code Q&A |

3. **`base_url`**: Ollama server address
   ```yaml
   # Local (default)
   base_url: "http://localhost:11434"
   
   # Remote server
   base_url: "http://192.168.1.100:11434"
   
   # Custom port
   base_url: "http://localhost:8080"
   ```

4. **`temperature`**: Response randomness
   
   **Guidelines**:
   ```yaml
   # Factual Q&A (RAG) ✅ Recommended
   temperature: 0.1-0.3
   
   # Balanced
   temperature: 0.5-0.7
   
   # Creative writing
   temperature: 0.8-1.0
   
   # Deterministic (testing)
   temperature: 0.0
   ```
   
   **Effect Examples**:
   ```
   Temperature 0.1:
   "The Commodore 64 has 64KB of RAM."
   
   Temperature 0.7:
   "The C64 comes with 64 kilobytes of memory."
   
   Temperature 1.0:
   "Equipped with sixty-four thousand bytes of RAM, the Commodore 64..."
   ```

5. **`max_tokens`**: Maximum response length
   ```yaml
   # Concise answers (1 paragraph) ✅ Recommended for chat
   max_tokens: 256-512
   
   # Medium answers (2-3 paragraphs)
   max_tokens: 512-1024
   
   # Detailed answers (1+ pages)
   max_tokens: 1024-2048
   
   # Very long (reports)
   max_tokens: 2048-4096
   ```
   
   **Token Estimation**:
   ```
   256 tokens  ≈ 192 words  ≈ 1 paragraph
   512 tokens  ≈ 384 words  ≈ 2-3 paragraphs
   1024 tokens ≈ 768 words  ≈ 1 page
   2048 tokens ≈ 1536 words ≈ 2-3 pages
   ```

6. **`streaming`**: Enable real-time response
   ```yaml
   streaming: true   # ✅ Recommended for chat (better UX)
   streaming: false  # For batch processing
   ```

7. **`context_window`**: Maximum total context size
   ```yaml
   # Model-dependent (check model specs)
   context_window: 4096    # Llama 2
   context_window: 32768   # Mistral
   context_window: 128000  # Llama 3.2
   ```

---

### Retrieval Configuration

```yaml
retrieval:
  top_k: 5
  score_threshold: 0.3
  use_reranking: false
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  max_context_length: 3000
```

**Parameters**:

1. **`top_k`**: Number of chunks to retrieve
   
   **Guidelines**:
   ```yaml
   # Focused (fast, precise)
   top_k: 3
   
   # Balanced ✅ Recommended
   top_k: 5
   
   # Comprehensive (slow, broad)
   top_k: 10-20
   ```
   
   **Trade-offs**:
   | top_k | Speed | Coverage | Context Size | LLM Cost |
   |-------|-------|----------|--------------|----------|
   | 3 | Fast | Low | Small | Low |
   | 5 | Medium | Medium | Medium | Medium ✅ |
   | 10 | Slow | High | Large | High |

2. **`score_threshold`**: Minimum similarity score
   
   **Guidelines**:
   ```yaml
   # Strict (only highly relevant)
   score_threshold: 0.5-0.7
   
   # Balanced ✅ Recommended
   score_threshold: 0.3-0.5
   
   # Lenient (include marginal matches)
   score_threshold: 0.1-0.3
   ```
   
   **Effect**:
   ```yaml
   # Threshold 0.7: Only perfect matches
   # - Returns: 1-2 chunks (highly relevant)
   # - Risk: May return nothing
   
   # Threshold 0.3: Balanced ✅
   # - Returns: 3-5 chunks (good relevance)
   # - Risk: Minimal
   
   # Threshold 0.1: Very lenient
   # - Returns: 5-10 chunks (mixed relevance)
   # - Risk: Irrelevant context confuses LLM
   ```

3. **`use_reranking`**: Enable reranking of results
   ```yaml
   use_reranking: false  # ✅ Default (faster)
   use_reranking: true   # Better quality (2x slower)
   ```
   
   **What is Reranking?**
   ```
   1. Vector search returns top 20 candidates
   2. Reranker scores each candidate precisely
   3. Return top 5 after reranking
   
   Benefit: More accurate than vector similarity alone
   Cost: Additional model inference (slower)
   ```

4. **`reranker_model`**: Cross-encoder for reranking
   ```yaml
   # Default (fast, good)
   reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
   
   # Better quality (slower)
   reranker_model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
   ```

5. **`max_context_length`**: Maximum total context characters
   ```yaml
   # Prevents exceeding LLM context window
   max_context_length: 3000  # ~2000 tokens
   max_context_length: 6000  # ~4000 tokens
   ```

---

### RAG Pipeline Configuration

```yaml
rag:
  prompt_template: |
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
  include_sources: true
  return_source_documents: true
```

**Parameters**:

1. **`prompt_template`**: How to format prompts
   
   **Default Template**:
   ```yaml
   prompt_template: |
     Use the following pieces of context to answer the question at the end.
     If you don't know the answer, just say that you don't know, don't try to make up an answer.
     
     Context:
     {context}
     
     Question: {question}
     
     Answer:
   ```
   
   **Custom Templates** (see [Prompt Engineering](04-LLM-RAG-PIPELINE.md#prompt-engineering)):
   ```yaml
   # More instructional
   prompt_template: |
     You are an expert assistant. Answer based only on the context provided.
     
     Context: {context}
     Question: {question}
     Answer:
   
   # With examples (few-shot)
   prompt_template: |
     Answer questions using the context.
     
     Example:
     Context: The C64 has 64KB RAM.
     Question: How much memory?
     Answer: 64KB
     
     Your turn:
     Context: {context}
     Question: {question}
     Answer:
   ```

2. **`include_sources`**: Return source information
   ```yaml
   include_sources: true   # ✅ Recommended (transparency)
   include_sources: false  # Only return answer
   ```

3. **`return_source_documents`**: Include full source chunks
   ```yaml
   return_source_documents: true   # Full context
   return_source_documents: false  # Only metadata
   ```

---

### Chat Interface Configuration

```yaml
chat:
  interface: "gradio"
  host: "127.0.0.1"
  port: 7860
  share: false
  enable_history: true
  max_history_length: 10
```

**Parameters**:

1. **`interface`**: UI framework
   ```yaml
   interface: "gradio"     # ✅ Default (easiest)
   interface: "streamlit"  # Alternative
   interface: "fastapi"    # API only
   ```

2. **`host`**: Bind address
   ```yaml
   host: "127.0.0.1"   # ✅ Local only (secure)
   host: "0.0.0.0"     # All interfaces (accessible from network)
   ```

3. **`port`**: Port number
   ```yaml
   port: 7860    # ✅ Default
   port: 8080    # Alternative
   port: 3000    # Custom
   ```

4. **`share`**: Create public Gradio link
   ```yaml
   share: false  # ✅ Default (private)
   share: true   # Public link (for demos)
   ```
   
   **Warning**: `share: true` creates a public URL accessible by anyone!

5. **`enable_history`**: Track conversation history
   ```yaml
   enable_history: true   # ✅ Context-aware chat
   enable_history: false  # Each query independent
   ```

6. **`max_history_length`**: Maximum conversation turns
   ```yaml
   max_history_length: 10   # ✅ Last 10 messages
   max_history_length: 5    # Short memory
   max_history_length: 50   # Long memory
   ```

---

### Logging Configuration

```yaml
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/rag_system.log"
```

**Parameters**:

1. **`level`**: Logging verbosity
   ```yaml
   level: "DEBUG"    # All messages (development)
   level: "INFO"     # ✅ Standard (production)
   level: "WARNING"  # Only warnings/errors
   level: "ERROR"    # Only errors
   ```

2. **`format`**: Log message format
   ```yaml
   # Standard (includes timestamp, logger, level, message)
   format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   
   # Simple
   format: "%(levelname)s: %(message)s"
   
   # Detailed (includes filename and line number)
   format: "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
   ```

3. **`file`**: Log file location
   ```yaml
   file: "./logs/rag_system.log"     # ✅ Default
   file: "/var/log/rag/system.log"   # System logs
   file: null                         # No file logging (console only)
   ```

---

### Performance Configuration

```yaml
performance:
  enable_caching: true
  cache_ttl: 3600
  max_workers: 4
  batch_processing: true
```

**Parameters**:

1. **`enable_caching`**: Cache query results
   ```yaml
   enable_caching: true   # ✅ Faster repeated queries
   enable_caching: false  # Always fresh (development)
   ```

2. **`cache_ttl`**: Cache lifetime (seconds)
   ```yaml
   cache_ttl: 3600    # 1 hour
   cache_ttl: 86400   # 24 hours
   cache_ttl: 300     # 5 minutes
   ```

3. **`max_workers`**: Parallel processing threads
   ```yaml
   max_workers: 4   # ✅ Default (balanced)
   max_workers: 8   # More parallel (high CPU)
   max_workers: 2   # Less parallel (low CPU)
   ```

4. **`batch_processing`**: Enable batching
   ```yaml
   batch_processing: true   # ✅ Faster embedding generation
   batch_processing: false  # Sequential (debugging)
   ```

---

## CLI Commands

**Location**: `main.py`

### Command Structure

```bash
python main.py [--config CONFIG] [--verbose] <command> [command-options]
```

**Global Options**:
- `--config`: Path to config file (default: `config/config.yaml`)
- `--verbose`: Enable debug logging

---

### 1. Ingest Command

**Purpose**: Load and index documents into the vector database.

**Syntax**:
```bash
python main.py ingest [--path PATH] [--recursive]
```

**Options**:
- `--path`: Documents directory (default: `data/documents`)
- `--recursive`: Search subdirectories (default: `true`)

**Examples**:

```bash
# Ingest from default directory
python main.py ingest

# Ingest from custom directory
python main.py ingest --path /home/user/documents

# Ingest without recursion
python main.py ingest --path data/docs --no-recursive

# Ingest with verbose logging
python main.py --verbose ingest --path data/manuals
```

**Output**:
```
2024-12-25 10:30:00 - INFO - Starting document ingestion
2024-12-25 10:30:01 - INFO - Loading documents from data/documents
Loading documents: 100%|████████| 5/5 [00:02<00:00]
2024-12-25 10:30:03 - INFO - Loaded 5 documents
2024-12-25 10:30:03 - INFO - Created 1247 chunks from 5 documents
Generating embeddings: 100%|████████| 39/39 [00:25<00:00]
2024-12-25 10:30:28 - INFO - Successfully ingested 1247 chunks

✓ Successfully ingested 1247 document chunks
```

**What Happens**:
1. Loads documents from specified path
2. Chunks documents (based on config)
3. Generates embeddings (batched)
4. Stores in vector database
5. Persists to disk

---

### 2. Query Command

**Purpose**: Ask a question and get an answer with sources.

**Syntax**:
```bash
python main.py query "<question>" [--show-sources] [--no-show-sources]
```

**Options**:
- `--show-sources`: Display source documents (default: `true`)
- `--no-show-sources`: Hide source documents

**Examples**:

```bash
# Simple query
python main.py query "What is a raster interrupt?"

# Query without sources
python main.py query "How much RAM does the C64 have?" --no-show-sources

# Query with custom config
python main.py --config custom.yaml query "Explain sprite multiplexing"

# Query with verbose logging
python main.py --verbose query "What is the SID chip?"
```

**Output**:
```
2024-12-25 10:35:00 - INFO - Processing query: What is a raster interrupt?
2024-12-25 10:35:01 - INFO - Retrieved 5 relevant chunks (max score: 0.847)

Question: What is a raster interrupt?

Answer: A raster interrupt is a technique used on the Commodore 64 that triggers when the display raster beam reaches a specific screen line. This allows programs to make precise changes to graphics, colors, or sprites mid-screen, enabling advanced visual effects like sprite multiplexing.

--- Sources ---

1. c64_programmers_reference.pdf
   Relevance: 0.847
   Preview: A raster interrupt is triggered when the raster beam reaches a specific line on the screen. This allows precise timing control for grap...

2. c64_programmers_reference.pdf
   Relevance: 0.782
   Preview: The VIC-II chip can generate interrupts at specific raster positions. Programs can change graphics modes, colors, or sprites mid-scre...

3. c64_manual.pdf
   Relevance: 0.723
   Preview: Raster interrupts are commonly used for sprite multiplexing, where more than 8 sprites appear on screen by reusing the same sprite r...
```

---

### 3. Chat Command

**Purpose**: Launch interactive chat interface.

**Syntax**:
```bash
python main.py chat [--host HOST] [--port PORT] [--share]
```

**Options**:
- `--host`: Server address (default: `127.0.0.1`)
- `--port`: Port number (default: `7860`)
- `--share`: Create public Gradio link

**Examples**:

```bash
# Start chat with defaults
python main.py chat

# Custom host and port
python main.py chat --host 0.0.0.0 --port 8080

# Create public link (for demos)
python main.py chat --share

# Chat with custom config
python main.py --config custom.yaml chat
```

**Output**:
```
2024-12-25 10:40:00 - INFO - Starting chat interface
2024-12-25 10:40:02 - INFO - All components initialized successfully
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

**Access**: Open browser to `http://127.0.0.1:7860`

---

### 4. Stats Command

**Purpose**: Display system statistics and configuration.

**Syntax**:
```bash
python main.py stats
```

**Example**:
```bash
python main.py stats
```

**Output**:
```
=== RAG System Statistics ===

Vector Store:
  Collection: rag_documents
  Documents: 1247

Embedding Model:
  Model: sentence-transformers/all-MiniLM-L6-v2
  Dimensions: 384
  Device: cpu

LLM:
  Provider: ollama
  Model: llama3.2:latest
  Base URL: http://localhost:11434
  Temperature: 0.3
```

**Use Cases**:
- Verify configuration
- Check document count
- Diagnose issues
- Monitor system state

---

### 5. Clear Command

**Purpose**: Delete all documents from vector database.

**Syntax**:
```bash
python main.py clear [--confirm]
```

**Options**:
- `--confirm`: Skip confirmation prompt

**Examples**:

```bash
# Clear with confirmation
python main.py clear
# Prompt: Are you sure you want to clear all documents? (yes/no):

# Clear without confirmation
python main.py clear --confirm
```

**Output**:
```
Are you sure you want to clear all documents? (yes/no): yes
2024-12-25 10:45:00 - WARNING - Clearing vector database
✓ Database cleared successfully
```

**Warning**: This is irreversible! Backup data first.

---

## Environment Variables

**Location**: `.env` (create from `.env.example`)

### Available Variables

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest

# Paths
DOCUMENTS_PATH=./data/documents
VECTORSTORE_PATH=./data/vectorstore
MODELS_CACHE_PATH=./data/models
LOGS_PATH=./logs

# Performance
DEVICE=auto  # cpu, cuda, mps, auto
MAX_WORKERS=4
BATCH_SIZE=32

# Optional: API Keys (for cloud services)
# OPENAI_API_KEY=sk-...
# HUGGINGFACE_API_KEY=hf_...
```

### Usage

**Create `.env` file**:
```bash
cp .env.example .env
nano .env  # Edit values
```

**Environment variables override config.yaml**:
```python
# .env file
OLLAMA_MODEL=mistral

# Overrides config.yaml:
# llm:
#   model_name: "llama2"  # Ignored
```

**Loading in Python**:
```python
from dotenv import load_dotenv
import os

load_dotenv()

model = os.getenv("OLLAMA_MODEL", "llama2")  # "mistral"
```

---

## Configuration Examples

### Example 1: Fast Testing Configuration

```yaml
# config/fast.yaml

vector_db:
  persist_directory: "./data/test_vectors"
  collection_name: "test_docs"

embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  model_kwargs:
    device: "cpu"  # Force CPU for consistency
  encode_kwargs:
    batch_size: 16  # Smaller batches

document_processing:
  chunk_size: 500   # Smaller chunks
  chunk_overlap: 50

llm:
  model_name: "llama3.2:latest"  # Fastest model
  temperature: 0.3
  max_tokens: 256    # Short answers
  streaming: false   # Faster for testing

retrieval:
  top_k: 3           # Fewer results
  score_threshold: 0.5

logging:
  level: "DEBUG"     # Verbose output
```

**Usage**:
```bash
python main.py --config config/fast.yaml ingest --path test_docs
python main.py --config config/fast.yaml query "test question"
```

---

### Example 2: Production Configuration

```yaml
# config/production.yaml

vector_db:
  persist_directory: "/var/rag/vectors"
  collection_name: "production_docs"

embeddings:
  model_name: "sentence-transformers/all-mpnet-base-v2"  # Better quality
  model_kwargs:
    device: "cuda"  # Force GPU
  encode_kwargs:
    batch_size: 64  # Larger batches

document_processing:
  chunk_size: 1000
  chunk_overlap: 200

llm:
  model_name: "mistral"  # Better instruction following
  temperature: 0.2       # More factual
  max_tokens: 512
  streaming: true

retrieval:
  top_k: 5
  score_threshold: 0.3
  use_reranking: true    # Better quality

logging:
  level: "INFO"
  file: "/var/log/rag/system.log"

performance:
  enable_caching: true
  cache_ttl: 3600
  max_workers: 8
```

---

### Example 3: Code-Specific Configuration

```yaml
# config/code.yaml

embeddings:
  model_name: "microsoft/codebert-base"  # Code-optimized

document_processing:
  chunk_size: 1500  # Longer for code functions
  separators:
    - "\n\nclass "
    - "\n\ndef "
    - "\n\nfunction "
    - "\n\n"
    - "\n"
  supported_formats: ["py", "js", "java", "cpp", "md"]

llm:
  model_name: "codellama"  # Code-specialized model
  temperature: 0.1         # Deterministic for code
  max_tokens: 1024         # Longer code snippets

rag:
  prompt_template: |
    You are a programming assistant. Answer coding questions using the provided context.
    Include code examples where appropriate.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
```

---

## Advanced Usage

### 1. Multiple Configurations

**Use Case**: Different configs for different projects

```bash
# Project A (vintage computers)
python main.py --config config/vintage.yaml chat

# Project B (company docs)
python main.py --config config/company.yaml chat

# Project C (code analysis)
python main.py --config config/code.yaml chat
```

### 2. Configuration Inheritance

**Create base config**:
```yaml
# config/base.yaml (common settings)
embeddings:
  model_kwargs:
    device: "auto"
  encode_kwargs:
    normalize_embeddings: true
    batch_size: 32

logging:
  level: "INFO"
```

**Extend in specific configs**:
```python
# Load and merge configs
import yaml

with open("config/base.yaml") as f:
    config = yaml.safe_load(f)

with open("config/custom.yaml") as f:
    custom = yaml.safe_load(f)

# Merge (custom overrides base)
config.update(custom)
```

### 3. Programmatic Configuration

```python
from src.rag import RAGPipeline

# Load default config
pipeline = RAGPipeline()

# Override specific settings
pipeline.config['llm']['temperature'] = 0.1
pipeline.config['retrieval']['top_k'] = 10

# Re-initialize LLM with new settings
pipeline.llm = OllamaLLM(
    model_name=pipeline.config['llm']['model_name'],
    temperature=0.1  # New value
)
```

### 4. Config Validation

```python
def validate_config(config):
    """Validate configuration values."""
    
    # Check required keys
    required = ['vector_db', 'embeddings', 'llm', 'retrieval']
    for key in required:
        assert key in config, f"Missing required config: {key}"
    
    # Validate ranges
    assert 0.0 <= config['llm']['temperature'] <= 1.0
    assert config['retrieval']['top_k'] > 0
    assert config['document_processing']['chunk_size'] > 0
    
    # Validate paths
    from pathlib import Path
    Path(config['vector_db']['persist_directory']).mkdir(exist_ok=True)
    
    return True

# Use in initialization
config = yaml.safe_load(open("config/config.yaml"))
validate_config(config)
pipeline = RAGPipeline(config_path="config/config.yaml")
```

---

## Best Practices

### 1. Configuration Management

**Version Control**:
```bash
# Commit configs
git add config/*.yaml

# Don't commit .env (secrets)
echo ".env" >> .gitignore
```

**Documentation**:
```yaml
# Add comments to configs
embeddings:
  model_name: "all-MiniLM-L6-v2"  # Fast, general purpose (80MB)
  # Alternative: "all-mpnet-base-v2" for better quality
```

### 2. Environment-Specific Configs

```
config/
├── config.yaml         # Development (default)
├── production.yaml     # Production settings
├── testing.yaml        # Testing/CI
└── local.yaml          # Personal overrides (gitignored)
```

### 3. Performance Tuning

**Start with defaults**, then optimize:

```yaml
# Step 1: Baseline (defaults)
python main.py stats

# Step 2: Monitor performance
python main.py --verbose query "test"

# Step 3: Tune one setting at a time
# Example: Increase batch size
embeddings:
  encode_kwargs:
    batch_size: 64  # Was 32

# Step 4: Measure improvement
# Repeat until satisfied
```

### 4. Troubleshooting

**Issue**: Slow ingestion
```yaml
# Solutions:
embeddings:
  model_kwargs:
    device: "cuda"      # Use GPU
  encode_kwargs:
    batch_size: 64      # Larger batches
```

**Issue**: Poor retrieval quality
```yaml
# Solutions:
retrieval:
  top_k: 10              # More candidates
  score_threshold: 0.2   # Lower threshold
  use_reranking: true    # Better ranking
```

**Issue**: LLM timeouts
```yaml
# Solutions:
llm:
  max_tokens: 256        # Shorter responses
  model_name: "llama3.2:latest"  # Faster model
```

---

## Summary

**Configuration Files**:
- `config/config.yaml` - Main configuration
- `.env` - Environment variables (optional)

**CLI Commands**:
- `ingest` - Load documents
- `query` - Ask questions
- `chat` - Interactive interface
- `stats` - Show statistics
- `clear` - Reset database

**Key Settings**:
- Vector DB: `persist_directory`, `collection_name`
- Embeddings: `model_name`, `device`, `batch_size`
- LLM: `model_name`, `temperature`, `max_tokens`
- Retrieval: `top_k`, `score_threshold`

---

## Next Steps

Explore other documentation:
- **[01-ARCHITECTURE-OVERVIEW.md](01-ARCHITECTURE-OVERVIEW.md)** - System architecture
- **[02-DOCUMENT-PROCESSING.md](02-DOCUMENT-PROCESSING.md)** - Document handling
- **[03-EMBEDDINGS-VECTORSTORE.md](03-EMBEDDINGS-VECTORSTORE.md)** - Embeddings and search
- **[04-LLM-RAG-PIPELINE.md](04-LLM-RAG-PIPELINE.md)** - LLM integration and RAG workflow
- **[README.md](../README.md)** - Main project documentation
