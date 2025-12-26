# LLM and RAG Pipeline Deep Dive

## Table of Contents
1. [Overview](#overview)
2. [Local LLM Integration](#local-llm-integration)
3. [RAG Pipeline Orchestration](#rag-pipeline-orchestration)
4. [Query Processing Workflow](#query-processing-workflow)
5. [Prompt Engineering](#prompt-engineering)
6. [Complete Examples](#complete-examples)
7. [Best Practices](#best-practices)

---

## Overview

The RAG pipeline combines retrieval and generation to answer questions using local LLMs:

```
User Question
    ↓
Retrieve Context (from vector store)
    ↓
Format Prompt (question + context)
    ↓
Generate Answer (local LLM)
    ↓
Return Response (with sources)
```

### Key Components

1. **OllamaLLM**: Local LLM integration via Ollama API
2. **RAGPipeline**: Orchestrates entire RAG workflow
3. **Prompt Templates**: Structures context and questions
4. **Streaming Support**: Real-time response generation

---

## Local LLM Integration

**Location**: `src/llm/ollama_llm.py`

### Why Ollama?

**Comparison of Local LLM Solutions**:

| Solution | Setup | Models | API | Management |
|----------|-------|--------|-----|------------|
| **Ollama** | ✅ Easy | 50+ models | REST API | Built-in |
| llama.cpp | ⚠️ Complex | Manual download | Python bindings | Manual |
| GPT4All | ✅ Easy | Limited | Python API | Built-in |
| HuggingFace | ⚠️ Complex | All models | Transformers | Manual |

**Ollama Advantages**:
- **One-line installation**: `curl -fsSL https://ollama.com/install.sh | sh`
- **Model management**: `ollama pull llama3.2` (automatic download)
- **REST API**: Easy integration, language-agnostic
- **Model library**: Pre-optimized models (Llama, Mistral, Phi, etc.)
- **GPU support**: Automatic detection and usage
- **Memory management**: Automatic model loading/unloading

---

### Class: OllamaLLM

**Purpose**: Interface to Ollama for local LLM inference.

### Initialization

```python
def __init__(
    self,
    model_name: str = "llama2",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    context_window: int = 4096
):
```

**Parameters**:

1. **`model_name`**: Which Ollama model to use
   
   **Available Models** (selection):
   ```python
   # General purpose
   "llama3.2:latest"     # Meta's Llama 3.2 (2GB)
   "llama3.2:3b"         # Llama 3.2 3B parameters
   "llama2"              # Meta's Llama 2 (7B)
   "mistral"             # Mistral 7B
   "mixtral"             # Mixtral 8x7B (large, powerful)
   
   # Small/fast models
   "phi3"                # Microsoft Phi-3 (3.8GB)
   "gemma2"              # Google Gemma 2B
   "qwen2"               # Alibaba Qwen 2
   
   # Specialized
   "codellama"           # Code generation
   "llama3-groq-tool-use" # Function calling
   "nous-hermes2"        # Instruction following
   ```
   
   **Model Selection Guide**:
   ```python
   # For quick testing (fast, low memory)
   model_name = "llama3.2:latest"  # 2GB, fast
   
   # For better quality (slower, more memory)
   model_name = "llama2"  # 7B, 4GB RAM required
   
   # For best quality (slowest, most memory)
   model_name = "mixtral"  # 8x7B, 32GB RAM required
   
   # For code-related questions
   model_name = "codellama"
   ```

2. **`base_url`**: Ollama server endpoint
   - Default: `"http://localhost:11434"`
   - Custom: `"http://192.168.1.100:11434"` (remote server)
   - Must have Ollama running: `ollama serve`

3. **`temperature`**: Sampling randomness
   - Range: `0.0` to `1.0`
   - `0.0` - Deterministic (same answer every time)
   - `0.3-0.5` - Focused, factual (✅ Recommended for RAG)
   - `0.7-0.9` - Creative, varied
   - `1.0` - Very random

   **Example**:
   ```python
   # For factual Q&A (RAG)
   temperature = 0.3
   
   # For creative writing
   temperature = 0.9
   ```

4. **`max_tokens`**: Maximum response length
   - Default: `2048` tokens (~1500 words)
   - Shorter: `512` tokens (faster, concise)
   - Longer: `4096` tokens (detailed, slower)
   
   **Token Estimation**:
   ```
   1 token ≈ 0.75 words
   512 tokens ≈ 384 words ≈ 1 page
   2048 tokens ≈ 1536 words ≈ 4 pages
   ```

5. **`context_window`**: Maximum context size
   - Model-dependent:
     - Llama 3.2: 128K tokens
     - Llama 2: 4096 tokens
     - Mistral: 32K tokens
   - Limits total size of prompt + response
   - Larger = can include more context

**Example Usage**:
```python
from src.llm import OllamaLLM

# Default configuration
llm = OllamaLLM()

# Production configuration
llm = OllamaLLM(
    model_name="llama3.2:latest",
    base_url="http://localhost:11434",
    temperature=0.3,        # Factual responses
    max_tokens=512,         # Concise answers
    context_window=4096
)
```

---

### Method: `_verify_connection()`

**Purpose**: Ensure Ollama server is running and accessible.

**Process**:
```python
def _verify_connection(self):
    try:
        # Try to reach Ollama API
        response = requests.get(
            f"{self.base_url}/api/tags",
            timeout=5
        )
        response.raise_for_status()
        logger.info("✓ Successfully connected to Ollama server")
    except requests.exceptions.RequestException as e:
        logger.error("✗ Failed to connect to Ollama")
        logger.error("Make sure Ollama is running: ollama serve")
        raise ConnectionError(
            "Could not connect to Ollama. "
            "Please ensure Ollama is running (ollama serve)"
        )
```

**Common Issues**:

```bash
# Issue: Connection refused
# Solution: Start Ollama
ollama serve

# Issue: Model not found
# Solution: Pull the model
ollama pull llama3.2

# Issue: Port in use
# Solution: Check what's using port 11434
lsof -i :11434
```

---

### Method: `generate(prompt: str, ...) -> str`

**Purpose**: Generate text completion (non-streaming).

**Usage**:
```python
llm = OllamaLLM(model_name="llama3.2:latest")

prompt = """Context: The Commodore 64 has 64KB of RAM.

Question: How much memory does the C64 have?

Answer:"""

answer = llm.generate(
    prompt=prompt,
    max_tokens=512,
    temperature=0.3
)

print(answer)
# "The Commodore 64 has 64 kilobytes (64KB) of RAM."
```

**API Request**:
```python
payload = {
    "model": "llama3.2:latest",
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.3,
        "num_predict": 512
    }
}

response = requests.post(
    "http://localhost:11434/api/generate",
    json=payload,
    timeout=300  # 5 minutes
)
```

**Response Format**:
```json
{
  "model": "llama3.2:latest",
  "created_at": "2024-12-25T10:30:00.000Z",
  "response": "The Commodore 64 has 64 kilobytes (64KB) of RAM.",
  "done": true,
  "context": [/* token context */],
  "total_duration": 2500000000,
  "load_duration": 100000000,
  "prompt_eval_count": 50,
  "prompt_eval_duration": 800000000,
  "eval_count": 15,
  "eval_duration": 1600000000
}
```

**Performance Metrics**:
```python
result = response.json()

# Tokens per second
tokens_per_sec = (
    result["eval_count"] / 
    (result["eval_duration"] / 1e9)
)
print(f"Generation speed: {tokens_per_sec:.1f} tok/s")

# Total time
total_time = result["total_duration"] / 1e9
print(f"Total time: {total_time:.2f}s")
```

---

### Method: `stream_generate(prompt: str, ...) -> Iterator[str]`

**Purpose**: Stream text generation in real-time.

**Usage**:
```python
llm = OllamaLLM(model_name="llama3.2:latest")

prompt = "Explain how raster interrupts work on the C64."

print("Response: ", end="", flush=True)
for chunk in llm.stream_generate(prompt):
    print(chunk, end="", flush=True)
print()  # Newline at end

# Output streams in real-time:
# Response: A raster interrupt is a technique...
```

**Streaming Response Format**:
```json
{"model":"llama3.2:latest","response":"A ","done":false}
{"model":"llama3.2:latest","response":"raster ","done":false}
{"model":"llama3.2:latest","response":"interrupt ","done":false}
{"model":"llama3.2:latest","response":"is ","done":false}
...
{"model":"llama3.2:latest","response":"","done":true}
```

**Why Streaming?**
1. **User Experience**: Shows progress immediately
2. **Perceived Speed**: Feels faster than waiting
3. **Early Cancellation**: Can stop if answer is sufficient
4. **Interactive**: Better for chat interfaces

**Gradio Integration**:
```python
def chat_with_streaming(message, history):
    llm = OllamaLLM()
    prompt = format_prompt(message, history)
    
    response = ""
    for chunk in llm.stream_generate(prompt):
        response += chunk
        yield response  # Update UI in real-time
```

---

### Method: `get_model_info() -> Dict[str, Any]`

**Purpose**: Get metadata about the loaded model.

**Returns**:
```python
{
    "provider": "ollama",
    "model_name": "llama3.2:latest",
    "base_url": "http://localhost:11434",
    "temperature": 0.3,
    "max_tokens": 512,
    "context_window": 4096,
    "model_details": {
        "format": "gguf",
        "family": "llama",
        "parameter_size": "2B",
        "quantization_level": "Q4_0"
    }
}
```

**Usage**:
```python
llm = OllamaLLM()
info = llm.get_model_info()

print(f"Model: {info['model_name']}")
print(f"Size: {info['model_details']['parameter_size']}")
print(f"Format: {info['model_details']['format']}")
```

---

### Method: `list_available_models() -> List[str]`

**Purpose**: Get list of all downloaded Ollama models.

**Usage**:
```python
llm = OllamaLLM()
models = llm.list_available_models()

print("Available models:")
for model in models:
    print(f"  - {model}")

# Output:
# Available models:
#   - llama3.2:latest
#   - llama2:7b
#   - mistral:latest
#   - codellama:latest
```

**Equivalent CLI Command**:
```bash
ollama list
```

---

### Method: `pull_model(model_name: str) -> bool`

**Purpose**: Download a model from Ollama registry.

**Usage**:
```python
llm = OllamaLLM()

# Download a model
success = llm.pull_model("llama3.2:latest")

if success:
    print("Model downloaded successfully")
```

**Download Progress**:
```
Pulling model: llama3.2:latest
Pull status: pulling manifest
Pull status: pulling 0a72...
Pull status: pulling 1a21...
Pull status: verifying sha256 digest
Pull status: writing manifest
Pull status: success
Successfully pulled model: llama3.2:latest
```

**Equivalent CLI Command**:
```bash
ollama pull llama3.2:latest
```

---

## RAG Pipeline Orchestration

**Location**: `src/rag/pipeline.py`

### Class: RAGPipeline

**Purpose**: Orchestrate the complete RAG workflow.

### Architecture

```
RAGPipeline
├── DocumentLoader (load files)
├── DocumentChunker (split into chunks)
├── EmbeddingGenerator (text → vectors)
├── ChromaVectorStore (store & retrieve)
└── OllamaLLM (generate answers)
```

### Initialization

```python
def __init__(self, config_path: Optional[str] = None):
```

**Process**:
```python
# 1. Load configuration
config = yaml.safe_load(open(config_path))

# 2. Initialize all components
loader = DocumentLoader(supported_formats=['.pdf', '.txt', ...])
chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
embeddings = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
vector_store = ChromaVectorStore(collection_name="rag_documents")
llm = OllamaLLM(model_name="llama3.2:latest")

# 3. Ready to use
```

**Configuration File** (`config/config.yaml`):
```yaml
document_processing:
  supported_formats: ['.pdf', '.txt', '.md', '.docx']
  chunk_size: 1000
  chunk_overlap: 200

embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  model_kwargs:
    device: "auto"

vector_db:
  collection_name: "rag_documents"
  persist_directory: "./data/vectorstore"
  distance_metric: "cosine"

llm:
  model_name: "llama3.2:latest"
  base_url: "http://localhost:11434"
  temperature: 0.3
  max_tokens: 512

retrieval:
  top_k: 5
  score_threshold: 0.3

rag:
  prompt_template: |
    Context: {context}
    
    Question: {question}
    
    Answer:
```

**Example Usage**:
```python
from src.rag import RAGPipeline

# Initialize with default config
pipeline = RAGPipeline()

# Or custom config
pipeline = RAGPipeline(config_path="custom_config.yaml")
```

---

### Method: `ingest_documents(document_path: str) -> int`

**Purpose**: Load, chunk, embed, and store documents.

**Complete Workflow**:
```python
# 1. Load documents
documents = loader.load_directory("data/documents")
# Result: [Document(content="...", metadata={...}), ...]

# 2. Chunk documents
chunks = chunker.chunk_documents(documents)
# Result: [DocumentChunk(content="...", chunk_id="...", ...), ...]

# 3. Generate embeddings
texts = [chunk.content for chunk in chunks]
embeddings = embedding_generator.embed_texts(texts)
# Result: [[0.1, -0.2, ...], [0.3, 0.1, ...], ...]

# 4. Store in vector database
metadata = [chunk.metadata for chunk in chunks]
ids = [chunk.chunk_id for chunk in chunks]
vector_store.add_documents(texts, embeddings, metadata, ids)

# 5. Persist to disk
vector_store.persist()
```

**Usage Example**:
```python
pipeline = RAGPipeline()

# Ingest all documents from a directory
num_chunks = pipeline.ingest_documents(
    document_path="data/documents",
    recursive=True
)

print(f"Ingested {num_chunks} chunks")
```

**Progress Output**:
```
Starting document ingestion from data/documents
Loading documents: 100%|████████| 2/2 [00:01<00:00]
Loaded 2 documents
Created 729 chunks from 2 documents
Generating embeddings: 100%|████████| 23/23 [00:15<00:00]
Generated 729 embeddings
Stored 729 documents
Successfully ingested 729 chunks
```

---

### Method: `retrieve_context(query: str, ...) -> List[Tuple[str, float, Dict]]`

**Purpose**: Find relevant document chunks for a query.

**Process**:
```python
def retrieve_context(self, query: str, top_k: int = 5):
    # 1. Embed query
    query_embedding = embedding_generator.embed_text(query)
    
    # 2. Search vector store
    results = vector_store.similarity_search(
        query_embedding=query_embedding,
        k=top_k
    )
    
    # 3. Filter by score threshold
    filtered = [
        (text, score, meta)
        for text, score, meta in results
        if score >= score_threshold
    ]
    
    return filtered
```

**Usage**:
```python
pipeline = RAGPipeline()

query = "What is a raster interrupt?"
results = pipeline.retrieve_context(query, top_k=3)

for text, score, metadata in results:
    print(f"Score: {score:.3f}")
    print(f"Source: {metadata['filename']}")
    print(f"Text: {text[:100]}...")
    print()
```

**Output**:
```
Retrieved 3 relevant chunks (max score: 0.847, threshold: 0.3)

Score: 0.847
Source: c64_programmers_reference.pdf
Text: A raster interrupt is triggered when the raster beam reaches...

Score: 0.782
Source: c64_programmers_reference.pdf
Text: The VIC-II chip can generate interrupts at specific raster...

Score: 0.723
Source: c64_manual.pdf
Text: Raster interrupts allow precise timing control for graphics...
```

---

### Method: `generate_response(query: str, context: List[str]) -> str`

**Purpose**: Generate answer using LLM with retrieved context.

**Prompt Formatting**:
```python
# Template from config
template = """Context: {context}

Question: {question}

Answer:"""

# Fill in template
context_text = "\n\n".join(context)
prompt = template.format(
    context=context_text,
    question=query
)
```

**Complete Prompt Example**:
```
Context: A raster interrupt is triggered when the raster beam reaches a specific line on the screen. This allows precise timing control for graphics effects.

The VIC-II chip can generate interrupts at specific raster positions. Programs can change graphics modes, colors, or sprites mid-screen.

Raster interrupts are commonly used for sprite multiplexing, where more than 8 sprites appear on screen by reusing the same sprite registers.

Question: What is a raster interrupt?

Answer:
```

**LLM Response**:
```
A raster interrupt is a technique used on the Commodore 64 that triggers when the display raster beam reaches a specific screen line. This allows programs to make precise changes to graphics, colors, or sprites mid-screen, enabling advanced visual effects like sprite multiplexing.
```

**Usage**:
```python
pipeline = RAGPipeline()

query = "What is a raster interrupt?"
contexts = [
    "A raster interrupt is triggered when...",
    "The VIC-II chip can generate interrupts...",
    "Raster interrupts are commonly used for..."
]

answer = pipeline.generate_response(query, contexts)
print(answer)
```

---

### Method: `query(question: str, ...) -> Dict[str, Any]`

**Purpose**: Complete RAG pipeline - retrieve + generate.

**Complete Workflow**:
```python
def query(self, question: str) -> Dict[str, Any]:
    # 1. Retrieve relevant context
    results = self.retrieve_context(question)
    
    # 2. Extract text and metadata
    contexts = [text for text, _, _ in results]
    sources = [
        {
            "text": text[:200],
            "score": score,
            "metadata": meta
        }
        for text, score, meta in results
    ]
    
    # 3. Generate answer
    answer = self.generate_response(question, contexts)
    
    # 4. Return result with sources
    return {
        "answer": answer,
        "sources": sources
    }
```

**Usage Example**:
```python
pipeline = RAGPipeline()

result = pipeline.query("What is a raster interrupt?")

print("Answer:")
print(result["answer"])
print("\nSources:")
for i, source in enumerate(result["sources"], 1):
    print(f"{i}. Score: {source['score']:.3f}")
    print(f"   File: {source['metadata']['filename']}")
    print(f"   Text: {source['text']}")
    print()
```

**Output**:
```
Answer:
A raster interrupt is a technique used on the Commodore 64 that triggers when the display raster beam reaches a specific screen line. This allows programs to make precise changes to graphics, colors, or sprites mid-screen, enabling advanced visual effects like sprite multiplexing.

Sources:
1. Score: 0.847
   File: c64_programmers_reference.pdf
   Text: A raster interrupt is triggered when the raster beam reaches a specific line on the screen. This allows precise timing control for graphics effects.

2. Score: 0.782
   File: c64_programmers_reference.pdf
   Text: The VIC-II chip can generate interrupts at specific raster positions. Programs can change graphics modes, colors, or sprites mid-screen.

3. Score: 0.723
   File: c64_manual.pdf
   Text: Raster interrupts are commonly used for sprite multiplexing, where more than 8 sprites appear on screen by reusing the same sprite registers.
```

---

## Query Processing Workflow

### End-to-End Example

```python
# User asks a question
question = "How do I use sprite multiplexing on the C64?"

# 1. RETRIEVAL PHASE
# ==================

# Generate query embedding
query_vec = embed_text(question)
# Result: [0.123, -0.456, 0.789, ..., -0.234] (384 dims)

# Search vector database
vector_results = vector_store.similarity_search(query_vec, k=5)
# Result: [
#   ("Sprite multiplexing reuses...", 0.89, {...}),
#   ("The VIC-II has 8 sprite registers...", 0.82, {...}),
#   ("Raster interrupts allow...", 0.78, {...}),
#   ("Move sprites by changing...", 0.71, {...}),
#   ("The $D000-$D02E registers...", 0.68, {...})
# ]

# Filter by threshold (0.3)
filtered = [r for r in vector_results if r[1] >= 0.3]
# Result: All 5 results pass threshold

# Extract context
contexts = [text for text, score, meta in filtered]


# 2. GENERATION PHASE
# ===================

# Format prompt
prompt = f"""Context: {' '.join(contexts)}

Question: {question}

Answer:"""

# Generate with LLM
answer = llm.generate(prompt)
# Result: "Sprite multiplexing is a technique that..."


# 3. RETURN RESULT
# ================

result = {
    "answer": answer,
    "sources": [
        {
            "text": text[:200],
            "score": score,
            "metadata": metadata
        }
        for text, score, metadata in filtered
    ]
}
```

### Performance Breakdown

**Typical Query Timing**:
```
Embedding generation:      50ms
Vector similarity search: 100ms
LLM generation:          3000ms
-----------------------------------
Total:                   3150ms (~3 seconds)
```

**Optimization Opportunities**:
1. **Caching**: Cache common query embeddings
2. **Parallel**: Embed query while retrieving context
3. **Streaming**: Stream LLM response for better UX
4. **Batch**: Process multiple queries together

---

## Prompt Engineering

### Default Prompt Template

```python
prompt_template = """Context: {context}

Question: {question}

Answer:"""
```

### Advanced Prompt Templates

#### 1. **Instructional Prompt**
```python
prompt_template = """You are a helpful assistant answering questions about vintage computers.
Use only the information from the provided context. If you're not sure, say so.

Context:
{context}

Question: {question}

Provide a clear, accurate answer:"""
```

#### 2. **Few-Shot Prompt**
```python
prompt_template = """Answer questions using the provided context.

Example:
Context: The SID chip produces sound.
Question: What does the SID chip do?
Answer: The SID chip generates audio output.

Your turn:
Context: {context}
Question: {question}
Answer:"""
```

#### 3. **Chain-of-Thought Prompt**
```python
prompt_template = """Context: {context}

Question: {question}

Let's think through this step by step:
1. What does the context tell us?
2. How does it relate to the question?
3. What's the answer?

Answer:"""
```

#### 4. **Structured Output Prompt**
```python
prompt_template = """Context: {context}

Question: {question}

Answer in this format:
- Summary: [one sentence]
- Details: [2-3 sentences]
- Related topics: [list]

Answer:"""
```

### Prompt Optimization Tips

**1. Be Specific**:
```python
# Vague
"Answer the question."

# Specific
"Answer the question using only information from the context. If the answer isn't in the context, say 'I don't have that information.'"
```

**2. Control Length**:
```python
"Provide a brief answer in 1-2 sentences."
"Provide a detailed explanation with examples."
```

**3. Control Style**:
```python
"Explain like I'm a beginner."
"Provide a technical explanation with terminology."
```

**4. Add Constraints**:
```python
"Answer in bullet points."
"Include specific numbers and dates."
"Cite the source document for each fact."
```

---

## Complete Examples

### Example 1: Simple RAG Query

```python
from src.rag import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Ingest documents (one-time)
pipeline.ingest_documents("data/documents")

# Query
result = pipeline.query("What is the SID chip?")

print(result["answer"])
# "The SID chip (Sound Interface Device) is the Commodore 64's sound generator..."
```

### Example 2: Streaming Response

```python
pipeline = RAGPipeline()

question = "Explain sprite multiplexing"

# Retrieve context
results = pipeline.retrieve_context(question)
contexts = [text for text, _, _ in results]

# Generate with streaming
print("Answer: ", end="", flush=True)
for chunk in pipeline.llm.stream_generate(
    pipeline.config['rag']['prompt_template'].format(
        context="\n\n".join(contexts),
        question=question
    )
):
    print(chunk, end="", flush=True)
print()
```

### Example 3: Multiple Queries (Batch)

```python
pipeline = RAGPipeline()

questions = [
    "What is a raster interrupt?",
    "How much RAM does the C64 have?",
    "What is the SID chip?"
]

for question in questions:
    result = pipeline.query(question)
    print(f"Q: {question}")
    print(f"A: {result['answer']}\n")
```

### Example 4: Custom Prompt Template

```python
pipeline = RAGPipeline()

# Override default prompt
custom_template = """You are an expert on vintage computers.

Context:
{context}

Question: {question}

Provide a detailed technical answer:"""

# Temporarily change template
original = pipeline.config['rag']['prompt_template']
pipeline.config['rag']['prompt_template'] = custom_template

result = pipeline.query("How do raster interrupts work?")
print(result["answer"])

# Restore original
pipeline.config['rag']['prompt_template'] = original
```

### Example 5: Metadata Filtering

```python
pipeline = RAGPipeline()

# Only search in specific document
query_embedding = pipeline.embedding_generator.embed_text(
    "What is sprite multiplexing?"
)

results = pipeline.vector_store.similarity_search(
    query_embedding=query_embedding,
    k=5,
    filter_dict={'filename': 'c64_programmers_reference.pdf'}
)

# Use filtered results for generation
contexts = [text for text, _, _ in results]
answer = pipeline.generate_response(
    "What is sprite multiplexing?",
    contexts
)
print(answer)
```

---

## Best Practices

### 1. **Model Selection**

**For RAG Applications**:
```python
# Good choices:
"llama3.2:latest"  # Fast, good quality
"mistral"          # Excellent instruction following
"phi3"             # Small, efficient

# Avoid:
"mixtral"          # Too slow for interactive use (unless you have powerful hardware)
```

### 2. **Temperature Tuning**

```python
# Factual Q&A (RAG)
temperature = 0.1-0.3  # Low randomness, consistent answers

# Creative tasks
temperature = 0.7-0.9  # Higher randomness, varied outputs

# Exact reproduction (testing)
temperature = 0.0  # Deterministic
```

### 3. **Context Management**

**Calculate Token Count**:
```python
def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters"""
    return len(text) // 4

context = "\n\n".join(contexts)
tokens = estimate_tokens(context)

if tokens > 2000:
    # Reduce context
    contexts = contexts[:3]  # Use fewer chunks
```

**Context Window Budget**:
```
Total context window: 4096 tokens
- System prompt: 100 tokens
- Context: 2000 tokens (max)
- Question: 50 tokens
- Reserved for response: 1946 tokens
```

### 4. **Retrieval Optimization**

**Adjust top_k and threshold**:
```python
# More context (better coverage, slower)
top_k = 10
score_threshold = 0.2

# Less context (faster, more focused)
top_k = 3
score_threshold = 0.5
```

**Quality vs Speed Trade-off**:
```python
# Fast mode (interactive chat)
config = {
    'retrieval': {'top_k': 3, 'score_threshold': 0.4},
    'llm': {'max_tokens': 256, 'temperature': 0.3}
}

# Quality mode (detailed analysis)
config = {
    'retrieval': {'top_k': 10, 'score_threshold': 0.2},
    'llm': {'max_tokens': 1024, 'temperature': 0.2}
}
```

### 5. **Error Handling**

```python
def safe_query(pipeline, question):
    try:
        result = pipeline.query(question)
        return result
    except ConnectionError:
        return {
            "answer": "Error: Ollama server not running. Start with: ollama serve",
            "sources": []
        }
    except Exception as e:
        logger.error(f"Query error: {e}")
        return {
            "answer": f"Error processing query: {str(e)}",
            "sources": []
        }
```

### 6. **Monitoring and Logging**

```python
import time
import logging

logger = logging.getLogger(__name__)

def query_with_metrics(pipeline, question):
    start = time.time()
    
    # Retrieval
    retrieval_start = time.time()
    results = pipeline.retrieve_context(question)
    retrieval_time = time.time() - retrieval_start
    
    # Generation
    generation_start = time.time()
    contexts = [text for text, _, _ in results]
    answer = pipeline.generate_response(question, contexts)
    generation_time = time.time() - generation_start
    
    total_time = time.time() - start
    
    # Log metrics
    logger.info(f"Query: {question}")
    logger.info(f"Retrieval: {retrieval_time:.3f}s")
    logger.info(f"Generation: {generation_time:.3f}s")
    logger.info(f"Total: {total_time:.3f}s")
    logger.info(f"Chunks retrieved: {len(results)}")
    
    return {"answer": answer, "sources": results}
```

### 7. **Testing**

```python
def test_rag_pipeline():
    pipeline = RAGPipeline()
    
    # Test retrieval
    results = pipeline.retrieve_context("test query")
    assert len(results) > 0, "No results retrieved"
    
    # Test generation
    answer = pipeline.generate_response(
        "test question",
        ["test context"]
    )
    assert len(answer) > 0, "Empty answer"
    
    # Test end-to-end
    result = pipeline.query("What is the C64?")
    assert "answer" in result
    assert "sources" in result
    
    print("✓ All tests passed")
```

---

## Next Steps

Continue to:
- **[05-CONFIGURATION-CLI.md](05-CONFIGURATION-CLI.md)** - Configuration and CLI usage
- **[03-EMBEDDINGS-VECTORSTORE.md](03-EMBEDDINGS-VECTORSTORE.md)** - Return to embeddings
- **[01-ARCHITECTURE-OVERVIEW.md](01-ARCHITECTURE-OVERVIEW.md)** - Return to architecture overview
