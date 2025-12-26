# Embeddings and Vector Store Deep Dive

## Table of Contents
1. [Overview](#overview)
2. [Embedding Generation](#embedding-generation)
3. [Vector Store](#vector-store)
4. [Similarity Search](#similarity-search)
5. [Performance Optimization](#performance-optimization)
6. [Best Practices](#best-practices)

---

## Overview

This module bridges the gap between human-readable text and machine-searchable vectors:

```
Text → Embeddings → Vector Store → Similarity Search → Relevant Results
```

### Key Concepts

**Embeddings**: Dense numerical representations of text
- Transform text into fixed-size vectors (e.g., 384 dimensions)
- Semantically similar text → similar vectors
- Enable mathematical similarity calculations

**Vector Store**: Database optimized for high-dimensional vectors
- Stores embeddings with metadata
- Performs fast similarity search
- Scales to millions of vectors

**Similarity Search**: Find vectors closest to a query vector
- Cosine similarity: measures angle between vectors (0-1)
- Top-k retrieval: return k most similar items
- Filtering: constrain by metadata

---

## Embedding Generation

**Location**: `src/embeddings/generator.py`

### Class: EmbeddingGenerator

**Purpose**: Convert text into numerical vectors using sentence-transformers.

### Architecture

```
Text Input
    ↓
Tokenization (convert to IDs)
    ↓
Transformer Model (BERT-based)
    ↓
Pooling (mean pooling)
    ↓
Normalization (L2 norm)
    ↓
384-dimensional vector
```

---

### Initialization

```python
def __init__(
    self,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    cache_folder: Optional[str] = None,
    batch_size: int = 32,
    normalize: bool = True
):
```

**Parameters**:

1. **`model_name`**: Which embedding model to use
   - Default: `"sentence-transformers/all-MiniLM-L6-v2"`
   - Why this model:
     - **Size**: 80MB (fast download)
     - **Speed**: ~200 sentences/sec on CPU
     - **Quality**: Good for general text
     - **Dimensions**: 384 (efficient)
   
   - **Alternatives**:
     ```python
     # Better quality, slower, larger
     "sentence-transformers/all-mpnet-base-v2"  # 420MB, 768 dims
     
     # Multilingual support
     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
     
     # Domain-specific (e.g., code)
     "microsoft/codebert-base"
     ```

2. **`device`**: Computation device
   - `"auto"` - Try GPU, fall back to CPU (✅ Recommended)
   - `"cuda"` - Force GPU (fails if unavailable)
   - `"cpu"` - Force CPU
   - `"mps"` - Apple Silicon GPU
   
3. **`cache_folder`**: Where to store downloaded models
   - Default: `~/.cache/huggingface/`
   - Custom: `"./data/models"`
   - Benefits: Offline usage, controlled location

4. **`batch_size`**: Texts processed per batch
   - Default: `32`
   - Trade-off:
     - Larger = faster but more memory
     - Smaller = slower but safer
   - Adjust based on:
     - GPU memory (GPU: 64-128, CPU: 16-32)
     - Text length (long texts: smaller batch)

5. **`normalize`**: L2 normalization
   - Default: `True`
   - Purpose: Makes all vectors unit length
   - Benefits:
     - Cosine similarity = dot product (faster)
     - Comparable scores across queries
     - Standard practice for semantic search

**Example Usage**:
```python
from src.embeddings import EmbeddingGenerator

# Default configuration
generator = EmbeddingGenerator()

# Custom configuration
generator = EmbeddingGenerator(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="auto",
    cache_folder="./data/models",
    batch_size=32,
    normalize=True
)
```

---

### Method: `_get_device(requested_device: str) -> str`

**Purpose**: Intelligent device selection with fallback.

**Logic Flow**:
```python
if device == "auto" or "cuda":
    if torch.cuda.is_available():
        ✓ Use CUDA
    else:
        ✗ Fall back to CPU
elif device == "mps":
    if torch.backends.mps.is_available():
        ✓ Use MPS
    else:
        ✗ Fall back to CPU
else:
    → Use CPU
```

**Output Messages**:
```
# GPU available
✓ GPU available: NVIDIA Quadro M5000M
✓ Using CUDA for embeddings

# GPU requested but unavailable
✗ CUDA requested but not available
→ Falling back to CPU for embeddings

# CPU mode
→ Using CPU for embeddings
```

**Why This Matters**:
- **Transparent**: User knows what hardware is being used
- **Graceful**: No crashes if GPU unavailable
- **Optimal**: Uses fastest available device

**Device Detection**:
```python
# Check CUDA availability
torch.cuda.is_available()  # True/False
torch.cuda.get_device_name(0)  # "NVIDIA Quadro M5000M"

# Check MPS (Apple Silicon) availability
torch.backends.mps.is_available()  # True/False
```

---

### Method: `_load_model()`

**Purpose**: Load and initialize the sentence-transformer model.

**Process**:
```python
from sentence_transformers import SentenceTransformer

self.model = SentenceTransformer(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",  # or "cuda"
    cache_folder="./data/models"
)
```

**What Happens**:
1. **Check Cache**: Look for model in `cache_folder`
2. **Download if Needed**: Fetch from Hugging Face Hub (~80MB)
3. **Load Weights**: Load model parameters into memory
4. **Move to Device**: Transfer to CPU/GPU
5. **Log Success**: Confirm loading complete

**Download Process** (first time only):
```
Downloading (…)5d2/.gitattributes: 100%|██| 1.48k/1.48k
Downloading (…)_Pooling/config.json: 100%|██| 190/190
Downloading (…)b05d2/README.md: 100%|██| 10.5k/10.5k
Downloading (…)5d2/config.json: 100%|██| 612/612
Downloading (…)ce_transformers.json: 100%|██| 116/116
Downloading model.safetensors: 100%|██| 90.9M/90.9M
Downloading (…)nce_bert_config.json: 100%|██| 53.0/53.0
Downloading (…)cial_tokens_map.json: 100%|██| 112/112
Downloading (…)5d2/tokenizer.json: 100%|██| 466k/466k
Downloading (…)okenizer_config.json: 100%|██| 350/350
Downloading (…)b05d2/vocab.txt: 100%|██| 232k/232k
```

**Error Handling**:
```python
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("sentence-transformers not installed")
    raise
```

**Success Message**:
```
✓ Embedding model loaded successfully on CPU
```

---

### Method: `embed_text(text: str) -> List[float]`

**Purpose**: Generate embedding for a single text string.

**Usage**:
```python
generator = EmbeddingGenerator()

text = "What is a raster interrupt?"
embedding = generator.embed_text(text)

print(f"Embedding dimension: {len(embedding)}")  # 384
print(f"First 5 values: {embedding[:5]}")
# [-0.023, 0.145, -0.089, 0.234, -0.012]
```

**Process**:
```python
def embed_text(self, text: str) -> List[float]:
    embedding = self.model.encode(
        text,
        normalize_embeddings=True,  # L2 normalize
        show_progress_bar=False      # No progress for single text
    )
    return embedding.tolist()  # Convert numpy → list
```

**Performance**:
- **CPU**: ~5-10ms per text
- **GPU**: ~1-2ms per text
- **Batch**: More efficient for multiple texts

**Embedding Properties**:
```python
embedding = generator.embed_text("Hello world")

# Length is always 384
len(embedding)  # 384

# Normalized to unit length (if normalize=True)
import numpy as np
np.linalg.norm(embedding)  # ~1.0

# All values in range [-1, 1]
min(embedding), max(embedding)  # (-0.5, 0.5) typically
```

---

### Method: `embed_texts(texts: List[str], show_progress: bool = True) -> List[List[float]]`

**Purpose**: Generate embeddings for multiple texts efficiently (batched).

**Usage**:
```python
generator = EmbeddingGenerator(batch_size=32)

texts = [
    "What is a raster interrupt?",
    "How does the VIC-II chip work?",
    "Explain sprite multiplexing",
    # ... 1000 more texts
]

embeddings = generator.embed_texts(texts, show_progress=True)

print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")
```

**Batch Processing**:
```python
# Process in batches of 32
texts = [text1, text2, ..., text100]

# Internally:
batch_1 = texts[0:32]   → Process
batch_2 = texts[32:64]  → Process
batch_3 = texts[64:96]  → Process
batch_4 = texts[96:100] → Process
```

**Progress Bar**:
```
Generating embeddings for 1000 texts
Batches: 100%|████████████| 32/32 [00:15<00:00,  2.13it/s]
Generated 1000 embeddings
```

**Performance Comparison**:
```python
import time

# Sequential (slow)
start = time.time()
embeddings = [generator.embed_text(t) for t in texts]
seq_time = time.time() - start

# Batched (fast)
start = time.time()
embeddings = generator.embed_texts(texts)
batch_time = time.time() - start

print(f"Sequential: {seq_time:.2f}s")  # 50s
print(f"Batched: {batch_time:.2f}s")    # 15s
print(f"Speedup: {seq_time/batch_time:.1f}x")  # 3.3x
```

**Why Batching is Faster**:
1. **Parallelism**: GPU can process multiple texts simultaneously
2. **Reduced Overhead**: One model forward pass for batch
3. **Memory Efficiency**: Better cache utilization

---

### Method: `get_embedding_dimension() -> int`

**Purpose**: Get the dimensionality of embeddings.

**Usage**:
```python
generator = EmbeddingGenerator()
dim = generator.get_embedding_dimension()
print(dim)  # 384
```

**Why This Matters**:
- **Vector Store Setup**: ChromaDB needs to know dimension
- **Validation**: Ensure embeddings match expected size
- **Model Comparison**: Different models have different dimensions

**Dimension Comparison**:
| Model | Dimensions | Size | Speed |
|-------|------------|------|-------|
| all-MiniLM-L6-v2 | 384 | 80MB | Fast |
| all-mpnet-base-v2 | 768 | 420MB | Slower |
| text-embedding-ada-002 | 1536 | API | API |

---

### Method: `get_model_info() -> dict`

**Purpose**: Get metadata about the loaded model.

**Returns**:
```python
{
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'embedding_dimension': 384,
    'device': 'cpu',
    'normalize': True,
    'batch_size': 32
}
```

**Usage**:
```python
generator = EmbeddingGenerator()
info = generator.get_model_info()

print(f"Model: {info['model_name']}")
print(f"Running on: {info['device']}")
print(f"Dimensions: {info['embedding_dimension']}")
```

---

## Vector Store

**Location**: `src/vector_store/chroma_store.py`

### Class: ChromaVectorStore

**Purpose**: Store and retrieve document embeddings using ChromaDB.

### Why ChromaDB?

| Feature | ChromaDB | FAISS | Pinecone |
|---------|----------|-------|----------|
| **Setup** | Single line | Medium | API signup |
| **Persistence** | Built-in | Manual | Cloud |
| **Metadata** | Native | External | Native |
| **Scale** | <1M docs | >1M docs | Unlimited |
| **Cost** | Free | Free | Paid |
| **Network** | Local | Local | Cloud |

**ChromaDB Architecture**:
```
ChromaDB
├── Client (PersistentClient)
│   └── path: ./data/vectorstore
├── Collections (tables)
│   ├── rag_documents (your collection)
│   │   ├── Documents (text)
│   │   ├── Embeddings (vectors)
│   │   ├── Metadata (dict)
│   │   └── IDs (unique strings)
│   └── other_collection
└── Index (HNSW algorithm)
```

---

### Initialization

```python
def __init__(
    self,
    collection_name: str,
    persist_directory: str,
    distance_metric: str = "cosine"
):
```

**Parameters**:

1. **`collection_name`**: Name of the collection (like a table)
   - Example: `"rag_documents"`
   - Multiple collections allowed per database
   - Used to organize different document sets

2. **`persist_directory`**: Where to save the database
   - Example: `"./data/vectorstore"`
   - Creates directory if it doesn't exist
   - Survives program restarts

3. **`distance_metric`**: How to measure similarity
   - `"cosine"` - Angle between vectors (✅ Recommended)
   - `"l2"` - Euclidean distance
   - `"ip"` - Inner product

**Initialization Process**:
```python
# 1. Create directory
Path(persist_directory).mkdir(parents=True, exist_ok=True)

# 2. Initialize ChromaDB client
client = chromadb.PersistentClient(path=persist_directory)

# 3. Get or create collection
try:
    collection = client.get_collection(name=collection_name)
    # Collection exists
except:
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": distance_metric}
    )
    # New collection created
```

**Example**:
```python
from src.vector_store import ChromaVectorStore

store = ChromaVectorStore(
    collection_name="rag_documents",
    persist_directory="./data/vectorstore",
    distance_metric="cosine"
)
```

---

### Distance Metrics Explained

#### Cosine Similarity
```python
similarity = cos(θ) = (A · B) / (|A| × |B|)
```

**Range**: -1 to 1 (higher = more similar)
**Use Case**: Semantic similarity (✅ Recommended for RAG)

**Example**:
```python
vec1 = [1, 0, 0]  # "machine learning"
vec2 = [0.7, 0.7, 0]  # "ML algorithms"
# cosine = high (~0.7)

vec3 = [0, 0, 1]  # "cooking recipes"
# cosine = low (~0)
```

**Why Normalized Embeddings**:
```python
# If vectors are normalized (length = 1):
cosine_similarity = dot_product(A, B)
# Faster computation!
```

#### L2 Distance (Euclidean)
```python
distance = √[(x1-x2)² + (y1-y2)² + ... + (zn-zn)²]
```

**Range**: 0 to ∞ (lower = more similar)
**Use Case**: Exact matches, geometric proximity

#### Inner Product
```python
similarity = A · B = a1*b1 + a2*b2 + ... + an*bn
```

**Range**: -∞ to ∞ (higher = more similar)
**Use Case**: When magnitude matters

---

### Method: `add_documents(...) -> List[str]`

**Purpose**: Store documents with their embeddings in ChromaDB.

**Signature**:
```python
def add_documents(
    self,
    documents: List[str],           # Text content
    embeddings: List[List[float]],  # Vector embeddings
    metadata: Optional[List[Dict[str, Any]]] = None,  # Metadata
    ids: Optional[List[str]] = None  # Unique IDs
) -> List[str]:
```

**Usage Example**:
```python
store = ChromaVectorStore(
    collection_name="rag_documents",
    persist_directory="./data/vectorstore"
)

# Prepare data
documents = [
    "The Commodore 64 has 64KB of RAM",
    "The VIC-II chip handles graphics",
    "The SID chip produces sound"
]

# Generate embeddings
generator = EmbeddingGenerator()
embeddings = generator.embed_texts(documents)

# Metadata for each document
metadata = [
    {'filename': 'c64_manual.pdf', 'page': 1, 'chunk_index': 0},
    {'filename': 'c64_manual.pdf', 'page': 5, 'chunk_index': 1},
    {'filename': 'c64_manual.pdf', 'page': 8, 'chunk_index': 2}
]

# Add to vector store
ids = store.add_documents(
    documents=documents,
    embeddings=embeddings,
    metadata=metadata
)

print(f"Added {len(ids)} documents")
```

**ID Generation**:
```python
if ids is None:
    # Auto-generate IDs
    existing_count = collection.count()
    ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
    # Result: ["doc_0", "doc_1", "doc_2"]
```

**Metadata Requirement**:
```python
# ChromaDB requires non-empty metadata
if metadata is None:
    metadata = [{"source": "unknown"} for _ in documents]
else:
    # Ensure all metadata dicts have at least one key
    metadata = [
        m if m else {"source": "unknown"}
        for m in metadata
    ]
```

**What Gets Stored**:
```
Collection: rag_documents
├── Document 0
│   ├── ID: "doc_0"
│   ├── Text: "The Commodore 64 has 64KB of RAM"
│   ├── Embedding: [0.123, -0.456, ..., 0.789] (384 dims)
│   └── Metadata: {'filename': 'c64_manual.pdf', 'page': 1}
├── Document 1
│   ├── ID: "doc_1"
│   ├── Text: "The VIC-II chip handles graphics"
│   ├── Embedding: [-0.234, 0.567, ..., -0.123] (384 dims)
│   └── Metadata: {'filename': 'c64_manual.pdf', 'page': 5}
└── ...
```

---

### Method: `similarity_search(...) -> List[Tuple[str, float, Dict]]`

**Purpose**: Find documents most similar to a query.

**Signature**:
```python
def similarity_search(
    self,
    query_embedding: List[float],           # Query vector
    k: int = 5,                             # Top-k results
    filter_dict: Optional[Dict[str, Any]] = None  # Metadata filters
) -> List[Tuple[str, float, Dict[str, Any]]]:
    # Returns: [(text, similarity_score, metadata), ...]
```

**Complete Example**:
```python
# 1. Generate query embedding
query = "How much memory does the Commodore 64 have?"
query_embedding = generator.embed_text(query)

# 2. Search vector store
results = store.similarity_search(
    query_embedding=query_embedding,
    k=3  # Return top 3 results
)

# 3. Process results
for text, score, metadata in results:
    print(f"Score: {score:.3f}")
    print(f"Source: {metadata['filename']}")
    print(f"Text: {text[:100]}...")
    print()

# Output:
# Score: 0.847
# Source: c64_manual.pdf
# Text: The Commodore 64 has 64KB of RAM...
#
# Score: 0.723
# Source: c64_manual.pdf
# Text: Memory is divided into ROM and RAM...
#
# Score: 0.691
# Source: c64_reference.pdf
# Text: The system uses 64 kilobytes of memory...
```

**With Metadata Filtering**:
```python
# Only search within specific file
results = store.similarity_search(
    query_embedding=query_embedding,
    k=5,
    filter_dict={'filename': 'c64_manual.pdf'}
)

# Only search specific page range
results = store.similarity_search(
    query_embedding=query_embedding,
    k=5,
    filter_dict={'page': {'$gte': 10, '$lte': 20}}
)
```

**Distance → Similarity Conversion**:
```python
# ChromaDB returns distances, we convert to similarity
similarity = 1 - distance

# For cosine distance:
# distance = 0.0 → similarity = 1.0 (identical)
# distance = 1.0 → similarity = 0.0 (orthogonal)
# distance = 2.0 → similarity = -1.0 (opposite)
```

---

### Method: `delete_documents(ids: List[str]) -> bool`

**Purpose**: Remove documents from the collection.

**Usage**:
```python
# Delete specific documents
ids_to_delete = ["doc_5", "doc_6", "doc_7"]
success = store.delete_documents(ids_to_delete)

if success:
    print(f"Deleted {len(ids_to_delete)} documents")
```

**When to Use**:
- Remove outdated documents
- Delete duplicates
- Clear specific sources
- Manage storage space

---

### Method: `get_collection_stats() -> Dict[str, Any]`

**Purpose**: Get information about the collection.

**Returns**:
```python
{
    'name': 'rag_documents',
    'count': 729,
    'metadata': {'hnsw:space': 'cosine'}
}
```

**Usage**:
```python
stats = store.get_collection_stats()
print(f"Collection: {stats['name']}")
print(f"Documents: {stats['count']}")
print(f"Distance metric: {stats['metadata']['hnsw:space']}")
```

---

### Method: `clear_collection() -> bool`

**Purpose**: Delete all documents from the collection.

**Usage**:
```python
if store.clear_collection():
    print("Collection cleared")
```

**What Happens**:
```python
1. Delete existing collection
2. Recreate empty collection with same settings
3. Ready for new documents
```

**Warning**: This is irreversible! Make backups first.

---

## Similarity Search

### How Similarity Search Works

#### 1. **Query Processing**
```python
query = "What is a raster interrupt?"
query_embedding = generator.embed_text(query)
# [0.123, -0.456, ..., 0.789] (384 dimensions)
```

#### 2. **Vector Comparison**
```python
# For each document in collection:
for doc_embedding in collection:
    similarity = cosine_similarity(query_embedding, doc_embedding)
    # Store (document, similarity) pair
```

#### 3. **Ranking**
```python
# Sort by similarity (highest first)
results = sorted(results, key=lambda x: x[1], reverse=True)
```

#### 4. **Top-K Selection**
```python
# Return top k results
top_results = results[:k]
```

### HNSW Algorithm

**What is HNSW?**: Hierarchical Navigable Small World

**Purpose**: Fast approximate nearest neighbor search

**How It Works**:
```
Traditional Search: Compare query to ALL vectors
Time: O(n) - slow for large collections

HNSW: Build graph structure, navigate to nearest neighbors
Time: O(log n) - fast even for millions of vectors
```

**Trade-offs**:
- **Speed**: 100-1000x faster than brute force
- **Accuracy**: ~99% recall (finds ~99% of true nearest neighbors)
- **Memory**: Extra memory for graph structure

**Configuration in ChromaDB**:
```python
collection = client.create_collection(
    name="documents",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 100,  # Build quality
        "hnsw:search_ef": 100,         # Search quality
        "hnsw:M": 16                   # Graph connectivity
    }
)
```

---

## Performance Optimization

### 1. **Batch Embedding**

**Slow** (sequential):
```python
embeddings = []
for text in texts:
    emb = generator.embed_text(text)
    embeddings.append(emb)
# Time: 50 seconds for 1000 texts
```

**Fast** (batched):
```python
embeddings = generator.embed_texts(texts)
# Time: 15 seconds for 1000 texts (3.3x faster)
```

### 2. **GPU Acceleration**

**Setup**:
```python
# Automatic GPU detection
generator = EmbeddingGenerator(device="auto")

# Force GPU
generator = EmbeddingGenerator(device="cuda")
```

**Speedup**:
```
CPU (Intel i7): ~200 texts/sec
GPU (RTX 3060): ~2000 texts/sec (10x faster)
```

### 3. **Optimal Batch Size**

**Finding the Right Size**:
```python
import time

batch_sizes = [8, 16, 32, 64, 128]
texts = ["sample text"] * 1000

for batch_size in batch_sizes:
    generator = EmbeddingGenerator(batch_size=batch_size)
    
    start = time.time()
    embeddings = generator.embed_texts(texts, show_progress=False)
    elapsed = time.time() - start
    
    print(f"Batch size {batch_size}: {elapsed:.2f}s")

# Output:
# Batch size 8: 25.3s
# Batch size 16: 18.7s
# Batch size 32: 15.2s  ← Sweet spot
# Batch size 64: 15.1s
# Batch size 128: 15.0s (marginal improvement, more memory)
```

**Guidelines**:
- **CPU**: 16-32
- **GPU (4GB)**: 32-64
- **GPU (8GB+)**: 64-128

### 4. **Collection Size Impact**

| Documents | Search Time | Index Size |
|-----------|-------------|------------|
| 1K | ~10ms | ~1MB |
| 10K | ~15ms | ~10MB |
| 100K | ~25ms | ~100MB |
| 1M | ~50ms | ~1GB |

**Conclusion**: ChromaDB scales well up to ~1M documents.

### 5. **Metadata Indexing**

**Without Metadata Filter**:
```python
results = store.similarity_search(query_embedding, k=5)
# Searches all 100K documents
```

**With Metadata Filter**:
```python
results = store.similarity_search(
    query_embedding,
    k=5,
    filter_dict={'filename': 'specific.pdf'}
)
# Only searches subset (e.g., 729 documents)
# Much faster!
```

---

## Best Practices

### 1. **Model Selection**

**Choose Based On**:

| Priority | Model | Dimensions | Size |
|----------|-------|------------|------|
| Speed | all-MiniLM-L6-v2 | 384 | 80MB |
| Quality | all-mpnet-base-v2 | 768 | 420MB |
| Multilingual | paraphrase-multilingual | 384 | 420MB |
| Domain-specific | Fine-tune your own | Custom | Custom |

### 2. **Embedding Strategy**

**Documents**:
```python
# Embed once during ingestion
chunks = chunker.chunk_documents(documents)
texts = [chunk.content for chunk in chunks]
embeddings = generator.embed_texts(texts)
store.add_documents(texts, embeddings)
```

**Queries**:
```python
# Embed each query
query_embedding = generator.embed_text(user_question)
results = store.similarity_search(query_embedding)
```

### 3. **Collection Management**

**One Collection per Document Set**:
```python
# Good: Separate collections
store_manuals = ChromaVectorStore(collection_name="manuals")
store_guides = ChromaVectorStore(collection_name="guides")

# Can search each independently or combine results
```

**Multiple Collections vs. Metadata Filtering**:
```python
# Option 1: Separate collections (better isolation)
store_c64 = ChromaVectorStore(collection_name="c64_docs")
store_amiga = ChromaVectorStore(collection_name="amiga_docs")

# Option 2: One collection with metadata (simpler)
store = ChromaVectorStore(collection_name="retro_docs")
# Add with metadata: {'system': 'c64'} or {'system': 'amiga'}
# Search with filter: filter_dict={'system': 'c64'}
```

### 4. **Backup and Restore**

**Backup**:
```bash
# ChromaDB stores in persist_directory
cp -r ./data/vectorstore ./backups/vectorstore_2024-01-15
```

**Restore**:
```bash
cp -r ./backups/vectorstore_2024-01-15 ./data/vectorstore
```

### 5. **Monitoring**

```python
# Check collection size
stats = store.get_collection_stats()
print(f"Documents: {stats['count']}")

# Monitor search performance
import time
start = time.time()
results = store.similarity_search(query_embedding, k=5)
print(f"Search time: {time.time() - start:.3f}s")

# Check embedding dimension
dim = generator.get_embedding_dimension()
assert dim == 384, "Dimension mismatch!"
```

---

## Complete Usage Example

```python
from pathlib import Path
from src.embeddings import EmbeddingGenerator
from src.vector_store import ChromaVectorStore
from src.document_processing import DocumentLoader, DocumentChunker

# 1. Load documents
loader = DocumentLoader()
documents = loader.load_directory(Path('data/documents'))
print(f"Loaded {len(documents)} documents")

# 2. Chunk documents
chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk_documents(documents)
print(f"Created {len(chunks)} chunks")

# 3. Initialize embedding generator
generator = EmbeddingGenerator(device="auto", batch_size=32)

# 4. Generate embeddings
texts = [chunk.content for chunk in chunks]
embeddings = generator.embed_texts(texts, show_progress=True)
print(f"Generated {len(embeddings)} embeddings")

# 5. Initialize vector store
store = ChromaVectorStore(
    collection_name="rag_documents",
    persist_directory="./data/vectorstore"
)

# 6. Add to vector store
metadata = [chunk.metadata for chunk in chunks]
ids = store.add_documents(
    documents=texts,
    embeddings=embeddings,
    metadata=metadata
)
print(f"Stored {len(ids)} documents")

# 7. Query the system
query = "What is a raster interrupt?"
query_embedding = generator.embed_text(query)

results = store.similarity_search(
    query_embedding=query_embedding,
    k=3
)

print(f"\nQuery: {query}\n")
for i, (text, score, meta) in enumerate(results, 1):
    print(f"{i}. Score: {score:.3f}")
    print(f"   Source: {meta['filename']}")
    print(f"   Text: {text[:150]}...")
    print()
```

---

## Next Steps

Continue to:
- **[04-LLM-RAG-PIPELINE.md](04-LLM-RAG-PIPELINE.md)** - LLM integration and complete RAG pipeline
- **[02-DOCUMENT-PROCESSING.md](02-DOCUMENT-PROCESSING.md)** - Return to document processing
- **[01-ARCHITECTURE-OVERVIEW.md](01-ARCHITECTURE-OVERVIEW.md)** - Return to architecture overview
