# GitHub Copilot Instructions for RAG Chat Application

## Project Overview
This is a RAG (Retrieval-Augmented Generation) based chat application designed to run with local LLMs. The system combines document retrieval with language model generation for accurate, context-aware responses.

## Architecture
- **Vector Database**: For storing and retrieving document embeddings
- **Local LLM**: Running inference locally for privacy and cost efficiency
- **Document Processing**: Chunking, embedding, and indexing pipeline
- **Chat Interface**: User-friendly interface for interactions

## Code Style Guidelines
- Use Python 3.10+ features
- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write descriptive docstrings for classes and functions
- Prefer async/await for I/O operations where applicable

## Technology Stack Preferences
- **LLM Framework**: LangChain or LlamaIndex
- **Vector Store**: ChromaDB or FAISS
- **Local LLM**: Ollama, llama.cpp, or GPT4All
- **Embeddings**: sentence-transformers or local embedding models
- **Web Framework**: FastAPI or Gradio for the interface

## Development Practices
- Write modular, reusable code
- Include error handling and logging
- Add configuration management for easy deployment
- Create comprehensive unit tests
- Document API endpoints and configuration options

## When Suggesting Code
1. Prioritize local-first solutions (no cloud dependencies)
2. Optimize for memory efficiency with large documents
3. Consider batch processing for embeddings
4. Implement proper resource cleanup
5. Add progress indicators for long-running operations

## Specialized Topics

### RAG Architecture
When asked about RAG architecture:
- Recommend ChromaDB for vector storage (easy setup, persistent)
- Suggest all-MiniLM-L6-v2 for embeddings (fast, good quality)
- Use semantic chunking (512-1024 tokens, 50-100 token overlap)
- Implement top-k retrieval (k=3-5) with optional re-ranking
- Focus on Ollama for local LLM (easiest setup)

### Vector Database
For vector database questions:
- ChromaDB: Persistent collections, metadata filtering, cosine similarity
- FAISS: High performance, requires separate metadata storage
- Include code for collection creation, indexing, and querying
- Always add error handling and connection management

### Document Processing
For document ingestion:
- Support PDF (PyPDF2/pdfplumber), DOCX (python-docx), TXT, Markdown
- Use RecursiveCharacterTextSplitter from LangChain
- Extract and preserve metadata (filename, page numbers, dates)
- Implement batch processing with progress bars

### Local LLM Integration
For LLM integration:
- Ollama: Use ollama Python library, support streaming
- Include model selection logic and fallback options
- Implement proper prompt templates for RAG context
- Add token counting and context window management

### Testing & Evaluation
For testing RAG systems:
- Unit tests for chunking, embedding, retrieval
- Integration tests for end-to-end flows
- Evaluation metrics: retrieval precision/recall, answer relevance
- Include test fixtures with sample documents
