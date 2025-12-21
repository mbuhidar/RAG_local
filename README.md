# RAG-Based Chat Application with Local LLM

A Retrieval-Augmented Generation (RAG) chat application that runs entirely on local infrastructure, ensuring privacy and control over your data.

## GitHub Copilot Agent Skills

This project includes custom GitHub Copilot agent skills to accelerate development. Use the following triggers in your Copilot chat:

### Available Skills

- **@rag-arch** - Get help with RAG architecture and design decisions
- **@vector-db** - Assistance with vector database operations
- **@local-llm** - Local LLM integration and optimization
- **@doc-process** - Document processing pipeline development
- **@embeddings** - Embedding model selection and optimization
- **@rag-test** - Testing and evaluation strategies
- **@chat-ui** - Chat interface development
- **@optimize** - Performance optimization tips

### Usage Examples

```
# In GitHub Copilot Chat:
@rag-arch How should I chunk documents for optimal retrieval?
@vector-db Show me how to create a ChromaDB collection with metadata filtering
@local-llm What's the best way to load an Ollama model with streaming?
@embeddings Which embedding model works best for technical documentation?
```

## Project Structure

```
RAG_local/
├── .github/
│   └── copilot-instructions.md    # Project-wide Copilot instructions
├── .vscode/
│   ├── settings.json               # VS Code settings
│   └── copilot-agent-skills.json   # Custom Copilot agent skills
├── src/
│   ├── embeddings/                 # Embedding generation
│   ├── retrieval/                  # Vector search and retrieval
│   ├── llm/                        # Local LLM integration
│   ├── processing/                 # Document processing
│   └── chat/                       # Chat interface
├── tests/                          # Unit and integration tests
├── config/                         # Configuration files
├── data/                          # Sample documents and datasets
└── README.md
```

## Getting Started

1. **Install Dependencies** (to be created)
2. **Set Up Local LLM** (instructions coming)
3. **Configure Vector Database** (setup guide coming)
4. **Run the Application** (commands coming)

## Features (Planned)

- 📚 Multi-format document ingestion (PDF, DOCX, TXT, MD)
- 🔍 Semantic search with vector embeddings
- 🤖 Local LLM inference (no cloud required)
- 💬 Interactive chat interface
- 📊 Conversation history management
- ⚡ Optimized for consumer hardware
- 🔒 Complete data privacy

## Technology Stack

- **LLM Framework**: LangChain / LlamaIndex
- **Vector Store**: ChromaDB / FAISS
- **Local LLM**: Ollama / llama.cpp
- **Embeddings**: sentence-transformers
- **Interface**: FastAPI + Gradio / Streamlit
- **Language**: Python 3.10+

## Development

Use the GitHub Copilot agent skills to accelerate development. The skills are configured to provide context-aware assistance specific to RAG applications.

## License

MIT License (or your preferred license)
