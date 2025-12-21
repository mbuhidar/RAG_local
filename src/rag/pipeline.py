"""
RAG Pipeline Module

Orchestrates the complete RAG pipeline including document ingestion,
embedding generation, retrieval, and response generation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..document_processing import Document, DocumentChunk
from ..document_processing.chunker import DocumentChunker
from ..document_processing.loader import DocumentLoader
from ..embeddings import EmbeddingGenerator
from ..llm.ollama_llm import OllamaLLM
from ..vector_store.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline orchestrator."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize RAG pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self._initialize_components()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = "config/config.yaml"

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing RAG pipeline components")

        # Document processing
        self.loader = DocumentLoader(
            supported_formats=self.config['document_processing']['supported_formats']
        )
        self.chunker = DocumentChunker(
            chunk_size=self.config['document_processing']['chunk_size'],
            chunk_overlap=self.config['document_processing']['chunk_overlap'],
            separators=self.config['document_processing']['separators']
        )

        # Embeddings
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.config['embeddings']['model_name'],
            device=self.config['embeddings']['model_kwargs']['device'],
            cache_folder=self.config['embeddings']['cache_folder'],
            batch_size=self.config['embeddings']['encode_kwargs']['batch_size'],
            normalize=self.config['embeddings']['encode_kwargs']['normalize_embeddings']
        )

        # Vector store
        self.vector_store = ChromaVectorStore(
            collection_name=self.config['vector_db']['collection_name'],
            persist_directory=self.config['vector_db']['persist_directory'],
            distance_metric=self.config['vector_db']['distance_metric']
        )

        # LLM
        self.llm = OllamaLLM(
            model_name=self.config['llm']['model_name'],
            base_url=self.config['llm']['base_url'],
            temperature=self.config['llm']['temperature'],
            max_tokens=self.config['llm']['max_tokens'],
            context_window=self.config['llm']['context_window']
        )

        logger.info("All components initialized successfully")

    def ingest_documents(
        self,
        document_path: str,
        recursive: bool = True
    ) -> int:
        """
        Ingest documents from a directory.

        Args:
            document_path: Path to documents directory
            recursive: Whether to search subdirectories

        Returns:
            Number of chunks ingested
        """
        logger.info(f"Starting document ingestion from {document_path}")

        # Load documents
        documents = self.loader.load_directory(
            Path(document_path),
            recursive=recursive
        )

        if not documents:
            logger.warning("No documents found to ingest")
            return 0

        # Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

        # Generate embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_generator.embed_texts(
            chunk_texts,
            show_progress=True
        )

        # Store in vector database
        metadata = [chunk.metadata for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        self.vector_store.add_documents(
            documents=chunk_texts,
            embeddings=embeddings,
            metadata=metadata,
            ids=chunk_ids
        )

        # Persist to disk
        self.vector_store.persist()

        logger.info(f"Successfully ingested {len(chunks)} chunks")
        return len(chunks)

    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Retrieve relevant context for a query.

        Args:
            query: Query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of (text, score, metadata) tuples
        """
        top_k = top_k or self.config['retrieval']['top_k']
        score_threshold = score_threshold or self.config['retrieval']['score_threshold']

        # Generate query embedding
        query_embedding = self.embedding_generator.embed_text(query)

        # Search vector store
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=top_k
        )

        # Filter by score threshold
        filtered_results = [
            (text, score, meta)
            for text, score, meta in results
            if score >= score_threshold
        ]

        if results:
            max_score = max(score for _, score, _ in results)
            logger.info(f"Retrieved {len(filtered_results)} relevant chunks (max score: {max_score:.3f}, threshold: {score_threshold})")
        else:
            logger.info("No results found from vector search")
        return filtered_results

    def generate_response(
        self,
        query: str,
        context: List[str],
        stream: bool = False
    ) -> str:
        """
        Generate response using LLM.

        Args:
            query: User query
            context: List of context strings
            stream: Whether to stream response

        Returns:
            Generated response
        """
        # Format prompt
        context_text = "\n\n".join(context)
        prompt = self.config['rag']['prompt_template'].format(
            context=context_text,
            question=query
        )

        # Generate response
        if stream:
            return self.llm.stream_generate(prompt)
        else:
            return self.llm.generate(prompt)

    def query(
        self,
        question: str,
        return_sources: bool = True,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve context and generate response.

        Args:
            question: User question
            return_sources: Whether to return source documents
            stream: Whether to stream response

        Returns:
            Dictionary with answer and optional sources
        """
        logger.info(f"Processing query: {question}")

        # Retrieve relevant context
        results = self.retrieve_context(question)

        if not results:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": []
            }

        # Extract context and metadata
        contexts = [text for text, _, _ in results]
        sources = [
            {
                "text": text[:200] + "..." if len(text) > 200 else text,
                "score": score,
                "metadata": meta
            }
            for text, score, meta in results
        ]

        # Generate response
        answer = self.generate_response(question, contexts, stream=stream)

        result = {"answer": answer}
        if return_sources and not stream:
            result["sources"] = sources

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "embedding_model": self.embedding_generator.get_model_info(),
            "llm": self.llm.get_model_info()
        }

    def clear_database(self) -> bool:
        """Clear all documents from vector database."""
        logger.warning("Clearing vector database")
        return self.vector_store.clear_collection()
