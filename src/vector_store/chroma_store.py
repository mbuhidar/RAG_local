"""
ChromaDB Vector Store Implementation

Provides a persistent vector database using ChromaDB for local storage.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb

from . import VectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of the vector store."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        distance_metric: str = "cosine"
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            distance_metric: Distance metric (cosine, l2, ip)
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with new API
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance_metric}
            )
            logger.info(f"Created new collection: {collection_name}")

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to ChromaDB.

        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadata: Optional metadata for each document
            ids: Optional custom IDs for documents

        Returns:
            List of document IDs
        """
        if ids is None:
            # Generate IDs if not provided
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

        if metadata is None:
            metadata = [{"source": "unknown"} for _ in documents]
        else:
            # Ensure all metadata dicts have at least one key
            metadata = [
                m if m else {"source": "unknown"}
                for m in metadata
            ]

        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadata,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to collection")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of (document, score, metadata) tuples
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_dict
            )

            # Format results
            documents = results['documents'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]

            return [
                (doc, 1 - dist, meta)  # Convert distance to similarity
                for doc, dist, meta in zip(documents, distances, metadatas)
            ]
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise

    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata
        }

    def clear_collection(self) -> bool:
        """Clear all documents from collection."""
        try:
            # Delete and recreate collection
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata=self.collection.metadata
            )
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

    def persist(self):
        """Persist the database to disk."""
        try:
            self.client.persist()
            logger.info("Database persisted to disk")
        except Exception as e:
            logger.error(f"Error persisting database: {e}")
