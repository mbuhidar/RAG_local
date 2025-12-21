"""
Vector Database Module for RAG System

This module provides abstraction for vector database operations including:
- Document storage and retrieval
- Similarity search
- Collection management
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    def add_documents(
        self, 
        documents: List[str], 
        embeddings: List[List[float]], 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add documents with their embeddings to the vector store."""
        pass

    @abstractmethod
    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        pass

    @abstractmethod
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        pass
