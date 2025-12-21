"""
Document Processing Module

Handles document loading, chunking, and preprocessing for RAG system.
Supports multiple file formats and intelligent chunking strategies.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document representation with content and metadata."""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None

    def __repr__(self) -> str:
        return f"Document(id={self.doc_id}, content_length={len(self.content)})"


@dataclass
class DocumentChunk:
    """Chunk of a document with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    parent_doc_id: Optional[str] = None

    def __repr__(self) -> str:
        return f"DocumentChunk(id={self.chunk_id}, content_length={len(self.content)})"
