"""
Document Chunking Module

Implements intelligent chunking strategies for breaking documents into 
manageable pieces for embedding and retrieval.
"""

import hashlib
import logging
from typing import List, Optional

from . import Document, DocumentChunk

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Chunks documents using various strategies."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize document chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            separators: List of separators to split on (in order of preference)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """
        Chunk a document into smaller pieces.

        Args:
            document: Document to chunk

        Returns:
            List of DocumentChunk objects
        """
        chunks = self._recursive_split(document.content)
        
        document_chunks = []
        for i, chunk_text in enumerate(chunks):
            # Create unique chunk ID
            chunk_id = self._generate_chunk_id(document, i)
            
            # Copy and extend metadata
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk_text)
            })

            document_chunks.append(DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                parent_doc_id=document.doc_id
            ))

        logger.info(f"Created {len(document_chunks)} chunks from document")
        return document_chunks

    def chunk_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of documents to chunk

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents")
        return all_chunks

    def _recursive_split(self, text: str) -> List[str]:
        """
        Recursively split text using different separators.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        final_chunks = []
        separator = self.separators[0]
        
        # Try to split with current separator
        splits = text.split(separator)
        
        good_splits = []
        for split in splits:
            if len(split) < self.chunk_size:
                good_splits.append(split)
            else:
                # If still too large, try next separator
                if len(self.separators) > 1:
                    chunker = DocumentChunker(
                        self.chunk_size,
                        self.chunk_overlap,
                        self.separators[1:]
                    )
                    good_splits.extend(chunker._recursive_split(split))
                else:
                    # Force split by character count
                    good_splits.extend(self._force_split(split))

        # Merge small chunks and handle overlap
        final_chunks = self._merge_chunks(good_splits, separator)
        return final_chunks

    def _force_split(self, text: str) -> List[str]:
        """Force split text by character count."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

    def _merge_chunks(self, chunks: List[str], separator: str) -> List[str]:
        """
        Merge small chunks and add overlap.

        Args:
            chunks: List of text chunks
            separator: Separator used for splitting

        Returns:
            List of merged chunks
        """
        merged = []
        current_chunk = []
        current_length = 0

        for chunk in chunks:
            chunk_length = len(chunk)
            
            # Add chunk if it fits
            if current_length + chunk_length + len(separator) <= self.chunk_size:
                current_chunk.append(chunk)
                current_length += chunk_length + len(separator)
            else:
                # Save current chunk and start new one
                if current_chunk:
                    merged.append(separator.join(current_chunk))
                
                # Start new chunk with overlap from previous
                if merged and self.chunk_overlap > 0:
                    overlap_text = merged[-1][-self.chunk_overlap:]
                    current_chunk = [overlap_text, chunk]
                    current_length = len(overlap_text) + chunk_length + len(separator)
                else:
                    current_chunk = [chunk]
                    current_length = chunk_length

        # Add final chunk
        if current_chunk:
            merged.append(separator.join(current_chunk))

        return merged

    def _generate_chunk_id(self, document: Document, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        base = f"{document.metadata.get('filename', 'unknown')}_{chunk_index}"
        return hashlib.md5(base.encode()).hexdigest()
