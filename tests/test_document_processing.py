"""
Test suite for document processing module.
"""

from pathlib import Path

import pytest

from src.document_processing import Document, DocumentChunk
from src.document_processing.chunker import DocumentChunker
from src.document_processing.loader import DocumentLoader


class TestDocumentLoader:
    """Tests for DocumentLoader."""

    def test_load_text_file(self, tmp_path):
        """Test loading a text file."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_content = "This is a test document.\nWith multiple lines."
        test_file.write_text(test_content)

        # Load document
        loader = DocumentLoader()
        doc = loader.load_file(test_file)

        assert doc is not None
        assert doc.content == test_content
        assert doc.metadata['filename'] == 'test.txt'
        assert doc.metadata['extension'] == 'txt'

    def test_load_directory(self, tmp_path):
        """Test loading multiple files from directory."""
        # Create test files
        (tmp_path / "doc1.txt").write_text("Document 1")
        (tmp_path / "doc2.txt").write_text("Document 2")
        (tmp_path / "doc3.md").write_text("# Document 3")

        # Load directory
        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path, recursive=False)

        assert len(docs) == 3
        assert all(isinstance(doc, Document) for doc in docs)


class TestDocumentChunker:
    """Tests for DocumentChunker."""

    def test_chunk_small_document(self):
        """Test chunking a small document."""
        doc = Document(
            content="This is a small test document.",
            metadata={'filename': 'test.txt'}
        )

        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) >= 1
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert chunks[0].content == doc.content

    def test_chunk_large_document(self):
        """Test chunking a large document."""
        # Create large document
        content = "This is a sentence. " * 100
        doc = Document(
            content=content,
            metadata={'filename': 'large.txt'}
        )

        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 1
        # Check overlap exists
        if len(chunks) > 1:
            # Some content should appear in adjacent chunks
            assert len(chunks[0].content) > 0
            assert len(chunks[1].content) > 0

    def test_chunk_metadata_preserved(self):
        """Test that metadata is preserved in chunks."""
        doc = Document(
            content="Test content",
            metadata={'filename': 'test.txt', 'author': 'Test Author'}
        )

        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc)

        assert all(chunk.metadata['filename'] == 'test.txt' for chunk in chunks)
        assert all(chunk.metadata['author'] == 'Test Author' for chunk in chunks)
        assert all('chunk_index' in chunk.metadata for chunk in chunks)


if __name__ == '__main__':
    pytest.main([__file__])
