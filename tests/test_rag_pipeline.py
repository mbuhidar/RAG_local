"""
Test suite for RAG pipeline.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from src.rag.pipeline import RAGPipeline


class TestRAGPipeline:
    """Tests for RAG pipeline."""

    @patch('src.rag.pipeline.yaml.safe_load')
    @patch('builtins.open')
    def test_pipeline_initialization(self, mock_open, mock_yaml):
        """Test pipeline initialization."""
        # Mock config
        mock_yaml.return_value = {
            'document_processing': {
                'supported_formats': ['txt', 'pdf'],
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'separators': ['\n\n', '\n']
            },
            'embeddings': {
                'model_name': 'test-model',
                'model_kwargs': {'device': 'cpu'},
                'cache_folder': './cache',
                'encode_kwargs': {'batch_size': 32, 'normalize_embeddings': True}
            },
            'vector_db': {
                'collection_name': 'test',
                'persist_directory': './test_db',
                'distance_metric': 'cosine'
            },
            'llm': {
                'model_name': 'llama2',
                'base_url': 'http://localhost:11434',
                'temperature': 0.7,
                'max_tokens': 2048,
                'context_window': 4096
            },
            'retrieval': {
                'top_k': 5,
                'score_threshold': 0.7
            },
            'rag': {
                'prompt_template': 'Context: {context}\n\nQuestion: {question}\n\nAnswer:'
            }
        }

        with patch('src.rag.pipeline.DocumentLoader'), \
             patch('src.rag.pipeline.DocumentChunker'), \
             patch('src.rag.pipeline.EmbeddingGenerator'), \
             patch('src.rag.pipeline.ChromaVectorStore'), \
             patch('src.rag.pipeline.OllamaLLM'):
            
            pipeline = RAGPipeline('config/test.yaml')
            assert pipeline.config is not None


if __name__ == '__main__':
    pytest.main([__file__])
