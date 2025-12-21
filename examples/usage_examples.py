"""
Example script demonstrating RAG system usage.
"""

import logging

from src.chat.interface import ChatInterface
from src.rag.pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Basic RAG pipeline usage."""
    logger.info("=== Example 1: Basic Usage ===")
    
    # Initialize pipeline
    pipeline = RAGPipeline('config/config.yaml')
    
    # Ingest documents
    num_chunks = pipeline.ingest_documents('data/documents')
    logger.info(f"Ingested {num_chunks} chunks")
    
    # Query
    result = pipeline.query("What is the main topic?")
    print(f"\nAnswer: {result['answer']}")


def example_custom_retrieval():
    """Example with custom retrieval parameters."""
    logger.info("=== Example 2: Custom Retrieval ===")
    
    pipeline = RAGPipeline('config/config.yaml')
    
    # Custom retrieval
    results = pipeline.retrieve_context(
        query="machine learning",
        top_k=3,
        score_threshold=0.8
    )
    
    print(f"\nFound {len(results)} relevant chunks")
    for i, (text, score, meta) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.3f}")
        print(f"   File: {meta.get('filename', 'Unknown')}")
        print(f"   Preview: {text[:100]}...")


def example_chat_interface():
    """Example launching chat interface."""
    logger.info("=== Example 3: Chat Interface ===")
    
    pipeline = RAGPipeline('config/config.yaml')
    interface = ChatInterface(pipeline)
    
    # Launch on localhost
    interface.launch(port=7860)


if __name__ == '__main__':
    # Run examples
    # example_basic_usage()
    # example_custom_retrieval()
    example_chat_interface()
