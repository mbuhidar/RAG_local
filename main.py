"""
Main Application Entry Point

Provides CLI commands for RAG system operations.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.chat.interface import ChatInterface
from src.rag.pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)


def ingest_command(args):
    """Ingest documents into the system."""
    logger.info("Starting document ingestion")
    
    pipeline = RAGPipeline(config_path=args.config)
    
    document_path = args.path or "data/documents"
    num_chunks = pipeline.ingest_documents(
        document_path=document_path,
        recursive=args.recursive
    )
    
    logger.info(f"Ingestion complete: {num_chunks} chunks indexed")
    print(f"\n✓ Successfully ingested {num_chunks} document chunks")


def query_command(args):
    """Query the RAG system."""
    pipeline = RAGPipeline(config_path=args.config)
    
    question = args.question
    logger.info(f"Processing query: {question}")
    
    result = pipeline.query(
        question=question,
        return_sources=args.show_sources
    )
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {result['answer']}")
    
    if args.show_sources and "sources" in result:
        print("\n--- Sources ---")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. {source['metadata'].get('filename', 'Unknown')}")
            print(f"   Relevance: {source['score']:.3f}")
            print(f"   Preview: {source['text'][:150]}...")


def chat_command(args):
    """Launch chat interface."""
    logger.info("Starting chat interface")
    
    pipeline = RAGPipeline(config_path=args.config)
    interface = ChatInterface(pipeline)
    
    interface.launch(
        host=args.host,
        port=args.port,
        share=args.share
    )


def stats_command(args):
    """Show system statistics."""
    pipeline = RAGPipeline(config_path=args.config)
    stats = pipeline.get_stats()
    
    print("\n=== RAG System Statistics ===\n")
    
    print("Vector Store:")
    print(f"  Collection: {stats['vector_store']['name']}")
    print(f"  Documents: {stats['vector_store']['count']}")
    
    print("\nEmbedding Model:")
    print(f"  Model: {stats['embedding_model']['model_name']}")
    print(f"  Dimensions: {stats['embedding_model']['embedding_dimension']}")
    print(f"  Device: {stats['embedding_model']['device']}")
    
    print("\nLLM:")
    print(f"  Provider: {stats['llm']['provider']}")
    print(f"  Model: {stats['llm']['model_name']}")
    print(f"  Base URL: {stats['llm']['base_url']}")
    print(f"  Temperature: {stats['llm']['temperature']}")


def clear_command(args):
    """Clear the vector database."""
    if not args.confirm:
        response = input("Are you sure you want to clear all documents? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled")
            return
    
    pipeline = RAGPipeline(config_path=args.config)
    success = pipeline.clear_database()
    
    if success:
        print("✓ Database cleared successfully")
    else:
        print("✗ Failed to clear database")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Local RAG Chat Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest documents
  python main.py ingest --path data/documents

  # Query the system
  python main.py query "What is machine learning?"

  # Launch chat interface
  python main.py chat

  # Show statistics
  python main.py stats
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument(
        '--path',
        type=str,
        help='Path to documents directory'
    )
    ingest_parser.add_argument(
        '--recursive',
        action='store_true',
        default=True,
        help='Search subdirectories'
    )
    ingest_parser.set_defaults(func=ingest_command)
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the system')
    query_parser.add_argument(
        'question',
        type=str,
        help='Question to ask'
    )
    query_parser.add_argument(
        '--show-sources',
        action='store_true',
        default=True,
        help='Show source documents'
    )
    query_parser.set_defaults(func=query_command)
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Launch chat interface')
    chat_parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host address'
    )
    chat_parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port number'
    )
    chat_parser.add_argument(
        '--share',
        action='store_true',
        help='Create public link'
    )
    chat_parser.set_defaults(func=chat_command)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    stats_parser.set_defaults(func=stats_command)
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear database')
    clear_parser.add_argument(
        '--confirm',
        action='store_true',
        help='Skip confirmation prompt'
    )
    clear_parser.set_defaults(func=clear_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    setup_logging(args.verbose)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
