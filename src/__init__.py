"""
RAG Local - Local RAG Chat Application

A complete RAG system that runs entirely on local infrastructure.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from src.chat import ChatInterface
from src.rag import RAGPipeline

__all__ = ['RAGPipeline', 'ChatInterface']
