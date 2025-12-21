"""
Embeddings Module

Handles generation of embeddings using local models.
Supports sentence-transformers and other local embedding providers.
"""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using local models."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_folder: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: Name of the sentence-transformer model
            device: Device to run on (cpu, cuda, mps)
            cache_folder: Folder to cache models
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.cache_folder = cache_folder

        if cache_folder:
            Path(cache_folder).mkdir(parents=True, exist_ok=True)

        self._load_model()

    def _load_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            cache_folder=self.cache_folder
        )
        logger.info(f"Model loaded on {self.device}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return embedding.tolist()

    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.model.get_sentence_embedding_dimension()

    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.get_embedding_dimension(),
            "max_sequence_length": self.model.max_seq_length,
            "batch_size": self.batch_size
        }
