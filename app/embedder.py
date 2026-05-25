"""
Embedder Module
Generates embeddings using sentence-transformers.
Uses all-MiniLM-L6-v2 model for efficient semantic representation.
"""

import logging
import os
from typing import List, Dict, Optional, Tuple
from typing import Any as any
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generates semantic embeddings for text using sentence-transformers.
    
    Attributes:
        model_name: Name of sentence-transformer model
        model: Loaded SentenceTransformer model
        embedding_dim: Dimension of embeddings
    """
    
    # Default model
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        """
        Initialize Embedder with sentence-transformer model.
        
        Args:
            model_name: Model identifier (default: all-MiniLM-L6-v2)
            device: Device to use ('cpu', 'cuda', 'mps')
            
        Raises:
            ValueError: If model cannot be loaded
        """
        self.model_name = model_name
        self.device = device
        
        try:
            logger.info(f"Loading embedding model: {model_name} on device: {device}")
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim: int = self.model.get_sentence_embedding_dimension() or 384
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise ValueError(f"Cannot load embedding model {model_name}: {str(e)}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (default 64)
            show_progress: Show progress bar (default True)
            
        Returns:
            Array of embeddings (shape: [n_texts, embedding_dim])
        """
        if not texts:
            logger.warning("Empty text list provided")
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        # Filter empty texts
        texts = [t if t and t.strip() else "" for t in texts]
        
        try:
            logger.info(f"Embedding {len(texts)} texts with batch size {batch_size}")
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )
            logger.info(f"Successfully embedded {len(texts)} texts")
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error embedding batch: {str(e)}")
            raise
    
    def embed_chunks(
        self,
        chunks: List[Dict[str, any]],
        text_field: str = "chunk_text",
        batch_size: int = 64
    ) -> Tuple[np.ndarray, List[Dict[str, any]]]:
        """
        Embed text chunks from chunker output.
        
        Args:
            chunks: List of chunk dictionaries
            text_field: Field name containing text (default "chunk_text")
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (embeddings, chunks_with_ids)
        """
        if not chunks:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []
        
        # Extract texts
        texts = [chunk.get(text_field, "") for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts, batch_size=batch_size)
        
        # Add embedding metadata to chunks
        chunks_with_embeddings = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            chunk_copy["embedding"] = embeddings[i]
            chunks_with_embeddings.append(chunk_copy)
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return embeddings, chunks_with_embeddings
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "max_seq_length": self.model.max_seq_length if hasattr(self.model, 'max_seq_length') else "Unknown"
        }
    
    @staticmethod
    def compute_similarity(
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ('cosine', 'euclidean', 'dot')
            
        Returns:
            Similarity score (0-1 for cosine, varies for others)
        """
        if metric == "cosine":
            # Cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            # Ensure in [0, 1] range
            return float(np.clip(similarity, 0, 1))
        
        elif metric == "euclidean":
            distance = np.linalg.norm(embedding1 - embedding2)
            # Convert to similarity (inverse relationship)
            return float(1.0 / (1.0 + distance))
        
        elif metric == "dot":
            return float(np.dot(embedding1, embedding2))
        
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def embed_and_normalize(self, text: str) -> np.ndarray:
        """
        Generate and normalize embedding.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector
        """
        embedding = self.embed_text(text)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
