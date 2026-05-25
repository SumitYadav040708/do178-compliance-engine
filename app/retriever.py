"""
Retriever Module
Manages FAISS vector database for semantic search and retrieval.
Handles index creation, persistence, and similarity-based retrieval.
"""

import logging
import json
import os
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSRetriever:
    """
    Manages FAISS index for efficient semantic retrieval.
    
    Attributes:
        index_path: Path to FAISS index file
        metadata_path: Path to metadata JSON
        index: FAISS index object
        metadata: Chunk metadata list
        embedding_dim: Dimension of embeddings
    """
    
    def __init__(
        self,
        index_path: str = "indexes/reference_index.faiss",
        metadata_path: str = "indexes/metadata.json"
    ):
        """
        Initialize FAISS Retriever.
        
        Args:
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load metadata
        """
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.index_path = os.path.join(BASE_DIR, "indexes", "reference_index.faiss")
        self.metadata_path = os.path.join(BASE_DIR, "indexes", "metadata.json")
        self.index = None
        self.metadata = []
        self.embedding_dim = None
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        
        logger.info(f"FAISS Retriever initialized (index: {index_path})")
    
    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        metric: str = "ip"  # inner product for normalized embeddings
    ) -> None:
        """
        Build FAISS index from embeddings and metadata.
        
        Args:
            embeddings: Array of embeddings (shape: [n_samples, embedding_dim])
            metadata: List of metadata dicts for each embedding
            metric: Metric type ('ip' for inner product, 'l2' for L2 distance)
            
        Raises:
            ValueError: If embeddings and metadata length mismatch
        """
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) "
                "must have same length"
            )
        
        if len(embeddings) == 0:
            logger.warning("Empty embeddings provided")
            return
        
        self.embedding_dim = embeddings.shape[1]
        embeddings = embeddings.astype(np.float32)
        
        logger.info(
            f"Building FAISS index: {len(embeddings)} embeddings, "
            f"dimension {self.embedding_dim}"
        )
        
        # Create index (simple flat index for reliability)
        if metric == "ip":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:  # l2
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Normalize embeddings for inner product metric
        if metric == "ip":
            faiss.normalize_L2(embeddings)
        
        # Add vectors to index
        self.index.add(embeddings)  # type: ignore
        self.metadata = metadata
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
    
    def save_index(self) -> bool:
        """
        Persist index and metadata to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.index is None:
                logger.error("No index to save")
                return False
            
            # Save FAISS index
            os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Saved FAISS index to {self.index_path}")
            
            # Save metadata
            os.makedirs(os.path.dirname(self.metadata_path) or ".", exist_ok=True)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metadata to {self.metadata_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self) -> bool:
        """
        Load persisted index and metadata from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.index_path):
                logger.warning(f"Index file not found: {self.index_path}")
                return False
            
            if not os.path.exists(self.metadata_path):
                logger.warning(f"Metadata file not found: {self.metadata_path}")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            self.embedding_dim = self.index.d
            logger.info(
                f"Loaded FAISS index: {self.index.ntotal} vectors, "
                f"dimension {self.embedding_dim}"
            )
            
            # Load metadata
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded {len(self.metadata)} metadata entries")
            
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar chunks from index.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity score to include
            
        Returns:
            List of result dicts with metadata and similarity score
        """
        if self.index is None:
            logger.error("No index loaded")
            return []
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Normalize for inner product
        faiss.normalize_L2(query_embedding)
        
        try:
            # Search
            distances, indices = self.index.search(query_embedding, k)  # type: ignore
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                # Convert inner product score to similarity [0, 1]
                # Inner product of normalized vectors is cosine similarity
                similarity = float(distance)
                similarity = np.clip(similarity, 0, 1)
                
                if similarity < threshold:
                    continue
                
                if idx >= len(self.metadata):
                    logger.warning(f"Index out of bounds: {idx}")
                    continue
                
                result = self.metadata[idx].copy()
                result["similarity_score"] = similarity
                result["rank"] = len(results) + 1
                results.append(result)
            
            logger.debug(f"Retrieved {len(results)} results for query")
            return results
        
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
        threshold: float = 0.0
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for multiple queries.
        
        Args:
            query_embeddings: Array of query embeddings
            k: Number of results per query
            threshold: Minimum similarity score
            
        Returns:
            List of result lists, one per query
        """
        if self.index is None:
            logger.error("No index loaded")
            return []
        
        results_batch = []
        for query_emb in query_embeddings:
            results = self.search(query_emb, k=k, threshold=threshold)
            results_batch.append(results)
        
        return results_batch
    
    def add_chunks(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """
        Add new chunks to existing index.
        
        Args:
            embeddings: New embeddings to add
            metadata: Metadata for new embeddings
            
        Returns:
            True if successful
        """
        try:
            if self.index is None:
                # Create new index
                self.build_index(embeddings, metadata)
                return True
            
            if len(embeddings) != len(metadata):
                raise ValueError("Embeddings and metadata length mismatch")
            
            embeddings = embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings)
            
            self.index.add(embeddings)  # type: ignore
            self.metadata.extend(metadata)
            
            logger.info(f"Added {len(embeddings)} chunks. Total: {self.index.ntotal}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding chunks: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {"status": "No index loaded"}
        
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "metadata_entries": len(self.metadata),
            "index_path": self.index_path,
            "metadata_path": self.metadata_path
        }
    
    def delete_index(self) -> bool:
        """
        Delete persisted index and metadata files.
        
        Returns:
            True if successful
        """
        try:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
                logger.info(f"Deleted index file: {self.index_path}")
            
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
                logger.info(f"Deleted metadata file: {self.metadata_path}")
            
            self.index = None
            self.metadata = []
            return True
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            return False
