import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
from annoy import AnnoyIndex

class MemoryStore:
    def __init__(self, vector_dim: int, n_trees: int = 10):
        """Initialize the memory store with Annoy index for fast retrieval.
        
        Args:
            vector_dim: Dimension of the embedding vectors
            n_trees: Number of trees for Annoy index (more trees = better accuracy but slower build)
        """
        self.vector_dim = vector_dim
        self.n_trees = n_trees
        self.storage: Dict[str, Dict[str, Any]] = {}  # Stores metadata
        self.index = AnnoyIndex(vector_dim, 'angular')  # For fast similarity search
        self.current_index = 0  # Counter for generating unique IDs
        
    def add_memory(self, embedding: np.ndarray, metadata: Optional[Dict] = None) -> str:
        """Add a new memory embedding to the store.
        
        Args:
            embedding: The vector embedding to store
            metadata: Optional dictionary of additional information
            
        Returns:
            key: Unique identifier for the stored memory
        """
        if embedding.shape[0] != self.vector_dim:
            raise ValueError(f"Expected embedding dim {self.vector_dim}, got {embedding.shape[0]}")
            
        key = f"mem_{self.current_index}"
        self.index.add_item(self.current_index, embedding)
        
        self.storage[key] = {
            'timestamp': time.time(),
            'embedding': embedding,
            'metadata': metadata or {},
            'index': self.current_index
        }
        
        self.current_index += 1
        return key
        
    def build_index(self):
        """Build the Annoy index. Must be called after adding items and before querying."""
        self.index.build(self.n_trees)
        
    def retrieve(self, query: np.ndarray, k: int = 5) -> Tuple[list, list]:
        """Retrieve the k nearest memories to the query vector.
        
        Args:
            query: Query vector
            k: Number of nearest neighbors to retrieve
            
        Returns:
            Tuple of (memory_keys, distances)
        """
        if query.shape[0] != self.vector_dim:
            raise ValueError(f"Expected query dim {self.vector_dim}, got {query.shape[0]}")
            
        # Get nearest neighbors from Annoy index
        indices, distances = self.index.get_nns_by_vector(query, k, include_distances=True)
        
        # Convert indices back to memory keys
        keys = []
        for idx in indices:
            for key, data in self.storage.items():
                if data['index'] == idx:
                    keys.append(key)
                    break
                    
        return keys, distances
        
    def get_memory(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve full memory data by key."""
        return self.storage.get(key)
        
    def get_timestamp(self, key: str) -> Optional[float]:
        """Get the timestamp of when a memory was added."""
        memory = self.storage.get(key)
        return memory['timestamp'] if memory else None
