import numpy as np
import time
import os
import json
from typing import Dict, Any, Optional, Tuple, List
from annoy import AnnoyIndex

class MemoryStore:
    """Memory store implementation using Annoy for efficient similarity search.
    
    This class provides the core functionality for storing and retrieving memory
    embeddings, optimized for the DeepSeek R1 32B model's memory requirements.
    It uses the Annoy library for approximate nearest neighbor search, which
    provides a good balance between search speed and memory usage.
    
    Key Features:
    - Efficient similarity search with Annoy
    - Metadata storage for each memory
    - Timestamp tracking for memory management
    - VRAM-aware implementation
    """
    
    def __init__(self, vector_dim: int, n_trees: int = 10, max_memory_gb: float = 32.0):
        """Initialize the memory store with Annoy index for fast retrieval.
        
        Args:
            vector_dim: Dimension of the embedding vectors (matches R1's hidden dim)
            n_trees: Number of trees for Annoy index (more trees = better accuracy but slower build)
            max_memory_gb: Maximum memory usage in GB (default 32GB, leaving room for R1)
        """
        self.vector_dim = vector_dim
        self.n_trees = n_trees
        self.max_memory_bytes = int(max_memory_gb * (1024 ** 3))
        self.storage: Dict[str, Dict[str, Any]] = {}  # Stores metadata
        self.index = AnnoyIndex(vector_dim, 'angular')  # For fast similarity search
        self.current_index = 0  # Counter for generating unique IDs
        
        # Track memory usage
        self._total_memory_usage = 0
        self._bytes_per_vector = vector_dim * np.dtype(np.float32).itemsize
        
    def add_memory(self, embedding: np.ndarray, metadata: Optional[Dict] = None, auto_build: bool = True) -> Optional[str]:
        """Add a new memory embedding to the store.
        
        Args:
            embedding: The vector embedding to store
            metadata: Optional dictionary of additional information
            auto_build: If True, rebuild index after adding item (slower but ensures retrievability)
            
        Returns:
            key: Unique identifier for the stored memory
        """
        if embedding.shape[0] != self.vector_dim:
            raise ValueError(f"Expected embedding dim {self.vector_dim}, got {embedding.shape[0]}")
            
        # Check memory limit
        if self.check_memory_limit():
            return None
            
        key = f"mem_{self.current_index}"
        self.index.add_item(self.current_index, embedding)
        
        self.storage[key] = {
            'timestamp': time.time(),
            'embedding': embedding,
            'metadata': metadata or {},
            'index': self.current_index
        }
        
        self.current_index += 1
        
        if auto_build:
            self.build_index()
            
        return key
        
    def build_index(self):
        """Build the Annoy index. Must be called after adding items and before querying.
        
        This is an expensive operation, so if adding multiple items,
        set auto_build=False in add_memory() and call this manually after
        all items are added.
        """
        # Create new index if current one is built
        try:
            self.index.build(self.n_trees)
        except Exception as e:
            if "You can't build a built index" in str(e):
                # Create new index and re-add all items
                new_index = AnnoyIndex(self.vector_dim, 'angular')
                for key, data in self.storage.items():
                    new_index.add_item(data['index'], data['embedding'])
                self.index = new_index
                self.index.build(self.n_trees)
            else:
                raise
        
    def batch_add_memories(self, embeddings: List[np.ndarray], 
                         metadata: Optional[List[Dict]] = None) -> List[str]:
        """Add multiple memory embeddings efficiently.
        
        Args:
            embeddings: List of embeddings to store
            metadata: Optional list of metadata dicts (must match embeddings length if provided)
            
        Returns:
            List of memory keys
        """
        if metadata and len(metadata) != len(embeddings):
            raise ValueError("metadata length must match embeddings length")
            
        keys = []
        for i, emb in enumerate(embeddings):
            meta = metadata[i] if metadata else None
            # Don't auto-build for each addition
            key = self.add_memory(emb, meta, auto_build=False)
            keys.append(key)
            
        # Build index once at the end
        self.build_index()
        return keys
        
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
        
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage statistics in bytes.
        
        Returns:
            Dict containing:
            - vectors: Memory used by embedding vectors
            - metadata: Approximate memory used by metadata
            - index: Approximate memory used by Annoy index
            - total: Total memory usage
        """
        n_vectors = len(self.storage)
        vectors_size = n_vectors * self._bytes_per_vector
        
        # Rough metadata size estimation (timestamps, keys, etc.)
        metadata_size = sum(
            len(str(v).encode()) + 128  # 128 bytes overhead per entry
            for v in self.storage.values()
        )
        
        # Annoy index size estimation (varies by n_trees)
        index_size = vectors_size * self.n_trees * 1.2  # 1.2x overhead factor
        
        total = vectors_size + metadata_size + index_size
        
        return {
            'vectors': vectors_size,
            'metadata': metadata_size,
            'index': int(index_size),
            'total': total
        }
        
    def check_memory_limit(self, new_vectors_count: int = 1) -> bool:
        """Check if adding new vectors would exceed memory limit.
        
        Args:
            new_vectors_count: Number of new vectors to be added
            
        Returns:
            True if memory limit would be exceeded, False otherwise
        """
        usage = self.get_memory_usage()
        new_usage = usage['total'] + (new_vectors_count * self._bytes_per_vector * (1 + self.n_trees * 1.2))
        return new_usage > self.max_memory_bytes
        
    def save_state(self, directory: str):
        """Save memory store state to disk.
        
        This saves both the Annoy index and the metadata storage.
        Note that this is optional functionality and not required
        for basic memory operations.
        
        Args:
            directory: Directory to save state files in
        """
        import os
        import json
        
        os.makedirs(directory, exist_ok=True)
        
        # Save Annoy index
        index_path = os.path.join(directory, 'memory.ann')
        self.index.save(index_path)
        
        # Save metadata
        metadata_path = os.path.join(directory, 'metadata.json')
        metadata = {
            'vector_dim': self.vector_dim,
            'n_trees': self.n_trees,
            'current_index': self.current_index,
            'storage': {
                k: {
                    'timestamp': v['timestamp'],
                    'metadata': v['metadata'],
                    'index': v['index']
                }
                for k, v in self.storage.items()
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        # Save embeddings separately (they're numpy arrays)
        embeddings_path = os.path.join(directory, 'embeddings.npz')
        embeddings = {
            k: v['embedding'] for k, v in self.storage.items()
        }
        np.savez_compressed(embeddings_path, **embeddings)
        
    def load_state(self, directory: str):
        """Load memory store state from disk.
        
        This loads both the Annoy index and the metadata storage.
        Note that this is optional functionality and not required
        for basic memory operations.
        
        Args:
            directory: Directory containing saved state files
        """
        import os
        import json
        
        # Load metadata first
        metadata_path = os.path.join(directory, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Verify compatibility
        if metadata['vector_dim'] != self.vector_dim:
            raise ValueError(f"Incompatible vector dimensions: {metadata['vector_dim']} vs {self.vector_dim}")
            
        # Load embeddings
        embeddings_path = os.path.join(directory, 'embeddings.npz')
        embeddings = np.load(embeddings_path)
        
        # Reset state
        self.storage.clear()
        self.index = AnnoyIndex(self.vector_dim, 'angular')
        
        # Restore metadata and embeddings
        for key, meta in metadata['storage'].items():
            self.storage[key] = {
                'timestamp': meta['timestamp'],
                'embedding': embeddings[key],
                'metadata': meta['metadata'],
                'index': meta['index']
            }
            self.index.add_item(meta['index'], embeddings[key])
            
        self.current_index = metadata['current_index']
        self.n_trees = metadata['n_trees']
        
        # Rebuild index
        self.build_index()
