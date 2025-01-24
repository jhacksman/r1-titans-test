import pytest
import numpy as np
import time
from memory_repository.memory_store import MemoryStore
from memory_repository.memory_management import forget_stale_entries

def test_memory_store_basic():
    """Test basic memory store operations."""
    # Initialize store
    dim = 128
    store = MemoryStore(vector_dim=dim)
    
    # Create test embedding
    embedding = np.random.random(dim).astype(np.float32)
    
    # Add and verify
    key = store.add_memory(embedding, metadata={'test': True})
    assert key is not None
    
    # Retrieve and verify
    keys, distances = store.retrieve(embedding, k=1)
    assert len(keys) == 1
    assert len(distances) == 1
    assert distances[0] < 1e-6  # Should be very close
    
    # Check metadata
    memory = store.get_memory(keys[0])
    assert memory is not None
    assert memory['metadata']['test'] is True

def test_memory_store_batch():
    """Test batch memory operations."""
    dim = 128
    store = MemoryStore(vector_dim=dim)
    
    # Create test embeddings
    embeddings = [np.random.random(dim).astype(np.float32) for _ in range(5)]
    metadata = [{'index': i} for i in range(5)]
    
    # Batch add
    keys = store.batch_add_memories(embeddings, metadata)
    assert len(keys) == 5
    
    # Verify all were added
    for i, emb in enumerate(embeddings):
        retrieved_keys, _ = store.retrieve(emb, k=1)
        memory = store.get_memory(retrieved_keys[0])
        assert memory['metadata']['index'] == i

def test_memory_store_forgetting():
    """Test memory forgetting mechanism."""
    dim = 128
    store = MemoryStore(vector_dim=dim)
    
    # Add memories with different timestamps
    embeddings = []
    for i in range(10):
        emb = np.random.random(dim).astype(np.float32)
        embeddings.append(emb)
        key = store.add_memory(emb, metadata={'index': i})
        # Simulate time passing
        store.storage[key]['timestamp'] -= (i * 1000)  # Make each progressively older
    
    # Should remove oldest entries
    removed = forget_stale_entries(store, age_threshold=5000, max_memories=5)
    assert removed == 5  # Should remove 5 oldest entries
    
    # Verify remaining entries
    assert len(store.storage) == 5
    
    # Check that we can still retrieve from remaining entries
    for emb in embeddings[:5]:  # These should be the newer ones
        keys, _ = store.retrieve(emb, k=1)
        assert len(keys) == 1

def test_memory_store_vram():
    """Test memory store VRAM limits."""
    dim = 1024  # Larger dimension to test memory limits
    store = MemoryStore(vector_dim=dim, max_memory_gb=0.001)  # Very small limit for testing
    
    # Try to add many embeddings
    count = 0
    while count < 1000:  # Safety limit
        emb = np.random.random(dim).astype(np.float32)
        key = store.add_memory(emb)
        if key is None:  # Should hit memory limit
            break
        count += 1
    
    assert count < 1000  # Should have hit limit
    
    # Verify memory usage
    usage = store.get_memory_usage()
    assert usage['total'] <= store.max_memory_bytes

def test_memory_store_persistence(tmp_path):
    """Test save/load functionality."""
    dim = 128
    store = MemoryStore(vector_dim=dim)
    
    # Add some test data
    embeddings = [np.random.random(dim).astype(np.float32) for _ in range(5)]
    metadata = [{'index': i} for i in range(5)]
    keys = store.batch_add_memories(embeddings, metadata)
    
    # Save state
    save_dir = tmp_path / "memory_store"
    store.save_state(str(save_dir))
    
    # Create new store and load state
    new_store = MemoryStore(vector_dim=dim)
    new_store.load_state(str(save_dir))
    
    # Verify data was preserved
    for i, emb in enumerate(embeddings):
        keys, _ = new_store.retrieve(emb, k=1)
        memory = new_store.get_memory(keys[0])
        assert memory['metadata']['index'] == i

def test_memory_store_surprise():
    """Test surprise-based memory management."""
    dim = 128
    store = MemoryStore(vector_dim=dim)
    
    # Add memories with different surprise scores
    embeddings = []
    for i in range(10):
        emb = np.random.random(dim).astype(np.float32)
        embeddings.append(emb)
        surprise = i / 10.0  # Increasing surprise scores
        store.add_memory(emb, metadata={'surprise_score': surprise})
    
    # Should remove low surprise entries
    removed = forget_stale_entries(
        store, 
        age_threshold=float('inf'),  # Don't remove based on age
        max_memories=10,  # Don't remove based on capacity
        min_surprise_score=0.5,  # Remove entries with surprise < 0.5
        rebuild_index=True  # Test index rebuilding
    )
    
    assert removed == 5  # Should remove 5 entries with lowest surprise
    assert len(store.storage) == 5
    
    # Test without index rebuilding
    store = MemoryStore(vector_dim=dim)
    for i in range(10):
        emb = np.random.random(dim).astype(np.float32)
        store.add_memory(emb, metadata={'surprise_score': i/10.0})
        
    removed = forget_stale_entries(
        store,
        age_threshold=float('inf'),
        max_memories=5,
        min_surprise_score=0.0,  # Don't remove based on surprise
        rebuild_index=False  # Skip index rebuilding
    )
    
    assert removed == 5
    assert len(store.storage) == 5
    
    # Verify index state
    assert store.index is not None  # Index should still exist
    assert store.index.get_n_items() > 0  # But may be stale
    
    # Test error handling with invalid parameters
    with pytest.raises(ValueError):
        forget_stale_entries(store, max_memories=-1)
    with pytest.raises(ValueError):
        forget_stale_entries(store, age_threshold=-1)
