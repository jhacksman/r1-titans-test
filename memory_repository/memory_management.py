import numpy as np
import time
from typing import List, Optional, Dict, Any
from annoy import AnnoyIndex
from .memory_store import MemoryStore

def surprise_score(current_state: np.ndarray, reference_states: Optional[List[np.ndarray]] = None) -> float:
    """Calculate surprise score for current hidden state.
    
    Uses a combination of:
    1. L2 norm of the hidden state (basic activity measure)
    2. If reference states provided, minimum cosine distance to previous states
    
    Args:
        current_state: Current hidden state vector
        reference_states: Optional list of previous hidden states for comparison
        
    Returns:
        float: Surprise score (higher = more surprising)
    """
    # Basic activity measure
    activity_score = np.linalg.norm(current_state)
    
    if reference_states and len(reference_states) > 0:
        # Calculate cosine similarities with reference states
        current_norm = np.linalg.norm(current_state)
        if current_norm == 0:
            return 0.0
            
        current_normalized = current_state / current_norm
        
        max_similarity = -1.0
        for ref_state in reference_states:
            ref_norm = np.linalg.norm(ref_state)
            if ref_norm == 0:
                continue
                
            ref_normalized = ref_state / ref_norm
            similarity = np.dot(current_normalized, ref_normalized)
            max_similarity = max(max_similarity, similarity)
        
        # Convert similarity to distance (1 = most surprising, 0 = least surprising)
        novelty_score = 1.0 - max(0.0, max_similarity)
        
        # Combine activity and novelty scores
        return 0.5 * (activity_score + novelty_score)
    
    return activity_score

def forget_stale_entries(memory_store: MemoryStore, 
                        age_threshold: float = 3600,
                        max_memories: int = 10000,
                        min_surprise_score: float = 0.3) -> int:
    """Remove old or excess memories from the store.
    
    Uses three criteria in order:
    1. Time-based pruning: Remove entries older than age_threshold
    2. Surprise-based pruning: Remove entries with low surprise scores
    3. Capacity-based pruning: If still over max_memories, remove oldest entries
    
    Args:
        memory_store: The MemoryStore instance to prune
        age_threshold: Maximum age in seconds before memory is considered stale
        max_memories: Maximum number of memories to keep
        min_surprise_score: Minimum surprise score to keep (if available)
        
    Returns:
        int: Number of memories removed
    """
    current_time = time.time()
    to_remove: List[str] = []
    
    # First pass: Remove entries older than threshold
    for key in memory_store.storage.keys():
        timestamp = memory_store.get_timestamp(key)
        if timestamp and (current_time - timestamp) > age_threshold:
            to_remove.append(key)
    
    # Second pass: Remove entries with low surprise scores
    remaining_entries = [
        (k, memory_store.storage[k]) 
        for k in memory_store.storage.keys() 
        if k not in to_remove
    ]
    
    for key, data in remaining_entries:
        surprise_score = data.get('metadata', {}).get('surprise_score', 1.0)
        if surprise_score < min_surprise_score:
            to_remove.append(key)
    
    # Third pass: If still over capacity, remove oldest entries
    if len(memory_store.storage) - len(to_remove) > max_memories:
        # Sort remaining entries by timestamp
        entries = [(k, memory_store.get_timestamp(k)) 
                  for k in memory_store.storage.keys() 
                  if k not in to_remove]
        entries.sort(key=lambda x: x[1] or 0)  # Sort by timestamp
        
        # Add oldest entries to removal list until we're under max_memories
        excess = len(memory_store.storage) - len(to_remove) - max_memories
        to_remove.extend([k for k, _ in entries[:excess]])
    
    # Remove all identified entries
    removed_count = 0
    for key in to_remove:
        try:
            del memory_store.storage[key]
            removed_count += 1
        except KeyError:
            continue  # Skip if already removed
    
    # If we removed any entries, rebuild the Annoy index
    if removed_count > 0:
        # Create new index with remaining items
        memory_store.index = AnnoyIndex(memory_store.vector_dim, 'angular')
        
        for key, data in memory_store.storage.items():
            memory_store.index.add_item(data['index'], data['embedding'])
        
        memory_store.index.build(memory_store.n_trees)
    
    return removed_count
