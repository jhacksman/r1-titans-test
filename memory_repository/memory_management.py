import numpy as np
import time
from typing import List, Optional, Dict, Any, Tuple, Union
from annoy import AnnoyIndex
from .memory_store import MemoryStore

def to_probability_distribution(hidden_state: np.ndarray, 
                             temperature: float = 1.0,
                             eps: float = 1e-9) -> np.ndarray:
    """Convert hidden state to probability distribution using softmax.
    
    Args:
        hidden_state: Input hidden state vector
        temperature: Softmax temperature (higher = more uniform)
        eps: Small constant for numerical stability
        
    Returns:
        Probability distribution over hidden state dimensions
        
    Implementation:
    1. Scale and shift for numerical stability
    2. Apply softmax with temperature
    3. Ensure valid probability distribution
    """
    # Handle edge case of zero vector
    if np.all(hidden_state == 0):
        return np.ones_like(hidden_state) / hidden_state.shape[0]
    
    # Scale inputs for numerical stability
    x = hidden_state / temperature
    x = x - np.max(x)  # Shift for numerical stability
    
    # Compute softmax
    exp_x = np.exp(x)
    probs = exp_x / (np.sum(exp_x) + eps)
    
    # Ensure valid probability distribution
    probs = np.maximum(probs, eps)  # Avoid exact zeros
    probs = probs / np.sum(probs)  # Renormalize
    
    return probs

def batch_to_probability_distributions(hidden_states: np.ndarray,
                                    temperature: float = 1.0,
                                    eps: float = 1e-9) -> np.ndarray:
    """Convert batch of hidden states to probability distributions.
    
    Args:
        hidden_states: Batch of hidden states [batch_size, hidden_dim]
        temperature: Softmax temperature
        eps: Small constant for numerical stability
        
    Returns:
        Batch of probability distributions
    """
    if hidden_states.ndim == 1:
        return to_probability_distribution(hidden_states, temperature, eps)
    
    # Process batch dimension
    return np.stack([
        to_probability_distribution(state, temperature, eps)
        for state in hidden_states
    ])

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """Compute KL divergence between two probability distributions.
    
    Implementation uses a numerically stable formula:
    KL(P||Q) = sum(P[i] * log((P[i] + eps)/(Q[i] + eps)))
    
    Args:
        p: First probability distribution
        q: Second probability distribution (reference)
        eps: Small constant for numerical stability
        
    Returns:
        KL divergence value
    """
    # Ensure valid probability distributions
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    
    # Normalize if needed
    if not np.isclose(np.sum(p), 1.0):
        p = p / np.sum(p)
    if not np.isclose(np.sum(q), 1.0):
        q = q / np.sum(q)
    
    # Compute KL divergence in a numerically stable way
    ratio = (p + eps) / (q + eps)
    log_ratio = np.log(ratio + eps)
    
    return np.sum(p * log_ratio)

def batch_kl_divergence(p_batch: np.ndarray, 
                       q_batch: np.ndarray,
                       eps: float = 1e-9) -> np.ndarray:
    """Compute KL divergence for batches of probability distributions.
    
    Args:
        p_batch: Batch of first probability distributions [batch_size, dim]
        q_batch: Batch of second probability distributions [batch_size, dim]
        eps: Small constant for numerical stability
        
    Returns:
        Array of KL divergence values [batch_size]
    """
    if p_batch.ndim == 1:
        return kl_divergence(p_batch, q_batch, eps)
        
    return np.array([
        kl_divergence(p, q, eps)
        for p, q in zip(p_batch, q_batch)
    ])

def surprise_score(current_state: np.ndarray, 
                         reference_states: Optional[List[np.ndarray]] = None,
                         temperature: float = 1.0,
                         use_kl: bool = True) -> float:
    """Calculate surprise score for current hidden state using KL divergence.
    
    Uses a combination of:
    1. KL divergence from reference states (if available)
    2. Fallback to L2 norm for basic activity measure
    
    Args:
        current_state: Current hidden state vector
        reference_states: Optional list of previous hidden states for comparison
        temperature: Temperature for softmax normalization
        use_kl: Whether to use KL divergence (if False, uses L2/cosine)
        
    Returns:
        float: Surprise score in [0, 1] (higher = more surprising)
    """
    if not use_kl:
        # Fallback to original L2/cosine method
        activity_score = np.linalg.norm(current_state)
        
        if not reference_states or len(reference_states) == 0:
            return min(activity_score, 1.0)
            
        # Calculate cosine similarities
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
            
        novelty_score = 1.0 - max(0.0, max_similarity)
        return 0.5 * (min(activity_score, 1.0) + novelty_score)
    
    # Convert current state to probability distribution
    current_dist = to_probability_distribution(current_state, temperature=temperature)
    
    if not reference_states or len(reference_states) == 0:
        # No reference states - use normalized L2 as fallback
        return min(np.linalg.norm(current_state), 1.0)
    
    # Convert reference states to distributions
    ref_distributions = batch_to_probability_distributions(
        np.stack(reference_states),
        temperature=temperature
    )
    
    # Compute KL divergence with each reference state
    kl_values = batch_kl_divergence(
        np.tile(current_dist, (len(ref_distributions), 1)),
        ref_distributions
    )
    
    # Convert KL values to surprise score with improved numerical stability
    # Use min KL value for similar states (lower KL = more similar)
    min_kl = np.min(kl_values)
    
    # Special case for zero vector
    if np.all(current_state == 0):
        return 0.0
        
    # Adaptive temperature scaling based on KL magnitude
    # This prevents saturation for very large or very small KL values
    scale_factor = 1.0 / (1.0 + np.log1p(min_kl))  # Adaptive scaling
    
    # Scale KL value to [0, 1] range using temperature-aware sigmoid
    # Higher temperature makes distribution more uniform, leading to lower surprise
    scaled_kl = min_kl * scale_factor / temperature if temperature > 0 else min_kl * scale_factor
    
    # Use numerically stable sigmoid implementation
    if scaled_kl >= 0:
        surprise = 1.0 / (1.0 + np.exp(-5.0 * (scaled_kl - 0.5)))
    else:
        exp_x = np.exp(5.0 * (scaled_kl - 0.5))
        surprise = exp_x / (1.0 + exp_x)
    
    # Ensure output is in valid range with smooth clamping
    eps = 1e-6  # Small constant to avoid exact 0 or 1
    return float(np.clip(surprise, eps, 1.0 - eps))

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
