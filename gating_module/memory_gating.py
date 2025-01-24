import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class MemoryGatingModule:
    """Memory as Gating (MAG) implementation for DeepSeek R1 32B model.
    
    This module implements a lightweight gating mechanism that combines
    short-term memory (from attention) with retrieved long-term memory.
    No training is involved - the gating coefficient is either fixed or
    derived from a surprise-based heuristic.
    
    Design Philosophy:
    - Maintain DeepSeek R1's 4-bit quantization compatibility
    - No context window modifications
    - VRAM-efficient implementation (part of 64GB budget)
    - No training or fine-tuning required
    
    The gating mechanism uses a simple weighted combination:
    output = α * current_hidden + (1-α) * memory_hidden
    where α is determined by either a fixed value or surprise score.
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 fixed_alpha: Optional[float] = None,
                 use_surprise_gating: bool = True):
        """Initialize the gating module.
        
        Args:
            hidden_dim: Dimension of hidden states
            fixed_alpha: If provided, use this fixed value for gating
            use_surprise_gating: If True, derive alpha from surprise score
        """
        self.hidden_dim = hidden_dim
        self.fixed_alpha = fixed_alpha
        self.use_surprise_gating = use_surprise_gating
        
        # No learnable parameters - we only need dimension info
        # and gating logic for inference
        
    def compute_gating_coefficient(self, 
                                 current_hidden: torch.Tensor,
                                 memory_hidden: torch.Tensor,
                                 surprise_score: Optional[float] = None) -> float:
        """Compute the gating coefficient (alpha) for combining hidden states.
        
        Args:
            current_hidden: Current hidden state from the model
            memory_hidden: Retrieved memory hidden state
            surprise_score: Optional pre-computed surprise score
            
        Returns:
            float: Gating coefficient between 0 and 1
        """
        if self.fixed_alpha is not None:
            return self.fixed_alpha
            
        if not self.use_surprise_gating or surprise_score is None:
            # Default to equal weighting if no surprise score
            return 0.5
            
        # Convert surprise score to gating coefficient
        # Higher surprise -> more weight on current hidden state
        alpha = min(max(surprise_score, 0.0), 1.0)
        return alpha
        
    def forward(self,
                current_hidden: torch.Tensor,
                memory_hidden: torch.Tensor,
                surprise_score: Optional[float] = None) -> torch.Tensor:
        """Combine current and memory hidden states using gating.
        
        Args:
            current_hidden: Hidden state from current context [batch, seq_len, hidden_dim]
            memory_hidden: Retrieved memory hidden state [batch, seq_len, hidden_dim]
            surprise_score: Optional score from surprise metric
            
        Returns:
            Combined hidden state with same shape as inputs
        """
        # Ensure shapes match
        assert current_hidden.shape == memory_hidden.shape, \
            f"Shape mismatch: {current_hidden.shape} vs {memory_hidden.shape}"
        
        # Compute gating coefficient
        alpha = self.compute_gating_coefficient(
            current_hidden, memory_hidden, surprise_score)
            
        # Combine hidden states using weighted sum
        combined = alpha * current_hidden + (1.0 - alpha) * memory_hidden
        
        return combined
        
    def get_memory_shape(self, batch_size: int, seq_len: int) -> Tuple[int, ...]:
        """Get expected shape of memory tensor for given batch/sequence."""
        return (batch_size, seq_len, self.hidden_dim)
