import torch
import time
from typing import Optional, Dict, Any
from .memory_gating import MemoryGatingModule
from ..memory_repository.memory_store import MemoryStore
from ..memory_repository.memory_management import surprise_score

class DeepSeekMemoryWrapper:
    """Wrapper for DeepSeek R1 model that integrates memory gating.
    
    This wrapper:
    1. Intercepts the forward pass to incorporate memory
    2. Manages memory storage and retrieval
    3. Handles surprise-based memory updates
    """
    
    def __init__(self,
                 model: Any,  # DeepSeek R1 model
                 memory_store: MemoryStore,
                 hidden_dim: int,
                 fixed_alpha: Optional[float] = None):
        """Initialize the wrapper.
        
        Args:
            model: DeepSeek R1 model instance
            memory_store: MemoryStore instance for long-term memory
            hidden_dim: Hidden state dimension
            fixed_alpha: Optional fixed gating coefficient
        """
        self.model = model
        self.memory_store = memory_store
        self.gating = MemoryGatingModule(
            hidden_dim=hidden_dim,
            fixed_alpha=fixed_alpha,
            use_surprise_gating=fixed_alpha is None
        )
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass with memory integration.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
            **kwargs: Additional arguments passed to model
            
        Returns:
            Dict containing model outputs with memory-enhanced hidden states
        """
        # Get base model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get current hidden states (last layer)
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        device = hidden_states.device
        
        # Convert to numpy for memory operations (minimize VRAM usage)
        hidden_np = hidden_states.detach().cpu().numpy()
        
        # Compute surprise score using flattened representation
        flat_hidden = hidden_np.reshape(-1, hidden_np.shape[-1])
        current_surprise = surprise_score(flat_hidden)
        
        # Retrieve relevant memories
        batch_size, seq_len, hidden_dim = hidden_states.shape
        memory_shape = (batch_size, seq_len, hidden_dim)
        
        # Get memory embeddings (k=1 for efficiency)
        memory_keys, distances = self.memory_store.retrieve(flat_hidden, k=1)
        
        # Initialize memory tensor on CPU first
        if memory_keys and len(memory_keys) > 0:
            memory_data = self.memory_store.get_memory(memory_keys[0])
            if memory_data:
                memory_hidden = torch.from_numpy(
                    memory_data['embedding']
                ).reshape(memory_shape)
            else:
                memory_hidden = torch.zeros(memory_shape)
        else:
            memory_hidden = torch.zeros(memory_shape)
            
        # Move to GPU only when ready to gate
        memory_hidden = memory_hidden.to(device)
        
        # Apply gating mechanism
        gated_hidden = self.gating.forward(
            hidden_states,
            memory_hidden,
            surprise_score=current_surprise
        )
        
        # Update memory if sufficiently surprising
        if current_surprise > 0.5:  # Configurable threshold
            self.memory_store.add_memory(
                flat_hidden,
                metadata={
                    'surprise_score': float(current_surprise),
                    'timestamp': time.time()
                }
            )
        
        # Update output hidden states
        outputs.hidden_states = outputs.hidden_states[:-1] + (gated_hidden,)
        
        return outputs
