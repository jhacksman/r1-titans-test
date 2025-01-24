import torch
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
        
        # Get current hidden states
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # Convert to numpy for memory operations
        hidden_np = hidden_states.detach().cpu().numpy()
        
        # Compute surprise score
        current_surprise = surprise_score(hidden_np)
        
        # Retrieve relevant memories
        batch_size, seq_len, _ = hidden_states.shape
        memory_shape = self.gating.get_memory_shape(batch_size, seq_len)
        
        # Get memory embeddings
        memory_keys, _ = self.memory_store.retrieve(
            hidden_np.reshape(-1, hidden_np.shape[-1]),
            k=1
        )
        
        if memory_keys:
            memory_data = self.memory_store.get_memory(memory_keys[0])
            if memory_data:
                memory_hidden = torch.from_numpy(
                    memory_data['embedding']
                ).reshape(memory_shape).to(hidden_states.device)
            else:
                memory_hidden = torch.zeros_like(hidden_states)
        else:
            memory_hidden = torch.zeros_like(hidden_states)
        
        # Apply gating
        gated_hidden = self.gating.forward(
            hidden_states,
            memory_hidden,
            surprise_score=current_surprise
        )
        
        # Store current hidden state if surprising enough
        if current_surprise > 0.5:  # Configurable threshold
            self.memory_store.add_memory(
                hidden_np.reshape(-1, hidden_np.shape[-1]),
                metadata={'surprise_score': current_surprise}
            )
        
        # Update output hidden states
        outputs.hidden_states = outputs.hidden_states[:-1] + (gated_hidden,)
        
        return outputs
