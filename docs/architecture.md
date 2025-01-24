# Architecture Overview

## Components

### Memory Repository
- Implements external memory storage using Annoy for efficient nearest neighbor search
- Manages memory entries with timestamps for forgetting mechanism
- Provides add/retrieve interface for memory operations

### Gating Module (MAG)
- Implements Memory as Gating approach
- Combines short-term memory (from attention) with long-term memory
- Uses fixed gating coefficient (no training required)

### Persistent Memory
- Stores pre-computed embeddings
- Provides constant access to important knowledge
- Integrates with gating mechanism

### Integration Layer
- Wraps DeepSeek R1 model
- Intercepts forward pass to incorporate memory
- Manages VRAM usage and memory retrieval

## Memory Management

### Surprise Metric
- Measures novelty of current hidden states
- Determines when to store new memories
- Based on simple vector operations (no training)

### Forgetting Mechanism
- Time-based pruning of old memories
- Maintains bounded memory growth
- Configurable retention parameters

## VRAM Considerations
- DeepSeek R1 (4-bit): ~20GB
- Memory operations primarily on CPU
- Limited GPU memory for active embeddings
- Total budget: 64GB VRAM

## Implementation Notes
- No context window modifications
- No fine-tuning or training
- Simple, rule-based memory management
- Modular design for future improvements
