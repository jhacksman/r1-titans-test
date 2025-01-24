# R1 Titans Memory Integration

This repository implements Titans-inspired memory modules for the DeepSeek R1 model (32B parameter version), enabling long-term and persistent memory capabilities while maintaining the model's existing reasoning abilities.

## Project Goals

- Enable DeepSeek R1 to reference and accumulate historical knowledge
- Avoid expanding token-based context window
- Avoid any new fine-tuning or RL steps
- Ensure memory modules fit within 64GB VRAM budget

## Architecture Overview

The implementation follows the Titans memory architecture, incorporating:
- Short-Term Memory (STM): Leveraging existing attention mechanisms
- Long-Term Memory (LTM): Neural memory module for cross-session context
- Persistent Memory: Input-independent parameters for task/context knowledge

Memory management includes:
- Surprise metric for detecting significant new information
- Forgetting mechanism to prevent unbounded memory growth

## Implementation Details

### Memory Components
- Memory as Gating (MAG) implementation
- External memory repository with Annoy-based retrieval
- Surprise-based memory management
- VRAM-optimized architecture

### Technical Specifications
- No context window modifications
- No fine-tuning or training required
- 4-bit quantized R1 (~20GB) compatibility
- Comprehensive test suite

## Directory Structure

```
.
├── memory_repository/     # External memory management
├── gating_module/        # MAG implementation
├── persistent_memory/    # Fixed knowledge embeddings
├── docs/                # Documentation
├── tests/              # Test suite
└── integrations/        # DeepSeek R1 integration code
```

## Hardware Requirements

- GPU with up to 64GB VRAM
- Base DeepSeek R1 model (~20GB in 4-bit quantization)
- Additional VRAM headroom for memory modules

## Development Status

🚧 Under active development

## License

MIT License
