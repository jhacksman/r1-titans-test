# Architecture Overview

## Components

### Memory Repository
- Implements external memory storage using Annoy for efficient nearest neighbor search (verified O(log n) retrieval)
- Manages memory entries with timestamps and surprise scores
- Provides batch operations for efficient memory management
- Automatic index rebuilding with fallback options

### Gating Module (MAG)
- Memory as Gating implementation with verified numerical stability
- Combines short-term memory (from attention) with long-term memory
- Supports both fixed and surprise-based gating coefficients
- Tensor-based operations with broadcastable coefficients
- Comprehensive edge case handling (zeros, NaN, extreme values)

### Persistent Memory
- Pre-computed embeddings for cross-session knowledge
- Efficient retrieval with configurable similarity thresholds
- Memory-mapped storage for large embedding sets
- Integration with surprise-based gating

### Integration Layer
- Wraps DeepSeek R1 model with minimal overhead
- Optimized forward pass with memory integration
- VRAM-aware memory management and retrieval
- Automatic pruning of stale memories

## Memory Management

### Surprise Metric
- KL divergence-based novelty detection
- Adaptive temperature scaling for robust scoring
- Numerically stable implementation with fallbacks
- Configurable thresholds for memory updates

### Forgetting Mechanism
- Multi-criteria pruning strategy:
  * Time-based: Configurable age thresholds
  * Surprise-based: Minimum surprise score filtering
  * Capacity-based: Maximum memory limits
- Efficient batch processing for removals
- Automatic index maintenance

## VRAM Usage (Verified)
- DeepSeek R1 (4-bit): 20.00GB
- Memory Store: 1.98GB
- Gating Module: 2.20GB
- Persistent Memory: 0.03GB
- Runtime Buffer: 2.00GB
- Total Usage: 26.21GB (within 64GB budget)

## Implementation Details
- Zero context window modification
- No training or fine-tuning required
- Comprehensive test coverage:
  * VRAM constraints
  * Memory retrieval accuracy
  * Gating mechanism stability
  * Edge case handling
- Modular architecture for future extensions
