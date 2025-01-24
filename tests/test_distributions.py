"""Tests for probability distribution conversion and KL divergence calculations."""

import numpy as np
import pytest
from memory_repository.memory_management import (
    to_probability_distribution,
    batch_to_probability_distributions,
    kl_divergence,
    batch_kl_divergence,
    surprise_score
)

def test_to_probability_distribution_basic():
    """Test basic probability distribution conversion."""
    # Test with simple vector
    x = np.array([1.0, 2.0, 3.0])
    probs = to_probability_distribution(x)
    
    # Check properties
    assert np.allclose(np.sum(probs), 1.0)  # Sums to 1
    assert np.all(probs >= 0)  # Non-negative
    assert probs[2] > probs[1] > probs[0]  # Maintains ordering

def test_to_probability_distribution_edge_cases():
    """Test edge cases for probability conversion."""
    # Zero vector
    x = np.zeros(5)
    probs = to_probability_distribution(x)
    assert np.allclose(probs, 0.2)  # Uniform distribution
    
    # Very large values
    x = np.array([1e5, 1e5 + 1, 1e5 + 2])
    probs = to_probability_distribution(x)
    assert np.allclose(np.sum(probs), 1.0)
    
    # Very small values
    x = np.array([1e-8, 2e-8, 3e-8])
    probs = to_probability_distribution(x)
    assert np.allclose(np.sum(probs), 1.0)

def test_batch_conversion():
    """Test batch conversion to probability distributions."""
    # Create batch of vectors
    batch = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [1e5, 1e5 + 1, 1e5 + 2]
    ])
    
    probs = batch_to_probability_distributions(batch)
    
    # Check batch properties
    assert probs.shape == batch.shape
    assert np.allclose(np.sum(probs, axis=1), 1.0)  # Each distribution sums to 1
    assert np.all(probs >= 0)  # All non-negative

def test_temperature_scaling():
    """Test temperature effects on distribution."""
    x = np.array([1.0, 2.0, 3.0])
    
    # Higher temperature = more uniform
    high_temp = to_probability_distribution(x, temperature=10.0)
    low_temp = to_probability_distribution(x, temperature=0.1)
    
    # High temperature should be more uniform
    high_temp_diff = np.max(high_temp) - np.min(high_temp)
    low_temp_diff = np.max(low_temp) - np.min(low_temp)
    assert high_temp_diff < low_temp_diff

def test_kl_divergence_basic():
    """Test basic KL divergence calculation."""
    # Test with simple distributions
    p = np.array([0.3, 0.7])
    q = np.array([0.4, 0.6])
    
    kl = kl_divergence(p, q)
    assert kl >= 0  # KL divergence is always non-negative
    assert np.isfinite(kl)  # Should be finite

def test_kl_divergence_edge_cases():
    """Test edge cases for KL divergence."""
    # Same distribution
    p = np.array([0.5, 0.5])
    kl = kl_divergence(p, p)
    assert np.isclose(kl, 0.0)  # KL divergence is 0 for identical distributions
    
    # Zero probabilities
    p = np.array([0.0, 1.0])
    q = np.array([0.5, 0.5])
    kl = kl_divergence(p, q)
    assert np.isfinite(kl)  # Should handle zeros gracefully
    
    # Very small probabilities
    p = np.array([1e-10, 1-1e-10])
    q = np.array([1e-11, 1-1e-11])
    kl = kl_divergence(p, q)
    assert np.isfinite(kl)

def test_batch_kl_divergence():
    """Test batch KL divergence calculation."""
    # Create batch of distributions
    p_batch = np.array([
        [0.3, 0.7],
        [0.5, 0.5],
        [0.1, 0.9]
    ])
    q_batch = np.array([
        [0.4, 0.6],
        [0.5, 0.5],
        [0.2, 0.8]
    ])
    
    kl_values = batch_kl_divergence(p_batch, q_batch)
    
    assert kl_values.shape == (3,)  # One value per batch
    assert np.all(kl_values >= 0)  # All should be non-negative
    assert np.all(np.isfinite(kl_values))  # All should be finite
    assert np.isclose(kl_values[1], 0.0)  # Second pair is identical

def test_surprise_score_kl():
    """Test surprise score calculation using KL divergence."""
    # Test with similar states
    current = np.array([1.0, 2.0, 3.0])
    similar = np.array([1.1, 2.1, 3.1])
    score = surprise_score(current, [similar], use_kl=True)
    assert 0 <= score <= 1
    assert score < 0.5  # Should be low surprise for similar states
    
    # Test with different states
    different = np.array([3.0, 2.0, 1.0])
    score = surprise_score(current, [different], use_kl=True)
    assert 0 <= score <= 1
    assert score > 0.5  # Should be high surprise for different states
    
    # Test with multiple reference states
    references = [similar, different]
    score = surprise_score(current, references, use_kl=True)
    assert 0 <= score <= 1
    
    # Test temperature effects
    score_high_temp = surprise_score(current, references, temperature=10.0, use_kl=True)
    score_low_temp = surprise_score(current, references, temperature=0.1, use_kl=True)
    assert score_high_temp < score_low_temp  # Higher temperature = more uniform = less surprise

def test_surprise_score_fallback():
    """Test surprise score fallback behavior."""
    # Test with no reference states
    state = np.array([1.0, 2.0, 3.0])
    score = surprise_score(state, None, use_kl=True)
    assert 0 <= score <= 1
    
    # Test with zero vector
    zero_state = np.zeros_like(state)
    score = surprise_score(zero_state, [state], use_kl=True)
    assert score == 0.0
    
    # Test original method
    score_original = surprise_score(state, [state], use_kl=False)
    assert 0 <= score_original <= 1

if __name__ == "__main__":
    pytest.main([__file__])
