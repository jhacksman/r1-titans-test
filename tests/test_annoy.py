import numpy as np
from annoy import AnnoyIndex

def test_annoy_basic():
    """Basic test to verify Annoy functionality."""
    # Create index
    dim = 10
    index = AnnoyIndex(dim, 'angular')
    
    # Add test vector
    test_vector = np.random.random(dim)
    index.add_item(0, test_vector)
    
    # Build and verify
    index.build(10)  # 10 trees
    
    # Search and verify
    indices, distances = index.get_nns_by_vector(test_vector, 1, include_distances=True)
    assert indices[0] == 0  # Should find the same vector
    assert distances[0] < 1e-6  # Should be very close to 0
    
    print("Annoy test successful")

if __name__ == "__main__":
    test_annoy_basic()
