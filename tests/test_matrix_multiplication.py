import numpy as np
import cupy as cp
import pytest
from benchmarks.matrix_multiplication import matrix_multiplication

@pytest.mark.parametrize("size", [10, 50, 100])  # Test small sizes
def test_matrix_multiplication(size):
    """
    Test that CPU and GPU matrix multiplication produce similar results.
    """
    cpu_time, gpu_time, speedup = matrix_multiplication(size)

    # Generate small matrices for comparison
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)

    # Compute expected result on CPU
    expected = np.dot(A, B)

    # Compute result on GPU
    A_gpu = cp.array(A)
    B_gpu = cp.array(B)
    result_gpu = cp.dot(A_gpu, B_gpu).get()  # Move GPU result back to CPU

    # Assert matrices are close within tolerance
    np.testing.assert_allclose(expected, result_gpu, rtol=1e-5, atol=1e-5)
