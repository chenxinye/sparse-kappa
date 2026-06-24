"""
Helper functions for inverse norm estimation.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from ..utils import apply_inverse


def power_iteration_inverse_norm1(
    A: sp.spmatrix,
    max_iter: int = 20,
    tol: float = 1e-6
) -> float:
    """
    Estimate ||A⁻¹||₁ using power iteration.
    
    Parameters
    ----------
    A : sparse matrix
        Input matrix
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    
    Returns
    -------
    norm_estimate : float
        Estimated ||A⁻¹||₁
    """
    n = A.shape[0]
    
    # Start with random vector
    x = cp.random.randn(n).astype(A.dtype)
    x /= cp.linalg.norm(x, ord=1)
    
    for _ in range(max_iter):
        # y = A⁻¹ x
        y = apply_inverse(A, x, method='lsmr')
        
        # Normalize
        norm_y = cp.linalg.norm(y, ord=1)
        y_new = y / norm_y
        
        # Check convergence
        if cp.linalg.norm(y_new - x, ord=1) < tol:
            break
        
        x = y_new
    
    # Final estimate
    y = apply_inverse(A, x, method='lsmr')
    return float(cp.linalg.norm(y, ord=1))