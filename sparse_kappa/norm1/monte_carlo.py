"""
Monte Carlo sampling method for 1-norm estimation.

Random sampling approach for quick estimates.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from typing import Dict, Any
from ..utils import apply_inverse, sparse_matrix_norm


def monte_carlo_norm1(
    A: sp.spmatrix,
    num_samples: int = 20,
    use_basis: bool = True,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Estimate 1-norm condition number using Monte Carlo sampling.
    
    Uses random sampling to estimate ||A^{-1}||_1.
    
    Parameters
    ----------
    A : sparse matrix
        Input matrix (n x n)
    num_samples : int, default=20
        Number of random samples
    use_basis : bool, default=True
        Include standard basis vectors in samples
    verbose : bool, default=False
        Print information
    
    Returns
    -------
    result : dict
        - 'condition_number': estimated kappa_1(A)
        - 'norm_A': ||A||_1
        - 'norm_Ainv': ||A^{-1}||_1
        - 'iterations': number of samples
        - 'converged': always True
    
    Algorithm
    ---------
    Sample random vectors and compute ||A^{-1} b||_1 / ||b||_1.
    Take maximum over all samples.
    """
    n = A.shape[0]
    
    # Compute ||A||_1 exactly using utility function
    norm_A = sparse_matrix_norm(A, ord=1)
    
    if verbose:
        print("Monte Carlo sampling for 1-norm condition number")
        print(f"Matrix size: {n} x {n}")
        print(f"Number of samples: {num_samples}")
        print(f"Include basis vectors: {use_basis}")
        print(f"||A||_1 = {norm_A:.6e}")
    
    max_norm = 0.0
    sample_count = 0
    
    # Sample standard basis vectors if requested
    if use_basis:
        num_basis = min(num_samples // 2, n)
        indices = cp.random.choice(n, size=num_basis, replace=False)
        
        for idx in indices:
            e_i = cp.zeros(n, dtype=A.dtype)
            e_i[int(idx)] = 1.0
            
            # Solve A x = e_i
            x = apply_inverse(A, e_i, method='lsmr', atol=1e-10, btol=1e-10)
            
            norm_x = float(cp.linalg.norm(x, ord=1))
            max_norm = max(max_norm, norm_x)
            sample_count += 1
            
            if verbose and sample_count % 5 == 0:
                print(f"Sample {sample_count}: ||A^{{-1}} e_{int(idx)}||_1 = {norm_x:.6e}")
    
    # Sample random ±1 vectors (good for 1-norm)
    num_random = num_samples - sample_count
    for i in range(num_random):
        # Random sign vector
        b = cp.random.choice([-1.0, 1.0], size=n).astype(A.dtype)
        
        # Solve A x = b
        x = apply_inverse(A, b, method='lsmr', atol=1e-10, btol=1e-10)
        
        norm_x = float(cp.linalg.norm(x, ord=1))
        norm_b = float(cp.linalg.norm(b, ord=1))
        
        # Normalize by ||b||_1
        normalized_norm = norm_x / norm_b
        max_norm = max(max_norm, normalized_norm)
        sample_count += 1
    
    norm_Ainv = max_norm
    cond = norm_A * norm_Ainv
    
    if verbose:
        print(f"\n||A^{{-1}}||_1 (estimated) = {norm_Ainv:.6e}")
        print(f"kappa_1(A) = {cond:.6e}")
    
    return {
        'condition_number': float(cond),
        'norm_A': float(norm_A),
        'norm_Ainv': float(norm_Ainv),
        'iterations': sample_count,
        'converged': True,
    }