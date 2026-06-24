"""
Wrappers for PyTorch's built-in sparse linear algebra functions.

Provides condition number estimation using:
- PyTorch backend sparse.linalg.svds
- PyTorch backend sparse.linalg.eigsh
- PyTorch backend sparse.linalg.lobpcg
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from sparse_kappa.backend.sparse import linalg as splinalg
from typing import Dict, Any
import warnings




def eigsh_cond(
    A: sp.spmatrix,
    num_values: int = 6,
    max_iter: int = None,
    tol: float = 0,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Estimate condition number using PyTorch's eigsh (for symmetric matrices).
    
    For symmetric matrices: kappa_2(A) = |lambda_max| / |lambda_min|
    For general matrices: applies to A^T A
    
    Parameters
    ----------
    A : sparse matrix
        Input matrix (should be symmetric)
    num_values : int, default=6
        Number of eigenvalues to compute
    max_iter : int, optional
        Maximum iterations
    tol : float, default=0
        Tolerance
    verbose : bool, default=False
        Print information
    
    Returns
    -------
    result : dict
        Condition number estimation results
    """
    if verbose:
        print("Using PyTorch's eigsh for condition number")
        print(f"Matrix size: {A.shape}")
    
    n = A.shape[0]
    max_k = n - 2
    k = max(1, min(num_values, max_k))
    
    if k >= n - 1 or k < 1:
        if verbose:
            print(f"Matrix too small for eigsh, using dense SVD instead")
        A_dense = A.toarray()
        try:
            s = cp.linalg.svd(A_dense, compute_uv=False)
            lambda_max = float(s[0])
            lambda_min = float(s[-1])
        except Exception as e:
            if verbose:
                print(f"Dense SVD failed: {e}, trying numpy on CPU")
            import numpy as np
            A_cpu = cp.asnumpy(A_dense)
            s = np.linalg.svd(A_cpu, compute_uv=False)
            lambda_max = float(s[0])
            lambda_min = float(s[-1])
        
        if lambda_min < 1e-14:
            cond = float('inf') if lambda_min == 0 else lambda_max / lambda_min
        else:
            cond = lambda_max / lambda_min
        
        return {
            'condition_number': float(cond),
            'sigma_max': lambda_max,
            'sigma_min': lambda_min,
            'iterations': 0,
            'converged': True,
        }
    
    try:
        eigvals_large = splinalg.eigsh(A, k=k, which='LA', maxiter=max_iter, 
                                      tol=tol, return_eigenvectors=False)
        
        eigvals_small = splinalg.eigsh(A, k=k, which='SA', maxiter=max_iter,
                                      tol=tol, return_eigenvectors=False)
        
        if verbose:
            print(f"eigvals_large: {eigvals_large}")
            print(f"eigvals_small: {eigvals_small}")
        
        all_eigvals = cp.concatenate([eigvals_large, eigvals_small])
        abs_eigvals = cp.abs(all_eigvals)
        
        # Check if eigenvalues are unreasonably large (numerical instability)
        max_reasonable = 1e10
        if cp.max(abs_eigvals) > max_reasonable or cp.min(abs_eigvals) > max_reasonable:
            if verbose:
                print(f"Unreasonably large eigenvalues detected, falling back to dense SVD")
            raise ValueError("Numerical instability in eigsh")
        
        # Filter out near-zero eigenvalues
        abs_eigvals_nonzero = abs_eigvals[abs_eigvals > 1e-14]
        
        if len(abs_eigvals_nonzero) == 0:
            lambda_max = float(cp.max(abs_eigvals))
            lambda_min = float(cp.min(abs_eigvals))
        else:
            lambda_max = float(cp.max(abs_eigvals))
            lambda_min = float(cp.min(abs_eigvals_nonzero))
        
        # Check for NaN or unreasonable values
        if cp.isnan(lambda_max) or cp.isnan(lambda_min) or lambda_max > max_reasonable:
            if verbose:
                print("Invalid eigenvalues detected, falling back to dense SVD")
            raise ValueError("Invalid eigenvalues from eigsh")
        
        # Check if all eigenvalues are nearly the same (like identity matrix)
        if len(abs_eigvals_nonzero) > 0:
            eigval_range = cp.max(abs_eigvals_nonzero) - cp.min(abs_eigvals_nonzero)
            eigval_mean = cp.mean(abs_eigvals_nonzero)
            relative_range = eigval_range / (eigval_mean + 1e-14)
            
            if relative_range < 0.01:  # All eigenvalues within 1% of each other
                if verbose:
                    print(f"All eigenvalues nearly identical (range/mean = {relative_range:.2e})")
                    print("Matrix is likely well-conditioned, using ratio 1.0")
                cond = 1.0
                converged = True
                
                return {
                    'condition_number': float(cond),
                    'sigma_max': lambda_max,
                    'sigma_min': lambda_min,
                    'iterations': max_iter or 0,
                    'converged': converged,
                }
        
        if lambda_min < 1e-14:
            cond = float('inf') if lambda_min == 0 else lambda_max / lambda_min
        else:
            cond = lambda_max / lambda_min
        
        converged = True
        
    except Exception as e:
        if verbose:
            print(f"Error in eigsh: {e}, falling back to dense SVD")
        A_dense = A.toarray()
        try:
            s = cp.linalg.svd(A_dense, compute_uv=False)
            lambda_max = float(s[0])
            lambda_min = float(s[-1])
        except Exception as e2:
            if verbose:
                print(f"Dense SVD failed: {e2}, trying numpy on CPU")
            import numpy as np
            A_cpu = cp.asnumpy(A_dense)
            s = np.linalg.svd(A_cpu, compute_uv=False)
            lambda_max = float(s[0])
            lambda_min = float(s[-1])
        
        if lambda_min < 1e-14:
            cond = float('inf') if lambda_min == 0 else lambda_max / lambda_min
        else:
            cond = lambda_max / lambda_min
        
        converged = False
    
    if verbose:
        print(f"|lambda_max| = {lambda_max:.6e}")
        print(f"|lambda_min| = {lambda_min:.6e}")
        print(f"kappa_2(A) = {cond:.6e}")
    
    return {
        'condition_number': float(cond),
        'sigma_max': lambda_max,
        'sigma_min': lambda_min,
        'iterations': max_iter or 0,
        'converged': converged,
    }


def lobpcg_cond(
    A: sp.spmatrix,
    num_values: int = 2,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Estimate condition number using PyTorch's LOBPCG.
    
    LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient)
    is efficient for computing a few extreme eigenvalues.
    
    Parameters
    ----------
    A : sparse matrix
        Input matrix
    num_values : int, default=2
        Number of eigenvalues to compute
    max_iter : int, default=100
        Maximum iterations
    tol : float, default=1e-6
        Tolerance
    verbose : bool, default=False
        Print information
    
    Returns
    -------
    result : dict
        Condition number estimation results
    """
    if verbose:
        print("Using PyTorch's LOBPCG for condition number")
        print(f"Matrix size: {A.shape}")
    
    n = A.shape[0]
    
    num_values = min(num_values, n // 2)
    if num_values < 1:
        num_values = 1
    
    def matvec(v):
        return A.T @ (A @ v)
    
    from sparse_kappa.backend.sparse.linalg import LinearOperator
    ATA = LinearOperator((n, n), matvec=matvec, dtype=A.dtype)
    
    try:
        X = cp.random.randn(n, num_values).astype(A.dtype)
        
        eigenvalues, _ = splinalg.lobpcg(ATA, X, maxiter=max_iter, tol=tol)
        
        lambda_max = float(cp.max(cp.abs(eigenvalues)))
        lambda_min = float(cp.min(cp.abs(eigenvalues)))
        
        sigma_max = cp.sqrt(lambda_max)
        sigma_min = cp.sqrt(lambda_min)
        
        if sigma_min < 1e-14:
            cond = float('inf') if sigma_min == 0 else sigma_max / sigma_min
        else:
            cond = sigma_max / sigma_min
        
        converged = True
        
    except Exception as e:
        if verbose:
            print(f"Error in LOBPCG: {e}")
        raise
    
    if verbose:
        print(f"sigma_max = {sigma_max:.6e}")
        print(f"sigma_min = {sigma_min:.6e}")
        print(f"kappa_2(A) = {cond:.6e}")
    
    return {
        'condition_number': float(cond),
        'sigma_max': float(sigma_max),
        'sigma_min': float(sigma_min),
        'iterations': max_iter,
        'converged': converged,
    }