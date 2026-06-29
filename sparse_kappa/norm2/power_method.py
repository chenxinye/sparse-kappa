"""
Power method for 2-norm condition number estimation (Aligned with PyTorch).

Estimates kappa_2(A) = σ_max / σ_min using standard power iteration.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from typing import Dict, Any, Optional
import numpy as np


def power_method_cond(
    A: sp.spmatrix,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Estimate 2-norm condition number using power method.
    
    ⚡ Aligned with PyTorch standard implementation:
    - Standard Rayleigh quotient formulation
    - Convergence detection on vector differences
    - Adaptive memory management for inverse iteration
    - Proper numerical stability (shift-invert)
    
    kappa_2(A) = σ_max / σ_min where σ are singular values.
    
    Parameters
    ----------
    A : PyTorch backend sparse.spmatrix
        Input sparse matrix (n x n)
    max_iter : int, default=100
        Maximum iterations
    tol : float, default=1e-6
        Convergence tolerance
    verbose : bool, default=False
        Print iteration information
    
    Returns
    -------
    result : dict
        - 'condition_number': estimated kappa_2(A)
        - 'sigma_max': largest singular value
        - 'sigma_min': smallest singular value
        - 'iterations': number of iterations
        - 'converged': convergence status
    
    Algorithm
    ---------
    1. Compute σ_max using power iteration on A^T A (sparse matvec)
    2. Compute σ_min using inverse power iteration on A^T A (adaptive solve)
    3. kappa_2(A) = σ_max / σ_min
    
    Complexity: O(k · nnz(A)) per iteration
    """
    if verbose:
        print("Power Method for 2-norm condition number")
        print(f"Matrix size: {A.shape[0]} x {A.shape[1]}")
    
    breakdown_tol = 1e-14  # Aligned with PyTorch
    
    # Compute largest singular value
    sigma_max, iter_max, conv_max = _compute_sigma_max(
        A, max_iter, tol, breakdown_tol, verbose
    )
    
    # Compute smallest singular value
    sigma_min, iter_min, conv_min = _compute_sigma_min(
        A, max_iter, tol, breakdown_tol, verbose
    )
    
    # Condition number
    if sigma_min < breakdown_tol:
        cond = 1e30
    else:
        cond = sigma_max / sigma_min
    
    # Condition numbers are mathematically >= 1; tiny sub-1 values here are
    # numerical artifacts and should be clamped, while non-finite values are failures.
    if not np.isfinite(cond):
        cond = 1e30
    elif cond < 1:
        cond = 1.0
    
    if verbose:
        print(f"\nσ_max = {sigma_max:.6e}")
        print(f"σ_min = {sigma_min:.6e}")
        print(f"kappa_2(A) = {cond:.6e}")
    
    return {
        'condition_number': float(cond),
        'sigma_max': float(sigma_max),
        'sigma_min': float(sigma_min),
        'iterations': iter_max + iter_min,
        'converged': conv_max and conv_min,
    }


def _compute_sigma_max(
    A: sp.spmatrix,
    max_iter: int,
    tol: float,
    breakdown_tol: float,
    verbose: bool
) -> tuple:
    """
    Compute largest singular value using standard power iteration.
    
    Algorithm (aligned with PyTorch):
        1. v_{k+1} = (A^T A) v_k
        2. v_{k+1} = v_{k+1} / ||v_{k+1}||
        3. σ_max^2 = v_k^T (A^T A) v_k  (Rayleigh quotient)
        4. Check convergence: ||v_{k+1} - v_k|| < tol
    
    Returns
    -------
    sigma_max : float
        Largest singular value
    iterations : int
        Number of iterations
    converged : bool
        Convergence status
    """
    n = A.shape[1]
    
    if verbose:
        print("\n[1/2] Computing σ_max (power iteration)...")
    
    # Random initial vector (seed for reproducibility)
    cp.random.seed(42)
    v = cp.random.randn(n).astype(A.dtype)
    v = v / cp.linalg.norm(v)
    
    sigma_max_sq = 0.0
    converged = False
    
    for k in range(max_iter):
        # Apply A^T A using sparse matvec
        Av = A @ v
        v_new = A.T @ Av
        
        # Check for breakdown
        norm_v_new = cp.linalg.norm(v_new)
        if norm_v_new < breakdown_tol:
            if verbose:
                print(f"  Breakdown at iteration {k+1}: ||v_new|| = {float(norm_v_new):.2e}")
            sigma_max = 0.0
            converged = True
            return float(sigma_max), k + 1, converged
        
        # Normalize
        v_new = v_new / norm_v_new
        
        # Rayleigh quotient (standard formulation)
        Av_new = A @ v_new
        ATAv_new = A.T @ Av_new
        sigma_max_sq = float(cp.dot(v_new, ATAv_new))
        sigma_max = cp.sqrt(cp.abs(sigma_max_sq))
        
        if verbose and (k + 1) % 10 == 0:
            print(f"  iter={k+1:3d} σ_max~={float(sigma_max):.6e}")
        
        # Convergence check: ||v_{k+1} - v_k|| < tol (aligned with PyTorch)
        vec_diff = cp.linalg.norm(v_new - v)
        if float(vec_diff) < tol:
            converged = True
            if verbose:
                print(f"  Converged at iteration {k+1}")
            break
        
        v = v_new
    
    if verbose:
        print(f"  Final σ_max = {float(sigma_max):.6e}, converged={converged}")
    
    return float(sigma_max), min(k + 1, max_iter), converged


def _compute_sigma_min(
    A: sp.spmatrix,
    max_iter: int,
    tol: float,
    breakdown_tol: float,
    verbose: bool
) -> tuple:
    """
    Compute smallest singular value using inverse power iteration with adaptive strategy.
    
    Algorithm (aligned with PyTorch):
        1. Solve (A^T A + shift·I) w = v
        2. v_{k+1} = w / ||w||
        3. σ_min^2 = v_k^T (A^T A) v_k  (Rayleigh quotient on original matrix)
        4. Check convergence: ||v_{k+1} - v_k|| < tol
    
    Strategy:
        - Small matrices (< 1GB memory): dense solve (fast)
        - Large matrices: sparse CG solve (memory-efficient)
    
    Returns
    -------
    sigma_min : float
        Smallest singular value
    iterations : int
        Number of iterations
    converged : bool
        Convergence status
    """
    from sparse_kappa.backend.sparse.linalg import cg
    
    n = A.shape[1]
    
    if verbose:
        print("\n[2/2] Computing σ_min (inverse iteration)...")
    
    # Memory management: decide strategy
    memory_required_gb = (n * n * 8) / (1024**3)
    use_dense_solve = memory_required_gb < 1.0
    
    if verbose:
        print(f"  Matrix size: {n} x {n}")
        print(f"  Memory required for dense A^T A: {memory_required_gb:.2f} GB")
        print(f"  Using {'dense' if use_dense_solve else 'sparse CG'} solve")
    
    # Random initial vector
    cp.random.seed(123)
    v = cp.random.randn(n).astype(A.dtype)
    v = v / cp.linalg.norm(v)
    
    sigma_min = 0.0
    converged = False
    
    # Strategy 1: Dense solve (for small matrices)
    if use_dense_solve:
        try:
            # Form dense A^T A
            ATA_dense = (A.T @ A).toarray()
            
            # Add small shift for numerical stability
            shift = breakdown_tol
            ATA_shifted = ATA_dense + shift * cp.eye(n, dtype=A.dtype)
            
            for k in range(max_iter):
                # Solve (A^T A + shift·I) w = v
                try:
                    w = cp.linalg.solve(ATA_shifted, v)
                except cp.linalg.LinAlgError as e:
                    if verbose:
                        print(f"  Solve failed at iteration {k+1}: {e}")
                    break
                
                # Check for NaN/Inf
                if cp.isnan(w).any() or cp.isinf(w).any():
                    if verbose:
                        print(f"  Invalid values at iteration {k+1}")
                    break
                
                # Normalize
                norm_w = cp.linalg.norm(w)
                if norm_w < breakdown_tol:
                    if verbose:
                        print(f"  Breakdown at iteration {k+1}: ||w|| too small")
                    sigma_min = 0.0
                    converged = True
                    break
                
                w = w / norm_w
                
                # Rayleigh quotient on ORIGINAL A^T A (without shift)
                ATAw = ATA_dense @ w
                sigma_min_sq = float(cp.dot(w, ATAw))
                sigma_min = cp.sqrt(cp.abs(sigma_min_sq))
                
                if verbose and (k + 1) % 10 == 0:
                    print(f"  iter={k+1:3d} σ_min~={float(sigma_min):.6e}")
                
                # Convergence check
                vec_diff = cp.linalg.norm(w - v)
                if float(vec_diff) < tol:
                    converged = True
                    if verbose:
                        print(f"  Converged at iteration {k+1}")
                    break
                
                v = w
            
        except cp.cuda.memory.OutOfMemoryError:
            if verbose:
                print("  OOM on dense solve, falling back to sparse CG")
            use_dense_solve = False
    
    # Strategy 2: Sparse CG solve (for large matrices)
    if not use_dense_solve:
        # Build sparse A^T A + shift·I operator
        shift = breakdown_tol
        ATA = A.T @ A
        ATA_shifted = ATA + shift * sp.eye(n, dtype=A.dtype, format='csr')
        
        def matvec_ata(x):
            """Compute (A^T A) x without shift (for Rayleigh quotient)"""
            return A.T @ (A @ x)
        
        for k in range(max_iter):
            # Solve (A^T A + shift·I) w = v using CG
            try:
                w, info = cg(ATA_shifted, v, maxiter=min(50, n), tol=1e-6)
                
                if info != 0:
                    if verbose:
                        print(f"  CG did not converge at iteration {k+1}, info={info}")
                    # Relax tolerance and retry
                    w, info = cg(ATA_shifted, v, maxiter=min(100, n), tol=1e-3)
                    if info != 0:
                        if verbose:
                            print(f"  CG failed again, breaking")
                        break
                
            except Exception as e:
                if verbose:
                    print(f"  CG solve failed at iteration {k+1}: {e}")
                break
            
            # Check for NaN/Inf
            if cp.isnan(w).any() or cp.isinf(w).any():
                if verbose:
                    print(f"  Invalid values at iteration {k+1}")
                break
            
            # Normalize
            norm_w = cp.linalg.norm(w)
            if norm_w < breakdown_tol:
                if verbose:
                    print(f"  Breakdown at iteration {k+1}: ||w|| too small")
                sigma_min = 0.0
                converged = True
                break
            
            w = w / norm_w
            
            # Rayleigh quotient on ORIGINAL A^T A (without shift)
            ATAw = matvec_ata(w)
            sigma_min_sq = float(cp.dot(w, ATAw))
            sigma_min = cp.sqrt(cp.abs(sigma_min_sq))
            
            if verbose and (k + 1) % 10 == 0:
                print(f"  iter={k+1:3d} σ_min~={float(sigma_min):.6e}")
            
            # Convergence check
            vec_diff = cp.linalg.norm(w - v)
            if float(vec_diff) < tol:
                converged = True
                if verbose:
                    print(f"  Converged at iteration {k+1}")
                break
            
            v = w
    
    if verbose:
        print(f"  Final σ_min = {float(sigma_min):.6e}, converged={converged}")
    
    return float(sigma_min), min(k + 1, max_iter), converged


# ============================================================================
# Wrapper function for compatibility with your existing code
# ============================================================================

def compute_condition_power_sparse_kappa(
    A: Any,  # scipy.sparse or PyTorch backend sparse
    max_iter: int = 10
) -> float:
    """
    Wrapper function for sparse-kappa power method (aligned with PyTorch).
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix or PyTorch backend sparse.spmatrix
        Input sparse matrix
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    float
        Estimated condition number
    """
    import scipy.sparse
    
    # Convert scipy.sparse to PyTorch backend sparse if needed
    if isinstance(A, scipy.sparse.spmatrix):
        A_coo = A.tocoo()
        indices = cp.array(np.vstack([A_coo.row, A_coo.col]))
        values = cp.array(A_coo.data)
        A_torch_sparse = sp.coo_matrix(
            (values, (indices[0], indices[1])),
            shape=A.shape
        ).tocsr()
    else:
        A_torch_sparse = A
    
    try:
        result = power_method_cond(
            A_torch_sparse,
            max_iter=max_iter,
            tol=1e-12,  # Aligned with PyTorch
            verbose=False
        )
        
        cond = float(result['condition_number'])
        
        # Mirror the main estimator guardrail for wrapper callers.
        if not np.isfinite(cond):
            return 1e30
        if cond < 1:
            return 1.0
        return cond
        
    except Exception as e:
        print(f"sparse-kappa Power (iter={max_iter}) failed: {e}")
        return 1e30