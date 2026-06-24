"""
Oettli-Prager method for 1-norm estimation with flexible solvers.

Based on linear programming formulation for matrix norm estimation.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from typing import Dict, Any
from ..utils import sparse_matrix_norm
from ..solvers import create_solver


def oettli_prager_norm1(
    A: sp.spmatrix,
    max_iter: int = 10,
    tol: float = 1e-6,
    verbose: bool = False,
    solver: str = 'auto',
    solver_kwargs: Dict[str, Any] = None,
    variant: str = 'adaptive',
    **kwargs
) -> Dict[str, Any]:
    """
    Estimate 1-norm condition number using Oettli-Prager method.
    
    The Oettli-Prager method reformulates the norm estimation as:
    ||A^{-1}||_1 = max_j ||A^{-1} e_j||_1
    
    Uses iterative refinement with dual-based column selection.
    
    Parameters
    ----------
    A : sparse matrix
        Input matrix (n x n)
    max_iter : int, default=10
        Maximum number of column samples to test
    tol : float, default=1e-6
        Convergence tolerance (unused for this method)
    verbose : bool, default=False
        Print information
    solver : str, default='auto'
        Solver type ('auto', 'lu', 'lsmr', 'cg', 'direct', 'gmres')
        - For Oettli-Prager, 'lu' is highly recommended due to multiple solves
    solver_kwargs : dict, optional
        Additional parameters for solver
    variant : str, default='adaptive'
        - 'adaptive': dual-based column selection (original Oettli-Prager)
        - 'random': random sampling (faster but less accurate)
        - 'hybrid': combination of both
    
    Returns
    -------
    result : dict
        - 'condition_number': estimated kappa_1(A)
        - 'norm_A': ||A||_1
        - 'norm_Ainv': ||A^{-1}||_1
        - 'iterations': number of columns tested
        - 'converged': convergence status
        - 'solver_info': solver information
    
    Algorithm
    ---------
    1. Start with column that has maximum ||A||_1
    2. Solve A x = e_j for selected columns
    3. Use dual information (A^T y = sign(x)) to select next column
    4. Iterate until convergence or max_iter
    
    Solver Recommendation
    ---------------------
    - **Best: 'lu'** - Multiple solves benefit greatly from factorization caching
    - Good: 'lsmr' with relaxed tolerance for large matrices
    - Avoid: 'direct' - no caching, will be slow
    
    Examples
    --------
    >>> # Recommended: LU solver for multiple solves
    >>> result = oettli_prager_norm1(A, max_iter=15, solver='lu', verbose=True)
    >>> 
    >>> # For large matrices: LSMR with relaxed tolerance
    >>> result = oettli_prager_norm1(A, max_iter=10, solver='lsmr',
    ...                              solver_kwargs={'atol': 1e-3, 'maxiter': 20})
    >>> 
    >>> # Random sampling variant (faster)
    >>> result = oettli_prager_norm1(A, max_iter=20, solver='lu', 
    ...                              variant='random')
    
    Reference
    ---------
    Oettli, W., & Prager, W. (1964). "Compatibility of approximate 
    solution of linear equations with given error bounds for 
    coefficients and right-hand sides." Numerische Mathematik.
    """
    n = A.shape[0]
    
    # Compute ||A||_1 exactly
    norm_A = sparse_matrix_norm(A, ord=1)
    
    if verbose:
        print("Oettli-Prager method for 1-norm condition number")
        print(f"Matrix size: {n} x {n}")
        print(f"Variant: {variant}")
        print(f"Solver: {solver}")
        print(f"||A||_1 = {norm_A:.6e}")
        print("Estimating ||A^{-1}||_1...")
    
    if solver_kwargs is None:
        solver_kwargs = {}
    
    # Create solvers (LU is highly recommended for this method)
    solver_A = create_solver(A, solver, **solver_kwargs)
    solver_AT = create_solver(A.T, solver, **solver_kwargs) if variant == 'adaptive' else None
    
    if verbose and solver == 'auto':
        print(f"Auto-selected solver: {solver_A.info()['solver_type']}")
    
    # Select algorithm variant
    if variant == 'adaptive':
        norm_Ainv, iterations, converged = _adaptive_oettli_prager(
            A, solver_A, solver_AT, n, max_iter, verbose
        )
    elif variant == 'random':
        norm_Ainv, iterations, converged = _random_sampling(
            A, solver_A, n, max_iter, verbose
        )
    elif variant == 'hybrid':
        norm_Ainv, iterations, converged = _hybrid_oettli_prager(
            A, solver_A, solver_AT, n, max_iter, verbose
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'adaptive', 'random', or 'hybrid'")
    
    cond = norm_A * norm_Ainv
    
    if verbose:
        print(f"\n||A^{{-1}}||_1 = {norm_Ainv:.6e}")
        print(f"kappa_1(A) = {cond:.6e}")
        print(f"Converged: {converged} after {iterations} iterations")
        print(f"Total solves: A={solver_A.solve_count}" + 
              (f", A^T={solver_AT.solve_count}" if solver_AT else ""))
    
    return {
        'condition_number': float(cond),
        'norm_A': float(norm_A),
        'norm_Ainv': float(norm_Ainv),
        'iterations': iterations,
        'converged': converged,
        'solver_info': {
            'solver_type': solver,
            'solver_A': solver_A.info(),
            'solver_AT': solver_AT.info() if solver_AT else None,
            'variant': variant,
        }
    }


def _adaptive_oettli_prager(A, solver_A, solver_AT, n, max_iter, verbose):
    """
    Adaptive column selection using dual problem.
    
    Original Oettli-Prager method with dual-based column selection.
    """
    # Find column with maximum norm in A (good starting point)
    A_abs = A.copy()
    A_abs.data = cp.abs(A_abs.data)
    col_norms = cp.array(A_abs.sum(axis=0)).ravel()
    
    j_max = int(cp.argmax(col_norms))
    
    tested_columns = set()
    norm_Ainv = 0.0
    best_column = None
    
    for it in range(max_iter):
        # Avoid testing same column twice
        if j_max in tested_columns:
            # Find next untested column with high potential
            found = False
            for idx in cp.flip(cp.argsort(col_norms), 0):
                if int(idx) not in tested_columns:
                    j_max = int(idx)
                    found = True
                    break
            
            if not found:
                converged = True
                break
        
        tested_columns.add(j_max)
        
        # Solve A x = e_j
        e_j = cp.zeros(n, dtype=A.dtype)
        e_j[j_max] = 1.0
        x = solver_A.solve(e_j)
        
        # Compute ||x||_1
        norm_x = float(cp.linalg.norm(x, ord=1))
        
        if verbose:
            print(f"Iter {it+1}: column {j_max}, ||A^{{-1}} e_{j_max}||_1 = {norm_x:.6e}")
        
        # Update best estimate
        if norm_x > norm_Ainv:
            norm_Ainv = norm_x
            best_column = j_max
            
            # Solve dual problem: A^T y = sign(x)
            sign_x = cp.sign(cp.real(x))
            sign_x[sign_x == 0] = 1
            
            y = solver_AT.solve(sign_x)
            
            # Find next column based on dual solution
            abs_y = cp.abs(y)
            j_max = int(cp.argmax(abs_y))
            
            if verbose:
                print(f"  Updated estimate: {norm_Ainv:.6e}, next column: {j_max}")
        else:
            # No improvement, try different column
            if verbose:
                print(f"  No improvement")
            
            for idx in cp.flip(cp.argsort(col_norms), 0):
                if int(idx) not in tested_columns:
                    j_max = int(idx)
                    break
            else:
                converged = True
                break
        
        if len(tested_columns) >= min(max_iter, n // 10):
            converged = True
            break
    else:
        converged = False
    
    return norm_Ainv, len(tested_columns), converged


def _random_sampling(A, solver_A, n, max_iter, verbose):
    """
    Random sampling approach (simpler, faster).
    
    Good for quick estimates or when LU factorization is available.
    """
    norm_Ainv = 0.0
    
    # Sample standard basis vectors first
    num_basis = min(max_iter, n)
    for i in range(num_basis):
        e_i = cp.zeros(n, dtype=A.dtype)
        e_i[i] = 1.0
        
        x = solver_A.solve(e_i)
        norm_x = float(cp.linalg.norm(x, ord=1))
        norm_Ainv = max(norm_Ainv, norm_x)
        
        if verbose and (i + 1) % 5 == 0:
            print(f"Sample {i+1}/{num_basis}: max ||A^{{-1}} e_i||_1 = {norm_Ainv:.6e}")
    
    # Sample random vectors if we have budget left
    if max_iter > num_basis:
        for i in range(max_iter - num_basis):
            b = cp.random.randn(n).astype(A.dtype)
            b = b / cp.linalg.norm(b, ord=1)
            
            x = solver_A.solve(b)
            norm_x = float(cp.linalg.norm(x, ord=1))
            norm_Ainv = max(norm_Ainv, norm_x)
    
    return norm_Ainv, max_iter, True


def _hybrid_oettli_prager(A, solver_A, solver_AT, n, max_iter, verbose):
    """
    Hybrid approach: adaptive selection + random sampling.
    
    Best of both worlds.
    """
    # Use adaptive for first half
    half_iter = max_iter // 2
    norm_Ainv_1, iters_1, _ = _adaptive_oettli_prager(
        A, solver_A, solver_AT, n, half_iter, verbose
    )
    
    # Use random for second half
    norm_Ainv_2, iters_2, _ = _random_sampling(
        A, solver_A, n, max_iter - half_iter, verbose
    )
    
    norm_Ainv = max(norm_Ainv_1, norm_Ainv_2)
    return norm_Ainv, iters_1 + iters_2, True