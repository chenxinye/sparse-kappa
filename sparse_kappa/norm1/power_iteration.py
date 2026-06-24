"""
Power iteration method for 1-norm estimation with flexible solvers.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from typing import Dict, Any
from ..utils import print_iteration, sparse_matrix_norm
from ..solvers import create_solver


def power_iteration_norm1(
    A: sp.spmatrix,
    max_iter: int = 20,
    tol: float = 1e-6,
    verbose: bool = False,
    solver: str = 'auto',
    solver_kwargs: Dict[str, Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Estimate 1-norm condition number using power iteration.
    
    Parameters
    ----------
    A : sparse matrix
        Input matrix
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print information
    solver : str
        Solver type ('auto', 'lu', 'lsmr', etc.)
    solver_kwargs : dict
        Solver parameters
    
    Returns
    -------
    result : dict
        Condition number estimation results
    """
    n = A.shape[0]
    
    norm_A = sparse_matrix_norm(A, ord=1)
    
    if verbose:
        print("Power iteration for 1-norm condition number")
        print(f"Matrix size: {n} x {n}")
        print(f"Solver: {solver}")
        print(f"||A||_1 = {norm_A:.6e}")
    
    if solver_kwargs is None:
        solver_kwargs = {}
    
    solver_A = create_solver(A, solver, **solver_kwargs)
    
    # Initialize with random vector
    x = cp.random.randn(n).astype(A.dtype)
    x = x / cp.linalg.norm(x, ord=1)
    
    norm_estimates = []
    
    for it in range(max_iter):
        # y = A^{-1} x
        y = solver_A.solve(x)
        
        # Estimate ||A^{-1}||_1
        norm_y = cp.linalg.norm(y, ord=1)
        norm_estimates.append(float(norm_y))
        
        if verbose and (it + 1) % 5 == 0:
            print_iteration(it + 1, norm_y)
        
        # Check convergence
        if it > 0 and abs(norm_estimates[-1] - norm_estimates[-2]) < tol * norm_estimates[-1]:
            converged = True
            iterations = it + 1
            break
        
        # Update x for next iteration
        x = y / norm_y
    else:
        converged = False
        iterations = max_iter
    
    norm_Ainv = norm_estimates[-1]
    cond = norm_A * norm_Ainv
    
    if verbose:
        print(f"\n||A^{{-1}}||_1 = {norm_Ainv:.6e}")
        print(f"kappa_1(A) = {cond:.6e}")
        print(f"Total solves: {solver_A.solve_count}")
    
    return {
        'condition_number': float(cond),
        'norm_A': float(norm_A),
        'norm_Ainv': float(norm_Ainv),
        'iterations': iterations,
        'converged': converged,
        'solver_info': solver_A.info(),
    }