"""
Hager algorithm for 1-norm condition number estimation.

This implementation estimates ||A^{-1}||_1 using Hager's original
single-vector 1-norm estimation algorithm applied implicitly to B = A^{-1}.

References
----------
- Hager, W. W. (1984). Condition estimates.
  SIAM Journal on Scientific and Statistical Computing, 5(2), 311-316.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from typing import Dict, Any, Tuple
from ..utils import sparse_matrix_norm, print_iteration
from ..solvers import create_solver


def hager_norm1_cond(
    A: sp.spmatrix,
    max_iter: int = 5,
    verbose: bool = False,
    solver: str = "lu",
    solver_kwargs: Dict[str, Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Estimate the 1-norm condition number kappa_1(A) = ||A||_1 * ||A^{-1}||_1
    using Hager's algorithm.

    Parameters
    ----------
    A : sparse matrix
        Input square real matrix (n x n).
    max_iter : int, default=5
        Maximum number of Hager iterations.
    verbose : bool, default=False
        Whether to print iteration information.
    solver : str, default='lu'
        Linear solver used for repeated solves with A and A^T.
        Direct LU-type solvers are recommended.
    solver_kwargs : dict, optional
        Additional keyword arguments passed to create_solver().

    Returns
    -------
    result : dict
        - 'condition_number': estimated kappa_1(A)
        - 'norm_A': exact ||A||_1
        - 'norm_Ainv': estimated ||A^{-1}||_1
        - 'iterations': number of iterations performed
        - 'converged': whether the Hager stopping criterion was met
        - 'solver_info': solver metadata

    Notes
    -----
    This is the original Hager-type iteration, not the later Higham
    improved variant.

    The algorithm estimates ||A^{-1}||_1 by applying the 1-norm estimator
    to B = A^{-1} implicitly:
        1. Start with x = ones(n) / n
        2. Solve A w = x
        3. Let est = ||w||_1
        4. Solve A^T z = sign(w)
        5. If ||z||_inf <= z^T x, stop
        6. Set x = e_j, where j = argmax |z_j|
        7. Repeat
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Hager 1-norm condition estimation requires a square matrix.")

    n = A.shape[0]

    if cp.iscomplexobj(A.data):
        raise ValueError(
            "This implementation only supports real matrices. "
            "For complex matrices, a separate complex-sign variant is needed."
        )

    if solver_kwargs is None:
        solver_kwargs = {}

    # Exact matrix 1-norm
    norm_A = sparse_matrix_norm(A, ord=1)

    if verbose:
        print("Hager 1-norm condition number estimation")
        print(f"Matrix size: {n} x {n}, NNZ: {A.nnz}")
        print(f"Solver: {solver}")
        print(f"||A||_1 = {norm_A:.6e}")
        print("Estimating ||A^{-1}||_1 using Hager iteration...")

    # Repeated solves with A and A^T
    solver_A = create_solver(A, solver, **solver_kwargs)
    solver_AT = create_solver(A.T, solver, **solver_kwargs)

    norm_Ainv, iterations, converged = _hager_inverse_norm_estimate(
        solver_A=solver_A,
        solver_AT=solver_AT,
        n=n,
        max_iter=max_iter,
        verbose=verbose,
    )

    cond = norm_A * norm_Ainv

    if verbose:
        print(f"\nEstimated ||A^(-1)||_1 = {norm_Ainv:.6e}")
        print(f"Estimated kappa_1(A) = {cond:.6e}")
        print(f"Converged: {converged} after {iterations} iterations")

        if hasattr(solver_A, "solve_count") and hasattr(solver_AT, "solve_count"):
            print(f"Total solves: A={solver_A.solve_count}, A^T={solver_AT.solve_count}")

    return {
        "condition_number": float(cond),
        "norm_A": float(norm_A),
        "norm_Ainv": float(norm_Ainv),
        "iterations": int(iterations),
        "converged": bool(converged),
        "solver_info": {
            "solver_type": solver,
            "solver_A": solver_A.info(),
            "solver_AT": solver_AT.info(),
        },
    }


def _hager_inverse_norm_estimate(
    solver_A,
    solver_AT,
    n: int,
    max_iter: int,
    verbose: bool = False,
) -> Tuple[float, int, bool]:
    """
    Estimate ||A^{-1}||_1 using Hager's original algorithm.

    Parameters
    ----------
    solver_A
        Solver object for systems with A.
    solver_AT
        Solver object for systems with A^T.
    n : int
        Matrix dimension.
    max_iter : int
        Maximum number of iterations.
    verbose : bool, default=False
        Whether to print progress.

    Returns
    -------
    est : float
        Estimated ||A^{-1}||_1.
    iterations : int
        Number of iterations performed.
    converged : bool
        Whether the Hager stopping criterion was satisfied.
    """
    dtype = solver_A.A.dtype

    # Initial vector x = ones(n)/n
    x = cp.ones(n, dtype=dtype) / n

    best_est = 0.0
    converged = False
    iterations = 0

    for k in range(max_iter):
        # Step 1: solve A w = x
        w = solver_A.solve(x)

        # Step 2: current estimate
        est = float(cp.linalg.norm(w, ord=1))
        best_est = max(best_est, est)
        iterations = k + 1

        if verbose:
            print_iteration(iterations, est)

        # Step 3: s = sign(w), with zeros mapped to +1
        s = cp.sign(w)
        s[s == 0] = 1.0

        # Step 4: solve A^T z = s
        z = solver_AT.solve(s)

        # Step 5: stopping criterion: ||z||_inf <= z^T x
        z_inf = float(cp.max(cp.abs(z)))
        ztx = float(cp.dot(z, x))

        if verbose:
            print(f"  ||z||_inf = {z_inf:.6e}, z^T x = {ztx:.6e}")

        if z_inf <= ztx:
            converged = True
            if verbose:
                print("  Hager stopping criterion satisfied")
            break

        # Step 6: choose j = argmax |z_j| and set x = e_j
        j = int(cp.argmax(cp.abs(z)))
        x_new = cp.zeros(n, dtype=dtype)
        x_new[j] = 1.0

        # Basic cycling guard: if x does not change, stop
        # This is not a Higham enhancement; it is just a practical safeguard.
        if j < n and cp.array_equal(x, x_new):
            converged = True
            if verbose:
                print("  Repeated basis vector encountered, stopping")
            break

        x = x_new

    return best_est, iterations, converged