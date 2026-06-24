"""
Block Higham-Tisseur 1-norm estimator for kappa_1(A).

Estimates ||A^{-1}||_1 implicitly using a block 1-norm estimator
applied to the linear operator B = A^{-1}, then returns

    kappa_1(A) ~= ||A||_1 * est(||A^{-1}||_1)

Reference
---------
Higham, N. J., & Tisseur, F. (2000).
A block algorithm for matrix 1-norm estimation,
with an application to 1-norm pseudospectra.
SIAM J. Matrix Anal. Appl., 21(4), 1185-1201.
"""


from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from typing import Dict, Any, Tuple
from ..utils import print_iteration, sparse_matrix_norm
from ..solvers import create_solver


def hager_higham_norm1(
    A: sp.spmatrix,
    max_iter: int = 10,
    verbose: bool = False,
    solver: str = "lu",
    solver_kwargs: Dict[str, Any] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Estimate the 1-norm condition number kappa_1(A) using the
    Hager-Higham 1-norm estimator.

    This method estimates ||A^{-1}||_1 via an iterative procedure
    that alternates between solving systems with A and A^T.

    Parameters
    ----------
    A : sparse matrix
        Real square sparse matrix.
    max_iter : int, default=10
        Maximum number of iterations.
    verbose : bool, default=False
        Whether to print iteration progress.
    solver : str, default='lu'
        Solver type used for linear solves.
    solver_kwargs : dict, optional
        Additional arguments passed to solver construction.

    Returns
    -------
    result : dict
        Dictionary containing:
        - condition_number : float
        - norm_A : float
        - norm_Ainv : float
        - iterations : int
        - converged : bool
        - stop_reason : str
        - solver_info : dict
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    n = A.shape[0]
    if n == 0:
        raise ValueError("Matrix must be nonempty.")

    if cp.iscomplexobj(A.data):
        raise ValueError("Only real matrices are supported.")

    if solver_kwargs is None:
        solver_kwargs = {}

    norm_A = sparse_matrix_norm(A, ord=1)

    # Precompute transpose to avoid repeated construction
    AT = A.T

    solver_A = create_solver(A, solver, **solver_kwargs)
    solver_AT = create_solver(AT, solver, **solver_kwargs)

    norm_Ainv, iterations, converged, stop_reason = _hager_higham_inverse_onenorm_estimate(
        solver_A, solver_AT, n, max_iter, verbose
    )

    cond = norm_A * norm_Ainv

    return {
        "condition_number": float(cond),
        "norm_A": float(norm_A),
        "norm_Ainv": float(norm_Ainv),
        "iterations": int(iterations),
        "converged": bool(converged),
        "stop_reason": stop_reason,
        "solver_info": {
            "solver_type": solver,
            "solver_A": solver_A.info(),
            "solver_AT": solver_AT.info(),
        },
    }


def _hager_higham_inverse_onenorm_estimate(
    solver_A,
    solver_AT,
    n: int,
    max_iter: int,
    verbose: bool = False,
) -> Tuple[float, int, bool, str]:
    """
    Estimate ||A^{-1}||_1 using the Hager-Higham algorithm.

    The iteration alternates:
        y = A^{-1} x
        z = A^{-T} sign(y)

    and updates x as a standard basis vector.

    Notes
    -----
    This implementation minimizes GPU synchronization and avoids
    repeated allocations.
    """
    dtype = getattr(getattr(solver_A, "A", None), "dtype", cp.float64)

    # Initialize vector x = ones / n
    x = cp.full(n, 1.0 / n, dtype=dtype)
    x_new = cp.zeros(n, dtype=dtype)

    est = 0.0
    est_old = 0.0
    prev_j = -1
    visited = set()

    converged = False
    stop_reason = "max_iter"

    for k in range(1, max_iter + 1):
        # Solve A^{-1} x
        y = solver_A.solve(x)

        # Compute 1-norm estimate
        est = float(cp.sum(cp.abs(y)).item())

        if verbose:
            print_iteration(k, est)

        # Compute sign vector
        s = cp.sign(y)
        s[s == 0] = 1.0

        # Solve A^{-T} s
        z = solver_AT.solve(s)
        abs_z = cp.abs(z)

        # Extract key scalar quantities (minimize sync)
        j = int(cp.argmax(abs_z).item())
        zmax = float(abs_z[j].item())
        gamma = float(cp.dot(z, x).item())

        # Stopping criteria
        if zmax <= gamma:
            converged = True
            stop_reason = "duality_gap_closed"
            break

        if j == prev_j or j in visited:
            converged = True
            stop_reason = "repeated_index"
            break

        if k > 1 and est <= est_old:
            est = est_old
            converged = True
            stop_reason = "estimate_not_increasing"
            break

        est_old = est
        prev_j = j
        visited.add(j)

        # Update x = e_j (reuse buffer)
        x_new.fill(0)
        x_new[j] = 1.0
        x, x_new = x_new, x

    return float(est), k, converged, stop_reason


# -------------------------
# Block Higham-Tisseur
# -------------------------

def block_higham_tisseur_norm1(
    A: sp.spmatrix,
    block_size: int = 2,
    max_iter: int = 5,
    verbose: bool = False,
    solver: str = "lu",
    solver_kwargs: Dict[str, Any] = None,
    random_state: int = 1234,
    **kwargs,
) -> Dict[str, Any]:
    """
    Estimate kappa_1(A) using a block 1-norm estimator
    inspired by Higham-Tisseur (2000).

    This method applies a block power-like iteration to the
    implicit operator B = A^{-1}.

    Parameters
    ----------
    A : sparse matrix
        Real square sparse matrix.
    block_size : int
        Number of block columns.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    dict
        Same structure as hager_higham_norm1.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    n = A.shape[0]
    t = min(block_size, n)

    if solver_kwargs is None:
        solver_kwargs = {}

    norm_A = sparse_matrix_norm(A, ord=1)

    AT = A.T
    solver_A = create_solver(A, solver, **solver_kwargs)
    solver_AT = create_solver(AT, solver, **solver_kwargs)

    norm_Ainv, iterations, converged, stop_reason = _block_inverse_onenorm_estimate(
        solver_A, solver_AT, n, t, max_iter, random_state, verbose
    )

    cond = norm_A * norm_Ainv

    return {
        "condition_number": float(cond),
        "norm_A": float(norm_A),
        "norm_Ainv": float(norm_Ainv),
        "iterations": int(iterations),
        "converged": bool(converged),
        "stop_reason": stop_reason,
        "solver_info": {
            "solver_type": solver,
            "solver_A": solver_A.info(),
            "solver_AT": solver_AT.info(),
        },
    }


def _block_inverse_onenorm_estimate(
    solver_A,
    solver_AT,
    n: int,
    t: int,
    max_iter: int,
    random_state: int,
    verbose: bool,
) -> Tuple[float, int, bool, str]:
    """
    Block 1-norm estimation following Algorithm 2.4 (Higham-Tisseur).

    This implementation focuses on reducing allocations and minimizing
    GPU synchronization overhead.
    """
    rng = cp.random.RandomState(random_state)
    dtype = getattr(getattr(solver_A, "A", None), "dtype", cp.float64)

    # Initialize block X
    X = cp.ones((n, t), dtype=dtype)
    if t > 1:
        for i in range(1, t):
            _resample_sign_column(X, i, rng)

    X /= float(n)

    S = cp.zeros((n, t), dtype=dtype)
    X_next = cp.zeros_like(X)

    est = 0.0
    est_old = 0.0
    ind_hist = cp.empty((0,), dtype=cp.int64)

    converged = False
    stop_reason = "max_iter"

    for k in range(1, max_iter + 1):
        # Apply inverse
        Y = _solve_matmat(solver_A, X)

        mags = cp.sum(cp.abs(Y), axis=0)
        j = int(cp.argmax(mags).item())
        est = float(mags[j].item())

        if verbose:
            print_iteration(k, est)

        if k > 1 and est <= est_old:
            est = est_old
            converged = True
            stop_reason = "estimate_not_increasing"
            break

        est_old = est

        # Compute sign matrix
        S = cp.sign(Y)
        S[S == 0] = 1.0

        # Apply transpose inverse
        Z = _solve_matmat(solver_AT, S)
        h = cp.max(cp.abs(Z), axis=1)

        # Select top indices
        ind = cp.argsort(-h)[:t]

        # Update X efficiently
        X_next.fill(0)
        X_next[ind, cp.arange(t)] = 1.0
        X, X_next = X_next, X

    return float(est), k, converged, stop_reason


# -------------------------
# Utilities
# -------------------------

def _solve_matmat(solver, B: cp.ndarray) -> cp.ndarray:
    """
    Solve A X = B.

    If solver supports batched RHS, use it.
    Otherwise fallback to column-wise solves.
    """
    try:
        X = solver.solve(B)
        if isinstance(X, cp.ndarray) and X.shape == B.shape:
            return X
    except Exception:
        pass

    X = cp.empty_like(B)
    for j in range(B.shape[1]):
        X[:, j] = solver.solve(B[:, j])
    return X


def _resample_sign_column(M: cp.ndarray, i: int, rng) -> None:
    """
    Replace column i with a random sign vector.
    """
    M[:, i] = (2 * rng.randint(0, 2, size=M.shape[0]) - 1).astype(M.dtype)