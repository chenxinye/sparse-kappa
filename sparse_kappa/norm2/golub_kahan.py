"""
Golub-Kahan bidiagonalization for condition number estimation.

More stable than forming A^T A explicitly.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from typing import Dict, Any
from ..utils import print_iteration

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from typing import Dict, Any, Optional


def _build_bidiagonal(alpha: cp.ndarray, beta: cp.ndarray, k: int) -> cp.ndarray:
    """
    Build k x k upper bidiagonal matrix B_k from alpha[0:k], beta[0:k-1].
    """
    B = cp.zeros((k, k), dtype=alpha.dtype)
    for i in range(k):
        B[i, i] = alpha[i]
        if i < k - 1:
            B[i, i + 1] = beta[i]
    return B


def golub_kahan_cond(
    A: sp.spmatrix,
    max_iter: int = 100,
    tol: float = 1e-8,
    reorthogonalize: bool = True,
    v0: Optional[cp.ndarray] = None,
    verbose: bool = False,
    breakdown_tol: float = 1e-14,
    dtype=cp.float64,
) -> Dict[str, Any]:
    """
    Unified Golub-Kahan bidiagonalization for estimating both largest and smallest
    singular values, and hence kappa_2(A) = sigma_max / sigma_min.

    Parameters
    ----------
    A : PyTorch backend sparse.spmatrix
        Input sparse matrix of shape (m, n).
    max_iter : int
        Maximum number of Golub-Kahan steps.
    tol : float
        Relative convergence tolerance on both edge singular values.
    reorthogonalize : bool
        Whether to reorthogonalize Lanczos vectors against the accumulated bases.
        Recommended for better smallest-singular-value estimates.
    v0 : cp.ndarray or None
        Optional starting right vector of shape (n,).
    verbose : bool
        Print iteration information.
    breakdown_tol : float
        Threshold for early termination on alpha/beta breakdown.
    dtype :
        Working dtype, e.g. cp.float64.

    Returns
    -------
    result : dict
        {
            'condition_number': float,
            'sigma_max': float,
            'sigma_min': float,
            'iterations': int,
            'converged': bool,
            'alpha': cp.ndarray,
            'beta': cp.ndarray,
        }

    Notes
    -----
    At step k, builds the k x k bidiagonal matrix B_k and computes its dense SVD.
    The singular values of B_k are Ritz approximations to singular values of A.
    """
    m, n = A.shape
    if min(m, n) == 0:
        raise ValueError("A must have nonzero dimensions.")

    # storage for GK scalars
    alpha = cp.zeros(max_iter, dtype=dtype)
    beta = cp.zeros(max_iter, dtype=dtype)

    # optional basis storage for reorthogonalization
    V = cp.zeros((n, max_iter + 1), dtype=dtype) if reorthogonalize else None
    U = cp.zeros((m, max_iter), dtype=dtype) if reorthogonalize else None

    # initial v
    if v0 is None:
        v = cp.random.standard_normal(n).astype(dtype)
    else:
        v = cp.asarray(v0, dtype=dtype).copy()

    nv = cp.linalg.norm(v)
    if nv == 0:
        raise ValueError("Initial vector v0 must be nonzero.")
    v /= nv

    if reorthogonalize:
        V[:, 0] = v

    u_prev = cp.zeros(m, dtype=dtype)
    v_prev = cp.zeros(n, dtype=dtype)
    beta_prev = cp.array(0.0, dtype=dtype)

    sigma_max_old = None
    sigma_min_old = None
    converged = False
    actual_iter = 0

    for k in range(max_iter):
        # u_k = A v_k - beta_{k-1} u_{k-1}
        u = A @ v - beta_prev * u_prev

        if reorthogonalize and k > 0:
            # full reorthogonalization of u against previous U basis
            Uk = U[:, :k]
            u = u - Uk @ (Uk.T @ u)

        alpha_k = cp.linalg.norm(u)
        alpha[k] = alpha_k

        if alpha_k < breakdown_tol:
            actual_iter = k
            break

        u /= alpha_k

        if reorthogonalize:
            U[:, k] = u

        # v_{k+1} = A^T u_k - alpha_k v_k
        w = A.T @ u - alpha_k * v

        if reorthogonalize:
            Vk1 = V[:, :k + 1]
            w = w - Vk1 @ (Vk1.T @ w)

        beta_k = cp.linalg.norm(w)
        beta[k] = beta_k

        # current k-step bidiagonal is size (k+1) x (k+1) in 1-based indexing
        kk = k + 1
        Bk = _build_bidiagonal(alpha, beta, kk)

        # dense SVD on the small bidiagonal matrix
        s = cp.linalg.svd(Bk, compute_uv=False)
        s = cp.sort(s)  # ascending
        sigma_min = float(s[0])
        sigma_max = float(s[-1])

        if verbose:
            cond_est = float("inf") if sigma_min == 0.0 else sigma_max / sigma_min
            print(
                f"[GK] iter={kk:3d} "
                f"alpha={float(alpha_k):.3e} beta={float(beta_k):.3e} "
                f"sigma_min~={sigma_min:.6e} sigma_max~={sigma_max:.6e} "
                f"kappa~={cond_est:.6e}"
            )

        # convergence check on both ends
        if sigma_max_old is not None and sigma_min_old is not None:
            rel_max = abs(sigma_max - sigma_max_old) / max(abs(sigma_max), 1e-30)
            rel_min = abs(sigma_min - sigma_min_old) / max(abs(sigma_min), 1e-30)
            if rel_max < tol and rel_min < tol:
                converged = True
                actual_iter = kk
                break

        sigma_max_old = sigma_max
        sigma_min_old = sigma_min

        actual_iter = kk

        # breakdown after finishing current Ritz update
        if beta_k < breakdown_tol:
            converged = True
            break

        # advance
        u_prev = u
        v_prev = v
        beta_prev = beta_k
        v = w / beta_k

        if reorthogonalize:
            V[:, k + 1] = v

    # final values
    if sigma_max_old is None or sigma_min_old is None:
        raise RuntimeError("Golub-Kahan failed before producing any bidiagonal iterate.")

    cond = float("inf") if sigma_min_old == 0.0 else sigma_max_old / sigma_min_old

    return {
        "condition_number": float(cond),
        "sigma_max": float(sigma_max_old),
        "sigma_min": float(sigma_min_old),
        "iterations": int(actual_iter),
        "converged": bool(converged),
        "alpha": alpha[:actual_iter].copy(),
        "beta": beta[:actual_iter].copy(),
    }