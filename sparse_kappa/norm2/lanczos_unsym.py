"""
Arnoldi/Lanczos-style method for 2-norm condition number estimation.

For PyTorch, we use `eigsh` on A^H A because A^H A is Hermitian positive
semidefinite. This is more appropriate than `eigs` and is supported by
`PyTorch backend sparse.linalg`.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from sparse_kappa.backend.sparse.linalg import LinearOperator, eigsh
from typing import Dict, Any
from ..utils import print_iteration


def lanczos_unsym_cond(
    A: sp.spmatrix,
    max_iter: int = 50,
    tol: float = 1e-6,
    verbose: bool = False,
    restart: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """
    Estimate the 2-norm condition number using a Lanczos/Arnoldi-style method.

    We apply `eigsh` to A^H A to estimate the extreme eigenvalues:
        lambda_max(A^H A) = sigma_max(A)^2
        lambda_min(A^H A) = sigma_min(A)^2

    Parameters
    ----------
    A : sparse matrix
        Input matrix
    max_iter : int, default=50
        Maximum eigensolver iterations
    tol : float, default=1e-6
        Convergence tolerance
    verbose : bool, default=False
        Print progress information
    restart : int, default=20
        Restart / Krylov subspace size hint. Mapped to `ncv` for eigsh.

    Returns
    -------
    result : dict
        - 'condition_number': estimated kappa_2(A)
        - 'sigma_max': largest singular value
        - 'sigma_min': smallest singular value
        - 'iterations': number of iterations used (estimated)
        - 'converged': convergence status
    """
    m, n = A.shape
    min_dim = min(m, n)

    if verbose:
        print("Arnoldi/Lanczos method for 2-norm condition number")
        print(f"Matrix size: {m} x {n}")

    # Degenerate cases
    if min_dim == 0:
        return {
            'condition_number': float('inf'),
            'sigma_max': 0.0,
            'sigma_min': 0.0,
            'iterations': 0,
            'converged': False,
        }

    # For a single singular value, kappa_2 is 1 if nonzero, else inf.
    if min_dim == 1:
        # A has only one singular value = ||A||
        col = A @ cp.ones((n,), dtype=A.dtype)
        sigma = float(cp.linalg.norm(col))
        cond = 1.0 if sigma > 0 else float('inf')
        return {
            'condition_number': cond,
            'sigma_max': sigma,
            'sigma_min': sigma,
            'iterations': 1,
            'converged': True,
        }

    # Build LinearOperator for A^H A
    # Use conjugate transpose to support complex matrices correctly.
    def matvec(v):
        return A.T.conj() @ (A @ v)

    # A^H A is Hermitian PSD on C^n / R^n
    ata_dtype = cp.result_type(A.dtype, cp.float32)
    ATA = LinearOperator((n, n), matvec=matvec, dtype=ata_dtype)

    # `eigsh` requires 1 <= k < n. We only need the extreme eigenvalue.
    k = 1

    # Map restart to ncv. eigsh requires k + 1 < ncv < n.
    # Choose a safe value if possible.
    if n >= 4:
        ncv = min(max(restart, k + 2), n - 1)
        if ncv <= k + 1:
            ncv = min(n - 1, k + 3)
    else:
        ncv = None

    try:
        # Largest algebraic eigenvalue of A^H A = sigma_max^2
        eigvals_large = eigsh(
            ATA,
            k=k,
            which='LA',
            ncv=ncv,
            maxiter=max_iter,
            tol=tol,
            return_eigenvectors=False,
        )

        # Smallest algebraic eigenvalue of A^H A = sigma_min^2
        eigvals_small = eigsh(
            ATA,
            k=k,
            which='SA',
            ncv=ncv,
            maxiter=max_iter,
            tol=tol,
            return_eigenvectors=False,
        )

        lambda_max = cp.real(eigvals_large[0])
        lambda_min = cp.real(eigvals_small[0])

        # Protect against tiny negative values from numerical roundoff
        lambda_max = cp.maximum(lambda_max, 0)
        lambda_min = cp.maximum(lambda_min, 0)

        sigma_max = float(cp.sqrt(lambda_max))
        sigma_min = float(cp.sqrt(lambda_min))

        if sigma_min == 0.0:
            cond = float('inf')
        else:
            cond = float(sigma_max / sigma_min)

        converged = True
        iterations = max_iter  # eigsh does not expose actual iteration count

    except Exception as e:
        if verbose:
            print(f"Warning: eigsh failed: {e}, using fallback")

        # Fallback: use your existing power / inverse-like routines
        from .power_method import compute_sigma_max, compute_sigma_min

        sigma_max, iter1, conv1 = compute_sigma_max(A, max_iter, tol, False)
        sigma_min, iter2, conv2 = compute_sigma_min(A, max_iter, tol, False)

        cond = float('inf') if sigma_min == 0.0 else float(sigma_max / sigma_min)
        converged = bool(conv1 and conv2)
        iterations = int(iter1 + iter2)

    if verbose:
        print(f"\nσ_max = {sigma_max:.6e}")
        print(f"σ_min = {sigma_min:.6e}")
        print(f"kappa_2(A) = {cond:.6e}")

    return {
        'condition_number': float(cond),
        'sigma_max': float(sigma_max),
        'sigma_min': float(sigma_min),
        'iterations': int(iterations),
        'converged': bool(converged),
    }