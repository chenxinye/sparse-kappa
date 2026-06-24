from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend.sparse import linalg as splinalg


def bicgstab(
    A,
    b,
    x0=None,
    *,
    rtol=1e-5,
    atol=0.0,
    maxiter=None,
    M=None,
    callback=None,
):
    """
    PyTorch implementation of BiCGSTAB with SciPy-like interface.

    Parameters
    ----------
    A : torch.Tensor, PyTorch backend sparse.spmatrix, or LinearOperator
        Matrix or linear operator representing the system.
    b : torch.Tensor
        Right-hand side, shape (n,) or (n, 1).
    x0 : torch.Tensor, optional
        Initial guess. Defaults to zero vector.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    maxiter : int, optional
        Maximum number of iterations. Defaults to 2*n if None.
    M : torch.Tensor, PyTorch backend sparse.spmatrix, or LinearOperator, optional
        Left preconditioner approximating A^{-1}. Applied as z = M @ r.
    callback : callable, optional
        User callback called as callback(xk) on each iteration.

    Returns
    -------
    x : torch.Tensor
        Approximate solution.
    info : int
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : breakdown / illegal input
    """
    A = splinalg.aslinearoperator(A)
    n, m = A.shape
    if n != m:
        raise ValueError(f"A must be square, got shape={A.shape}")

    b = cp.asarray(b)
    if b.ndim == 2:
        if 1 in b.shape:
            b = b.reshape(-1)
        else:
            raise ValueError(f"b must be a vector, got shape={b.shape}")
    elif b.ndim != 1:
        raise ValueError(f"b must be a vector, got ndim={b.ndim}")

    if b.shape[0] != n:
        raise ValueError(f"dimension mismatch: A.shape={A.shape}, b.shape={b.shape}")

    dtype = cp.result_type(getattr(A, "dtype", b.dtype), b.dtype, cp.float32)
    b = b.astype(dtype, copy=False)

    if x0 is None:
        x = cp.zeros_like(b)
    else:
        x = cp.asarray(x0, dtype=dtype).reshape(-1).copy()
        if x.shape != b.shape:
            raise ValueError(f"x0 shape mismatch: expected {b.shape}, got {x.shape}")

    if maxiter is None:
        maxiter = 2 * n

    if M is not None:
        M = splinalg.aslinearoperator(M)

        def apply_precond(v):
            z = M.matvec(v)
            return cp.asarray(z, dtype=dtype).reshape(-1)
    else:
        def apply_precond(v):
            return v

    def matvec(v):
        y = A.matvec(v)
        return cp.asarray(y, dtype=dtype).reshape(-1)

    bnrm2 = cp.linalg.norm(b)
    if bnrm2 == 0:
        return cp.zeros_like(b), 0

    r = b - matvec(x)
    rr = r.copy()

    # SciPy-like stopping threshold
    tol = max(float(atol), float(rtol) * float(bnrm2))

    rnrm2 = cp.linalg.norm(r)
    if rnrm2 <= tol:
        return x, 0

    rho_old = cp.asarray(1.0, dtype=dtype)
    alpha = cp.asarray(1.0, dtype=dtype)
    omega = cp.asarray(1.0, dtype=dtype)

    v = cp.zeros_like(b)
    p = cp.zeros_like(b)

    # breakdown threshold
    eps = cp.finfo(dtype).eps
    tiny = eps * eps

    for it in range(1, maxiter + 1):
        rho_new = cp.vdot(rr, r)  # conjugating first arg, OK for complex

        if cp.abs(rho_new) < tiny:
            return x, -10  # rho breakdown

        if it == 1:
            p = r.copy()
        else:
            if cp.abs(omega) < tiny:
                return x, -11  # omega breakdown
            beta = (rho_new / rho_old) * (alpha / omega)
            p = r + beta * (p - omega * v)

        phat = apply_precond(p)
        v = matvec(phat)

        rr_v = cp.vdot(rr, v)
        if cp.abs(rr_v) < tiny:
            return x, -12  # alpha denominator breakdown

        alpha = rho_new / rr_v
        s = r - alpha * v

        snrm2 = cp.linalg.norm(s)
        if snrm2 <= tol:
            x = x + alpha * phat
            if callback is not None:
                callback(x)
            return x, 0

        shat = apply_precond(s)
        t = matvec(shat)

        tt = cp.vdot(t, t)
        if cp.abs(tt) < tiny:
            return x, -13  # omega denominator breakdown

        omega = cp.vdot(t, s) / tt

        x = x + alpha * phat + omega * shat
        r = s - omega * t

        if callback is not None:
            callback(x)

        rnrm2 = cp.linalg.norm(r)
        if rnrm2 <= tol:
            return x, 0

        if cp.abs(omega) < tiny:
            return x, -11  # omega breakdown

        rho_old = rho_new

    return x, maxiter
