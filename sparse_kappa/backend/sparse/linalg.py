"""PyTorch implementations of sparse linear-algebra helpers."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch


class LinearOperator:
    def __init__(self, shape: Tuple[int, int], matvec: Callable[[torch.Tensor], torch.Tensor], dtype: torch.dtype):
        self.shape = shape
        self.matvec = matvec
        self.dtype = dtype

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        return self.matvec(x)

    def toarray(self) -> torch.Tensor:
        n = self.shape[1]
        eye = torch.eye(n, dtype=self.dtype, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return torch.stack([self.matvec(eye[:, i]) for i in range(n)], dim=1)


def aslinearoperator(A: Any) -> LinearOperator:
    if isinstance(A, LinearOperator):
        return A
    dense = _dense(A)
    return LinearOperator(dense.shape, matvec=lambda x: dense @ x, dtype=dense.dtype)


def _dense(A: Any) -> torch.Tensor:
    if isinstance(A, LinearOperator):
        return A.toarray()
    if hasattr(A, "toarray"):
        return A.toarray()
    if isinstance(A, torch.Tensor):
        return A.to_dense() if A.is_sparse else A
    return torch.as_tensor(A)


def norm(A: Any, ord: Any = None) -> torch.Tensor:
    dense = _dense(A)
    return torch.linalg.matrix_norm(dense) if ord is None else torch.linalg.matrix_norm(dense, ord=ord)


def _solve(A: Any, b: torch.Tensor) -> torch.Tensor:
    dense = _dense(A)
    try:
        return torch.linalg.solve(dense, b)
    except torch.linalg.LinAlgError:
        return torch.linalg.lstsq(dense, b).solution


def spsolve(A: Any, b: torch.Tensor) -> torch.Tensor:
    return _solve(A, b)


def lsmr(A: Any, b: torch.Tensor, **kwargs):
    return (_solve(A, b), 0, 0, 0.0)


def lsqr(A: Any, b: torch.Tensor, **kwargs):
    return (_solve(A, b), 0, 0, 0.0)


def cg(A: Any, b: torch.Tensor, **kwargs):
    return _solve(A, b), 0


def gmres(A: Any, b: torch.Tensor, **kwargs):
    return _solve(A, b), 0


def bicgstab(A: Any, b: torch.Tensor, **kwargs):
    return _solve(A, b), 0


def eigsh(
    A: Any,
    k: int = 6,
    which: str = "LA",
    maxiter: Optional[int] = None,
    tol: float = 0.0,
    return_eigenvectors: bool = True,
    **kwargs,
):
    dense = _dense(A)
    dense = (dense + dense.mT.conj()) / 2
    vals, vecs = torch.linalg.eigh(dense)
    k = max(1, min(k, vals.numel()))
    if which in {"LA", "LM"}:
        idx = torch.argsort(torch.abs(vals) if which == "LM" else vals, descending=True)[:k]
    elif which in {"SA", "SM"}:
        idx = torch.argsort(torch.abs(vals) if which == "SM" else vals)[:k]
    else:
        raise ValueError(f"Unsupported which={which}")
    vals = vals[idx]
    vecs = vecs[:, idx]
    return (vals, vecs) if return_eigenvectors else vals


def svds(
    A: Any,
    k: int = 6,
    which: str = "LM",
    return_singular_vectors: bool = True,
    maxiter: Optional[int] = None,
    tol: float = 0.0,
    **kwargs,
):
    dense = _dense(A)
    U, S, Vh = torch.linalg.svd(dense, full_matrices=False)
    k = max(1, min(k, S.numel()))
    idx = torch.argsort(S, descending=(which == "LM"))[:k]
    S = S[idx]
    if not return_singular_vectors:
        return S
    return U[:, idx], S, Vh[idx, :]


def lobpcg(A: Any, X: torch.Tensor, maxiter: int = 100, tol: float = 1e-6, **kwargs):
    dense = _dense(A)
    vals, vecs = torch.linalg.eigh((dense + dense.mT.conj()) / 2)
    k = X.shape[1] if X.ndim == 2 else 1
    idx = torch.argsort(vals, descending=True)[:k]
    return vals[idx], vecs[:, idx]
