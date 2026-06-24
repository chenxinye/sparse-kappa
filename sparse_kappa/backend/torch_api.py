"""Small array API surface backed by PyTorch.

The estimator code was originally written against a NumPy-like GPU array API. This module keeps
the same array-level vocabulary while executing all tensor work with PyTorch.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch

__version__ = torch.__version__
ndarray = torch.Tensor

float16 = torch.float16
float32 = torch.float32
float64 = torch.float64
complex64 = torch.complex64
complex128 = torch.complex128
int32 = torch.int32
int64 = torch.int64
bool_ = torch.bool


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_dtype(dtype: Any) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, np.dtype):
        dtype = dtype.type
    if dtype in (float, np.float64):
        return torch.float64
    if dtype in (np.float32,):
        return torch.float32
    if dtype in (int, np.int64):
        return torch.int64
    if dtype in (np.int32,):
        return torch.int32
    if dtype in (complex, np.complex128):
        return torch.complex128
    if dtype in (np.complex64,):
        return torch.complex64
    return dtype


def _as_tensor(x: Any, dtype: Any = None, device: Any = None) -> torch.Tensor:
    dtype = _normalize_dtype(dtype)
    device = torch.device(device) if device is not None else None
    if isinstance(x, torch.Tensor):
        out = x
        if dtype is not None or device is not None:
            out = out.to(dtype=dtype or out.dtype, device=device or out.device)
        return out
    arr = np.asarray(x)
    return torch.as_tensor(arr, dtype=dtype, device=device or _default_device())


def _patch_tensor_methods() -> None:
    if not hasattr(torch.Tensor, "astype"):
        torch.Tensor.astype = lambda self, dtype, copy=True: self.to(dtype=_normalize_dtype(dtype), copy=copy)  # type: ignore[attr-defined]
    if not hasattr(torch.Tensor, "copy"):
        torch.Tensor.copy = lambda self: self.clone()  # type: ignore[attr-defined]
    if not hasattr(torch.Tensor, "fill"):
        torch.Tensor.fill = lambda self, value: self.fill_(value)  # type: ignore[attr-defined]


_patch_tensor_methods()


def array(obj: Any, dtype: Any = None) -> torch.Tensor:
    return _as_tensor(obj, dtype=dtype)


def asarray(obj: Any, dtype: Any = None) -> torch.Tensor:
    return _as_tensor(obj, dtype=dtype)


def asnumpy(obj: Any) -> np.ndarray:
    if hasattr(obj, "toarray"):
        obj = obj.toarray()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    return np.asarray(obj)


def zeros(shape: Union[int, Sequence[int]], dtype: Any = float64) -> torch.Tensor:
    return torch.zeros(shape, dtype=_normalize_dtype(dtype), device=_default_device())


def ones(shape: Union[int, Sequence[int]], dtype: Any = float64) -> torch.Tensor:
    return torch.ones(shape, dtype=_normalize_dtype(dtype), device=_default_device())


def empty(shape: Union[int, Sequence[int]], dtype: Any = float64) -> torch.Tensor:
    return torch.empty(shape, dtype=_normalize_dtype(dtype), device=_default_device())


def empty_like(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def zeros_like(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def full(shape: Union[int, Sequence[int]], fill_value: Any, dtype: Any = float64) -> torch.Tensor:
    return torch.full((shape,) if isinstance(shape, int) else tuple(shape), fill_value, dtype=_normalize_dtype(dtype), device=_default_device())


def arange(*args: Any, dtype: Any = None) -> torch.Tensor:
    return torch.arange(*args, dtype=_normalize_dtype(dtype), device=_default_device())


def eye(n: int, m: Optional[int] = None, dtype: Any = float64, format: str = "csr") -> torch.Tensor:
    return torch.eye(n, m or n, dtype=_normalize_dtype(dtype), device=_default_device())


def logspace(start: float, stop: float, num: int, dtype: Any = float64) -> torch.Tensor:
    return torch.logspace(start, stop, num, dtype=_normalize_dtype(dtype), device=_default_device())


def concatenate(seq: Sequence[torch.Tensor], axis: int = 0) -> torch.Tensor:
    return torch.cat([_as_tensor(x) for x in seq], dim=axis)


def sort(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    return torch.sort(x, dim=axis).values


def flip(x: torch.Tensor, axis: Union[int, Sequence[int]] = 0) -> torch.Tensor:
    dims = (axis,) if isinstance(axis, int) else tuple(axis)
    return torch.flip(x, dims=dims)


def abs(x: Any) -> torch.Tensor:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return torch.abs(_as_tensor(x))


def sqrt(x: Any) -> torch.Tensor:
    return torch.sqrt(_as_tensor(x))


def sign(x: Any) -> torch.Tensor:
    return torch.sign(_as_tensor(x))


def real(x: Any) -> torch.Tensor:
    return torch.real(_as_tensor(x))


def maximum(x: Any, y: Any) -> torch.Tensor:
    return torch.maximum(_as_tensor(x), _as_tensor(y, dtype=_as_tensor(x).dtype, device=_as_tensor(x).device))


def max(x: torch.Tensor, axis: Optional[int] = None):
    return torch.max(x) if axis is None else torch.max(x, dim=axis).values


def min(x: torch.Tensor, axis: Optional[int] = None):
    return torch.min(x) if axis is None else torch.min(x, dim=axis).values


def mean(x: torch.Tensor, axis: Optional[int] = None):
    return torch.mean(x) if axis is None else torch.mean(x, dim=axis)


def sum(x: torch.Tensor, axis: Optional[int] = None):
    return torch.sum(x) if axis is None else torch.sum(x, dim=axis)


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.dot(x.reshape(-1), y.reshape(-1))


def vdot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.vdot(x.reshape(-1), y.reshape(-1))


def argmax(x: torch.Tensor, axis: Optional[int] = None):
    return torch.argmax(x) if axis is None else torch.argmax(x, dim=axis)


def argsort(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    return torch.argsort(x, dim=axis)


def isnan(x: Any) -> torch.Tensor:
    return torch.isnan(_as_tensor(x))


def isinf(x: Any) -> torch.Tensor:
    return torch.isinf(_as_tensor(x))


def iscomplexobj(x: Any) -> bool:
    if isinstance(x, torch.Tensor):
        return x.is_complex()
    return np.iscomplexobj(x)


def array_equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    return bool(torch.equal(x, y))


def result_type(*args: Any) -> torch.dtype:
    normalized = []
    for arg in args:
        normalized.append(_normalize_dtype(arg) if isinstance(arg, (torch.dtype, type, np.dtype)) else arg)
    if not normalized:
        return torch.get_default_dtype()
    dtype = normalized[0] if isinstance(normalized[0], torch.dtype) else _as_tensor(normalized[0]).dtype
    for arg in normalized[1:]:
        dtype = torch.promote_types(dtype, arg if isinstance(arg, torch.dtype) else _as_tensor(arg).dtype)
    return dtype


def finfo(dtype: Any):
    return torch.finfo(_normalize_dtype(dtype))


class _RandomState:
    def __init__(self, seed_value: Optional[int] = None):
        self.generator = torch.Generator(device=_default_device())
        if seed_value is not None:
            self.generator.manual_seed(seed_value)

    def randint(self, low: int, high: Optional[int] = None, size: Union[int, Sequence[int]] = None):
        if high is None:
            low, high = 0, low
        return torch.randint(low, high, (size,) if isinstance(size, int) else tuple(size), generator=self.generator, device=_default_device())

    def choice(self, a: Any, size: Union[int, Sequence[int]], replace: bool = True):
        return _choice(a, size=size, replace=replace, generator=self.generator)


def _choice(a: Any, size: Union[int, Sequence[int]], replace: bool = True, generator: Optional[torch.Generator] = None):
    size_tuple = (size,) if isinstance(size, int) else tuple(size)
    if isinstance(a, int):
        if replace:
            return torch.randint(0, a, size_tuple, generator=generator, device=_default_device())
        return torch.randperm(a, generator=generator, device=_default_device())[: math.prod(size_tuple)].reshape(size_tuple)
    values = _as_tensor(a)
    if replace:
        idx = torch.randint(0, values.numel(), size_tuple, generator=generator, device=values.device)
    else:
        idx = torch.randperm(values.numel(), generator=generator, device=values.device)[: math.prod(size_tuple)].reshape(size_tuple)
    return values[idx]


class _Random:
    RandomState = _RandomState

    @staticmethod
    def seed(seed_value: int) -> None:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

    @staticmethod
    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, device=_default_device())

    @staticmethod
    def standard_normal(size: Union[int, Sequence[int]]) -> torch.Tensor:
        return torch.randn((size,) if isinstance(size, int) else tuple(size), device=_default_device())

    @staticmethod
    def choice(a: Any, size: Union[int, Sequence[int]], replace: bool = True):
        return _choice(a, size=size, replace=replace)


random = _Random()


class _Linalg:
    LinAlgError = torch.linalg.LinAlgError

    @staticmethod
    def norm(x: torch.Tensor, ord: Any = None):
        if x.ndim == 1:
            return torch.linalg.vector_norm(x) if ord is None else torch.linalg.vector_norm(x, ord=ord)
        return torch.linalg.matrix_norm(x) if ord is None else torch.linalg.matrix_norm(x, ord=ord)

    @staticmethod
    def solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(A, b)

    @staticmethod
    def svd(A: torch.Tensor, compute_uv: bool = True):
        if compute_uv:
            return torch.linalg.svd(A, full_matrices=False)
        return torch.linalg.svdvals(A)


linalg = _Linalg()


class _CudaStreamNull:
    @staticmethod
    def synchronize() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()


class _CudaStream:
    null = _CudaStreamNull()


class _OutOfMemoryError(RuntimeError):
    pass


class _CudaMemory:
    OutOfMemoryError = torch.cuda.OutOfMemoryError if hasattr(torch.cuda, "OutOfMemoryError") else _OutOfMemoryError


class _Cuda:
    Stream = _CudaStream
    memory = _CudaMemory

    @staticmethod
    def synchronize() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()


cuda = _Cuda()
