"""PyTorch-backed compatibility layer used by sparse-kappa."""

from . import torch_api
from . import sparse

__all__ = ["torch_api", "sparse"]
