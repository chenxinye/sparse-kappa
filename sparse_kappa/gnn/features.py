"""Sparse matrix graph construction and default feature extraction."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from sparse_kappa.backend import sparse as sp

from .data import MatrixGraph


def matrix_to_dense_tensor(matrix: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert common dense/sparse matrix inputs to a dense torch tensor."""
    if sp.isspmatrix(matrix):
        tensor = matrix.toarray()
    elif isinstance(matrix, torch.Tensor):
        tensor = matrix.to_dense() if matrix.is_sparse else matrix
    elif hasattr(matrix, "toarray"):
        tensor = torch.as_tensor(matrix.toarray())
    else:
        tensor = torch.as_tensor(np.asarray(matrix))

    if tensor.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if tensor.is_complex():
        tensor = torch.real(tensor)
    return tensor.to(dtype=dtype)


class DefaultGraphFeatureExtractor:
    """
    Build a row/column bipartite graph from a sparse matrix.

    Row nodes occupy ``[0, m)`` and column nodes occupy ``[m, m+n)``. Each
    nonzero value creates two directed edges so the default model can pass
    messages in both directions.
    """

    node_feature_dim = 7
    edge_feature_dim = 4
    global_feature_dim = 8

    def __init__(self, dtype: torch.dtype = torch.float32, eps: float = 1e-12):
        self.dtype = dtype
        self.eps = eps

    def __call__(
        self,
        matrix: Any,
        target: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MatrixGraph:
        dense = matrix_to_dense_tensor(matrix, dtype=self.dtype)
        device = dense.device
        m, n = dense.shape
        abs_dense = dense.abs()
        nz = dense.nonzero(as_tuple=False)

        rows = nz[:, 0].long() if nz.numel() else torch.empty(0, dtype=torch.long, device=device)
        cols = nz[:, 1].long() if nz.numel() else torch.empty(0, dtype=torch.long, device=device)
        values = dense[rows, cols] if nz.numel() else torch.empty(0, dtype=self.dtype, device=device)

        row_abs = abs_dense.sum(dim=1)
        col_abs = abs_dense.sum(dim=0)
        row_sq = torch.sqrt((dense * dense).sum(dim=1).clamp_min(self.eps))
        col_sq = torch.sqrt((dense * dense).sum(dim=0).clamp_min(self.eps))
        row_nnz = (abs_dense > 0).sum(dim=1).to(self.dtype)
        col_nnz = (abs_dense > 0).sum(dim=0).to(self.dtype)

        scale = abs_dense.max().clamp_min(self.eps)
        max_dim = float(max(m, n, 1))
        row_nodes = torch.stack(
            [
                torch.ones(m, dtype=self.dtype, device=device),
                torch.zeros(m, dtype=self.dtype, device=device),
                row_nnz / max(float(n), 1.0),
                torch.log1p(row_abs) / torch.log1p(scale),
                row_sq / scale,
                torch.arange(m, dtype=self.dtype, device=device) / max_dim,
                torch.zeros(m, dtype=self.dtype, device=device),
            ],
            dim=1,
        )
        col_nodes = torch.stack(
            [
                torch.zeros(n, dtype=self.dtype, device=device),
                torch.ones(n, dtype=self.dtype, device=device),
                col_nnz / max(float(m), 1.0),
                torch.log1p(col_abs) / torch.log1p(scale),
                col_sq / scale,
                torch.arange(n, dtype=self.dtype, device=device) / max_dim,
                torch.zeros(n, dtype=self.dtype, device=device),
            ],
            dim=1,
        )

        diag_len = min(m, n)
        if diag_len:
            diag_abs = dense.diag().abs() / scale
            row_nodes[:diag_len, 6] = diag_abs
            col_nodes[:diag_len, 6] = diag_abs

        x = torch.cat([row_nodes, col_nodes], dim=0)

        src_forward = rows
        dst_forward = m + cols
        src_backward = m + cols
        dst_backward = rows
        edge_index = torch.stack(
            [
                torch.cat([src_forward, src_backward]),
                torch.cat([dst_forward, dst_backward]),
            ],
            dim=0,
        )
        edge_base = torch.stack([values / scale, values.abs() / scale, torch.sign(values)], dim=1)
        edge_attr = torch.cat(
            [
                torch.cat([edge_base, torch.ones(edge_base.shape[0], 1, device=device, dtype=self.dtype)], dim=1),
                torch.cat([edge_base, torch.zeros(edge_base.shape[0], 1, device=device, dtype=self.dtype)], dim=1),
            ],
            dim=0,
        )

        density = float(values.numel()) / float(max(m * n, 1))
        global_features = torch.tensor(
            [
                float(m) / max_dim,
                float(n) / max_dim,
                density,
                float(torch.log1p(values.abs().mean())) if values.numel() else 0.0,
                float(torch.log1p(scale)),
                float(torch.linalg.vector_norm(dense) / scale),
                self._symmetry_score(dense),
                float(diag_len) / max_dim,
            ],
            dtype=self.dtype,
            device=device,
        )

        target_tensor = None if target is None else torch.tensor(float(target), dtype=self.dtype, device=device)
        return MatrixGraph(
            x=x,
            edge_index=edge_index.long(),
            edge_attr=edge_attr,
            global_features=global_features,
            shape=(m, n),
            matrix=matrix,
            target=target_tensor,
            metadata=metadata,
        )

    def _symmetry_score(self, dense: torch.Tensor) -> float:
        if dense.shape[0] != dense.shape[1]:
            return 0.0
        denom = torch.linalg.matrix_norm(dense).clamp_min(self.eps)
        diff = torch.linalg.matrix_norm(dense - dense.mT)
        return float(1.0 / (1.0 + diff / denom))
