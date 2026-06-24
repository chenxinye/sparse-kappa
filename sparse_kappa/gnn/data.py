"""Dataset containers for GNN condition number prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch


@dataclass
class MatrixGraph:
    """Tensor graph representation of a sparse matrix."""

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    global_features: torch.Tensor
    shape: Tuple[int, int]
    matrix: Any = None
    target: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None

    def to(self, device: torch.device) -> "MatrixGraph":
        """Move graph tensors to a device."""
        return MatrixGraph(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            global_features=self.global_features.to(device),
            shape=self.shape,
            matrix=self.matrix,
            target=None if self.target is None else self.target.to(device),
            metadata=self.metadata,
        )


class MatrixConditionDataset:
    """
    Lightweight dataset for sparse matrices and scalar labels.

    Samples can be dictionaries with a ``matrix`` key, tuples ``(matrix, label)``,
    or raw matrices when labels are supplied separately.
    """

    def __init__(
        self,
        samples: Iterable[Any],
        labels: Optional[Sequence[float]] = None,
        matrix_key: str = "matrix",
        condition_key: str = "condition_number",
        inverse_norm_key: str = "norm_Ainv",
        metadata_key: str = "metadata",
    ):
        self.samples = list(samples)
        self.labels = None if labels is None else list(labels)
        self.matrix_key = matrix_key
        self.condition_key = condition_key
        self.inverse_norm_key = inverse_norm_key
        self.metadata_key = metadata_key

        if self.labels is not None and len(self.labels) != len(self.samples):
            raise ValueError("labels must have the same length as samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Any]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        label = None if self.labels is None else self.labels[index]

        if isinstance(sample, Mapping):
            out = dict(sample)
            if label is not None:
                out.setdefault(self.condition_key, label)
            if self.matrix_key not in out:
                raise KeyError(f"sample mapping must contain {self.matrix_key!r}")
            return out

        if isinstance(sample, tuple) and len(sample) >= 2:
            matrix, tuple_label = sample[0], sample[1]
            metadata = sample[2] if len(sample) > 2 else None
            return {
                self.matrix_key: matrix,
                self.condition_key: label if label is not None else tuple_label,
                self.metadata_key: metadata,
            }

        out = {self.matrix_key: sample}
        if label is not None:
            out[self.condition_key] = label
        return out

    def get_label(self, sample: Mapping[str, Any], target: str) -> float:
        if target == "condition":
            key = self.condition_key
        elif target == "inverse_norm":
            key = self.inverse_norm_key
        else:
            raise ValueError("target must be 'condition' or 'inverse_norm'")

        if key not in sample:
            raise KeyError(f"sample is missing label key {key!r}")
        return float(sample[key])
