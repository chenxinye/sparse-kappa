"""Training and inference API for GNN condition number prediction."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from sparse_kappa.gnn.data import MatrixConditionDataset, MatrixGraph
from sparse_kappa.gnn.features import DefaultGraphFeatureExtractor, matrix_to_dense_tensor
from sparse_kappa.gnn.models import SparseMatrixGNN


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
OptimizerFactory = Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer]
SchedulerFactory = Callable[[torch.optim.Optimizer], Any]


@dataclass
class TrainingConfig:
    """Default knobs for GNN estimator training."""

    norm: Union[int, float, str] = 2
    target: str = "condition"
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: Optional[str] = None
    log_targets: bool = True
    grad_clip: Optional[float] = 1.0
    verbose: bool = False
    scheduler: str = "plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    early_stopping_patience: Optional[int] = None


class GNNConditionEstimator:
    """
    Trainable estimator for condition numbers from sparse matrix graphs.

    ``target='condition'`` predicts ``kappa(A)`` directly. ``target='inverse_norm'``
    predicts ``||A^{-1}||`` and multiplies by ``||A||`` during prediction.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        feature_extractor: Optional[DefaultGraphFeatureExtractor] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.config = config or TrainingConfig()
        self.feature_extractor = feature_extractor or DefaultGraphFeatureExtractor()
        self.model = model or SparseMatrixGNN(
            node_feature_dim=self.feature_extractor.node_feature_dim,
            edge_feature_dim=self.feature_extractor.edge_feature_dim,
            global_feature_dim=self.feature_extractor.global_feature_dim,
        )
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    @property
    def device(self) -> torch.device:
        if self.config.device is not None:
            return torch.device(self.config.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        train_data: Iterable[Any],
        val_data: Optional[Iterable[Any]] = None,
        optimizer_factory: Optional[OptimizerFactory] = None,
        scheduler_factory: Optional[SchedulerFactory] = None,
        loss_fn: Optional[LossFn] = None,
        validator: Optional[Callable[["GNNConditionEstimator", Iterable[Any]], float]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> "GNNConditionEstimator":
        train_dataset = _ensure_dataset(train_data)
        val_dataset = None if val_data is None else _ensure_dataset(val_data)
        device = self.device
        self.model.to(device)

        optimizer = (
            optimizer_factory(self.model.parameters())
            if optimizer_factory is not None
            else torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        )
        scheduler = (
            scheduler_factory(optimizer)
            if scheduler_factory is not None
            else self._build_scheduler(optimizer)
        )
        loss_fn = loss_fn or nn.MSELoss()

        best_val = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            losses: List[float] = []
            for sample in train_dataset:
                graph = self._graph_from_sample(sample, train_dataset).to(device)
                target = self._target_tensor(train_dataset.get_label(sample, self.config.target), device)

                optimizer.zero_grad()
                pred = self.model(graph)
                loss = loss_fn(pred.reshape(()), target.reshape(()))
                loss.backward()
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            train_loss = float(sum(losses) / max(len(losses), 1))
            self.history["train_loss"].append(train_loss)

            if val_dataset is not None:
                val_loss = (
                    float(validator(self, val_dataset))
                    if validator is not None
                    else self.evaluate(val_dataset, loss_fn=loss_fn)
                )
                self.history["val_loss"].append(val_loss)
                metric = val_loss
            else:
                metric = train_loss

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(metric)
                else:
                    scheduler.step()

            if metric < best_val:
                best_val = metric
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
                if save_path is not None:
                    self.save(save_path)
            else:
                epochs_without_improvement += 1

            if self.config.verbose:
                msg = f"epoch={epoch:04d} train_loss={train_loss:.6e}"
                if val_dataset is not None:
                    msg += f" val_loss={metric:.6e}"
                print(msg)

            patience = self.config.early_stopping_patience
            if patience is not None and epochs_without_improvement >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if save_path is not None:
            self.save(save_path)
        return self

    def evaluate(self, data: Iterable[Any], loss_fn: Optional[LossFn] = None) -> float:
        dataset = _ensure_dataset(data)
        loss_fn = loss_fn or nn.MSELoss()
        device = self.device
        self.model.to(device)
        self.model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for sample in dataset:
                graph = self._graph_from_sample(sample, dataset).to(device)
                target = self._target_tensor(dataset.get_label(sample, self.config.target), device)
                pred = self.model(graph)
                losses.append(float(loss_fn(pred.reshape(()), target.reshape(())).cpu()))
        return float(sum(losses) / max(len(losses), 1))

    def predict(
        self,
        matrices: Union[Any, Sequence[Any]],
        return_dict: bool = False,
    ) -> Union[float, List[float], Dict[str, float], List[Dict[str, float]]]:
        is_single = not isinstance(matrices, (list, tuple))
        matrix_list = [matrices] if is_single else list(matrices)
        device = self.device
        self.model.to(device)
        self.model.eval()

        outputs: List[Dict[str, float]] = []
        with torch.no_grad():
            for matrix in matrix_list:
                graph = self.feature_extractor(matrix).to(device)
                pred_log = self.model(graph).reshape(())
                raw = self._inverse_target_transform(pred_log)
                norm_A = matrix_norm(matrix, self.config.norm)
                if self.config.target == "inverse_norm":
                    inverse_norm = raw
                    condition = norm_A * inverse_norm
                else:
                    inverse_norm = raw / norm_A if norm_A > 0 else float("inf")
                    condition = raw
                outputs.append(
                    {
                        "condition_number": float(condition),
                        "norm_A": float(norm_A),
                        "norm_Ainv": float(inverse_norm),
                        "target": self.config.target,
                        "norm": self.config.norm,
                    }
                )

        if is_single:
            return outputs[0] if return_dict else outputs[0]["condition_number"]
        return outputs if return_dict else [out["condition_number"] for out in outputs]

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "model_config": self.model.config() if hasattr(self.model, "config") else None,
            "training_config": asdict(self.config),
            "history": self.history,
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None,
        feature_extractor: Optional[DefaultGraphFeatureExtractor] = None,
        model: Optional[nn.Module] = None,
    ) -> "GNNConditionEstimator":
        payload = torch.load(path, map_location=map_location or "cpu")
        config = TrainingConfig(**payload["training_config"])
        feature_extractor = feature_extractor or DefaultGraphFeatureExtractor()
        if model is None:
            model_config = payload.get("model_config") or {}
            model = SparseMatrixGNN(**model_config)
        estimator = cls(model=model, feature_extractor=feature_extractor, config=config)
        estimator.model.load_state_dict(payload["model_state"])
        estimator.history = payload.get("history", {"train_loss": [], "val_loss": []})
        return estimator

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        if self.config.scheduler == "none":
            return None
        if self.config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
            )
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.config.epochs, 1))
        raise ValueError("scheduler must be 'plateau', 'cosine', or 'none'")

    def _graph_from_sample(self, sample: Mapping[str, Any], dataset: MatrixConditionDataset) -> MatrixGraph:
        target = None
        try:
            target = dataset.get_label(sample, self.config.target)
        except KeyError:
            target = None
        return self.feature_extractor(
            sample[dataset.matrix_key],
            target=target,
            metadata=sample.get(dataset.metadata_key),
        )

    def _target_tensor(self, value: float, device: torch.device) -> torch.Tensor:
        tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
        if self.config.log_targets:
            tensor = torch.log(tensor.clamp_min(1e-30))
        return tensor

    def _inverse_target_transform(self, value: torch.Tensor) -> float:
        if self.config.log_targets:
            return float(torch.exp(value).detach().cpu())
        return float(value.detach().cpu())


def train_gnn_condition_estimator(
    train_data: Iterable[Any],
    val_data: Optional[Iterable[Any]] = None,
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[TrainingConfig] = None,
    model: Optional[nn.Module] = None,
    feature_extractor: Optional[DefaultGraphFeatureExtractor] = None,
    optimizer_factory: Optional[OptimizerFactory] = None,
    scheduler_factory: Optional[SchedulerFactory] = None,
    loss_fn: Optional[LossFn] = None,
    validator: Optional[Callable[[GNNConditionEstimator, Iterable[Any]], float]] = None,
) -> GNNConditionEstimator:
    """Train a GNN condition estimator and optionally save the model."""
    estimator = GNNConditionEstimator(model=model, feature_extractor=feature_extractor, config=config)
    return estimator.fit(
        train_data,
        val_data=val_data,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        loss_fn=loss_fn,
        validator=validator,
        save_path=save_path,
    )


def matrix_norm(matrix: Any, norm: Union[int, float, str]) -> float:
    dense = matrix_to_dense_tensor(matrix, dtype=torch.float64)
    if norm == 2:
        return float(torch.linalg.matrix_norm(dense, ord=2))
    if norm == 1:
        return float(torch.linalg.matrix_norm(dense, ord=1))
    if norm in ("fro", "nuc", float("inf"), -float("inf"), -1, -2):
        return float(torch.linalg.matrix_norm(dense, ord=norm))
    return float(torch.linalg.matrix_norm(dense, ord=norm))


def _ensure_dataset(data: Iterable[Any]) -> MatrixConditionDataset:
    if isinstance(data, MatrixConditionDataset):
        return data
    return MatrixConditionDataset(data)
