"""Tests for GNN condition number prediction helpers."""

from pathlib import Path

import torch

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from sparse_kappa.gnn import (
    DefaultGraphFeatureExtractor,
    GNNConditionEstimator,
    MatrixConditionDataset,
    TrainingConfig,
    train_gnn_condition_estimator,
)


def _toy_samples():
    A1 = sp.diags(cp.array([1.0, 2.0, 4.0]), format="csr")
    A2 = sp.diags(cp.array([1.0, 3.0, 9.0]), format="csr")
    A3 = sp.diags(cp.array([2.0, 4.0, 8.0]), format="csr")
    return [
        {"matrix": A1, "condition_number": 4.0, "norm_Ainv": 1.0},
        {"matrix": A2, "condition_number": 9.0, "norm_Ainv": 1.0},
        {"matrix": A3, "condition_number": 4.0, "norm_Ainv": 0.5},
    ]


def test_default_feature_extractor_builds_bipartite_graph():
    extractor = DefaultGraphFeatureExtractor()
    graph = extractor(sp.diags(cp.array([1.0, 2.0, 4.0]), format="csr"))

    assert graph.x.shape == (6, extractor.node_feature_dim)
    assert graph.edge_attr.shape[1] == extractor.edge_feature_dim
    assert graph.edge_index.shape[0] == 2
    assert graph.global_features.shape == (extractor.global_feature_dim,)


def test_train_save_load_and_predict_direct_condition(tmp_path: Path):
    config = TrainingConfig(epochs=2, lr=1e-3, scheduler="none", target="condition")
    path = tmp_path / "gnn.pt"

    estimator = train_gnn_condition_estimator(_toy_samples(), save_path=path, config=config)
    pred = estimator.predict(sp.diags(cp.array([1.0, 2.0, 5.0]), format="csr"))
    loaded = GNNConditionEstimator.load(path)
    loaded_pred = loaded.predict(sp.diags(cp.array([1.0, 2.0, 5.0]), format="csr"))

    assert path.exists()
    assert pred > 0
    assert loaded_pred > 0


def test_inverse_norm_mode_reports_condition_components():
    config = TrainingConfig(epochs=1, lr=1e-3, scheduler="none", target="inverse_norm", norm=1)
    dataset = MatrixConditionDataset(_toy_samples())
    estimator = train_gnn_condition_estimator(dataset, config=config)
    result = estimator.predict(sp.diags(cp.array([1.0, 2.0, 5.0]), format="csr"), return_dict=True)

    assert result["condition_number"] > 0
    assert result["norm_A"] == 5.0
    assert result["norm_Ainv"] > 0
