"""
PyTorch Sparse Condition Number Estimation Library

A comprehensive GPU-accelerated library for estimating condition numbers
of sparse matrices using various iterative methods.
"""

from .sparse_kappa import cond_estimate, ConditionNumberEstimator
from .solvers import create_solver, SOLVER_MAP
from .config import configure_warnings
from .gnn import GNNConditionEstimator, TrainingConfig, train_gnn_condition_estimator

__version__ = "0.0.2"
__all__ = [
    "cond_estimate", 
    "ConditionNumberEstimator",
    "create_solver",
    "SOLVER_MAP",
    "GNNConditionEstimator",
    "TrainingConfig",
    "train_gnn_condition_estimator",
]


configure_warnings(convergence=False, format_conversion=False)
