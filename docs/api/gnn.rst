GNN Prediction API
==================

.. automodule:: sparse_kappa.gnn
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The GNN module predicts condition numbers from sparse matrices. The default
pipeline has four replaceable parts:

* ``DefaultGraphFeatureExtractor`` builds a row/column bipartite graph.
* ``SparseMatrixGNN`` performs lightweight message passing with PyTorch only.
* ``TrainingConfig`` controls target mode, norm, scheduler, optimizer defaults,
  logging, and early stopping.
* ``GNNConditionEstimator`` trains, saves, loads, evaluates, and predicts.

Training
--------

.. code-block:: python

   from sparse_kappa import TrainingConfig, train_gnn_condition_estimator

   train_samples = [
       {"matrix": A0, "condition_number": 10.0, "norm_Ainv": 0.5},
       {"matrix": A1, "condition_number": 20.0, "norm_Ainv": 1.0},
   ]

   estimator = train_gnn_condition_estimator(
       train_samples,
       config=TrainingConfig(target="condition", norm=2, epochs=100),
       save_path="models/gnn_cond.pt",
   )

Inverse-Norm Mode
-----------------

.. code-block:: python

   from sparse_kappa import TrainingConfig, train_gnn_condition_estimator

   estimator = train_gnn_condition_estimator(
       train_samples,
       config=TrainingConfig(target="inverse_norm", norm=1),
   )

   result = estimator.predict(A_test, return_dict=True)
   print(result["condition_number"], result["norm_A"], result["norm_Ainv"])

Customization
-------------

.. code-block:: python

   estimator.fit(
       train_samples,
       val_data=val_samples,
       optimizer_factory=lambda params: torch.optim.Adam(params, lr=5e-4),
       scheduler_factory=lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50),
       loss_fn=torch.nn.SmoothL1Loss(),
       validator=my_validation_callback,
   )
