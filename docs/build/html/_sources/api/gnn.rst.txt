GNN API
=======

Main components
---------------

* ``TrainingConfig``: training/task configuration.
* ``GNNConditionEstimator``: fit, evaluate, save/load, and predict interface.
* ``train_gnn_condition_estimator``: one-shot training helper.
* ``SparseMatrixGNN``: neural model implementation.
* ``MatrixConditionDataset`` / ``MatrixGraph``: graph dataset structures.

Training example
----------------

.. code-block:: python

   from sparse_kappa import TrainingConfig, train_gnn_condition_estimator

   train_samples = [
       {'matrix': A, 'condition_number': 12.0, 'norm_Ainv': 0.6},
   ]

   config = TrainingConfig(target='condition', norm=2, epochs=50, lr=1e-3)
   estimator = train_gnn_condition_estimator(train_samples, config=config)

Prediction example
------------------

.. code-block:: python

   pred = estimator.predict(A)
   print(pred)
