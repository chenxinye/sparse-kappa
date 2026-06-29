Examples
========

Basic 2-norm estimation
-----------------------

.. code-block:: python

   from sparse_kappa.backend import sparse as sp
   from sparse_kappa import cond_estimate

   A = sp.random(2000, 2000, density=0.005, format='csr')
   cond = cond_estimate(A, norm=2, method='svds')
   print(f"kappa_2(A) = {cond:.3e}")

Compare multiple methods
------------------------

.. code-block:: python

   methods = ['svds', 'lanczos', 'golub-kahan']
   for method in methods:
       value = cond_estimate(A, norm=2, method=method)
       print(f"{method:12s} -> {value:.3e}")

1-norm with LU solver
---------------------

.. code-block:: python

   result = cond_estimate(
       A,
       norm=1,
       method='hager-higham',
       solver='lu',
       return_dict=True,
   )
   print(result['condition_number'])
   print(result['solver_info'])

GNN training workflow
---------------------

.. code-block:: python

   from sparse_kappa import TrainingConfig, train_gnn_condition_estimator

   train_samples = [
       {'matrix': A, 'condition_number': 10.2, 'norm_Ainv': 0.4},
   ]

   config = TrainingConfig(target='condition', norm=2, epochs=20, lr=1e-3)
   estimator = train_gnn_condition_estimator(train_samples, config=config)
   pred = estimator.predict(A)
   print(pred)
