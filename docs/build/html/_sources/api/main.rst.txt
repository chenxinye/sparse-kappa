Main API
========

Primary entry points
--------------------

* ``cond_estimate(A, norm=2, method='auto', ...)``
  Main convenience function for condition number estimation.
* ``ConditionNumberEstimator(A, norm=2, method='auto', ...)``
  Class-based estimator for richer method selection and internal property handling.

``cond_estimate`` parameters
----------------------------

Common options:

* ``A``: input square matrix (sparse preferred).
* ``norm``: ``1`` or ``2``.
* ``method``: algorithm name (for example ``svds``, ``lanczos``, ``hager-higham``).
* ``max_iter`` / ``tol``: convergence controls.
* ``return_dict``: when ``True``, returns diagnostics in addition to the estimate.

Typical usage
-------------

.. code-block:: python

   from sparse_kappa.backend import sparse as sp
   from sparse_kappa import cond_estimate

   A = sp.random(1000, 1000, density=0.01, format='csr')

   cond = cond_estimate(A)  # auto-select 2-norm method
   detailed = cond_estimate(A, norm=2, method='svds', return_dict=True)

   print(cond)
   print(detailed['condition_number'], detailed['iterations'])

``ConditionNumberEstimator`` workflow
-------------------------------------

.. code-block:: python

   from sparse_kappa import ConditionNumberEstimator

   estimator = ConditionNumberEstimator(A, norm=1, method='hager-higham', solver='lu')
   result = estimator.estimate()
   print(result['method'], result['condition_number'])
