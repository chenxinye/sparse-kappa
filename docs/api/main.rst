Main API
========

.. automodule:: sparse_kappa
   :members:
   :undoc-members:
   :show-inheritance:

Core Functions
--------------

cond_estimate
~~~~~~~~~~~~~

.. autofunction:: sparse_kappa.cond_estimate

   **Examples**:

   .. code-block:: python

      # Simple usage
      cond = cond_estimate(A)

      # Specific method and norm
      cond = cond_estimate(A, norm=1, method='hager-higham', solver='lu')

      # Get detailed results
      result = cond_estimate(A, norm=2, method='svds', return_dict=True)
      print(result['condition_number'])
      print(result['sigma_max'])
      print(result['sigma_min'])

ConditionNumberEstimator
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sparse_kappa.ConditionNumberEstimator
   :members:
   :undoc-members:
   :show-inheritance:

   **Example**:

   .. code-block:: python

      from sparse_kappa import ConditionNumberEstimator

      # Create estimator
      estimator = ConditionNumberEstimator(A, norm=2, method='svds')

      # Get estimate
      result = estimator.estimate()

      # Check properties
      print(estimator.properties)