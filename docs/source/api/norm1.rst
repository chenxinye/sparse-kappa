1-norm methods API
==================

Available functions
-------------------

* ``hager_norm1_cond``
* ``hager_higham_norm1``
* ``block_higham_tisseur_norm1``
* ``power_iteration_norm1``
* ``oettli_prager_norm1``
* ``monte_carlo_norm1``

When to choose each method
--------------------------

* ``hager-higham``: default high-accuracy production choice.
* ``block-higham``: block variant for stronger robustness on some matrices.
* ``power``: fast rough estimate.
* ``oettli-prager``: adaptive/random/hybrid sampling style strategies.
* ``monte-carlo``: stochastic baseline.

Example
-------

.. code-block:: python

   from sparse_kappa import cond_estimate

   result = cond_estimate(
       A,
       norm=1,
       method='hager-higham',
       solver='lu',
       return_dict=True,
   )
   print(result['condition_number'])
