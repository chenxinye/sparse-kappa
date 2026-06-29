2-norm methods API
==================

Available functions
-------------------

* ``svds_cond``
* ``eigsh_cond``
* ``lobpcg_cond``
* ``power_method_cond``
* ``lanczos_cond``
* ``lanczos_unsym_cond``
* ``golub_kahan_cond``

Method guidance
---------------

* ``svds``: highest-accuracy baseline on small/medium matrices.
* ``eigsh``: symmetric/Hermitian-friendly mode.
* ``lanczos`` / ``lanczos_unsym``: scalable iterative alternatives.
* ``golub-kahan``: robust option for large sparse systems.
* ``power``: simplest low-cost estimate.

Example
-------

.. code-block:: python

   from sparse_kappa import cond_estimate

   cond = cond_estimate(A, norm=2, method='golub-kahan', max_iter=40)
   print(cond)
