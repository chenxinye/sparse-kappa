Performance guide
=================

Method selection
----------------

* Small/medium matrices: ``svds`` gives the most accurate result.
* Large sparse matrices: ``golub-kahan`` or ``lanczos`` is usually faster.
* 1-norm estimation: ``hager-higham`` with ``solver='lu'`` is typically the fastest stable option.

Practical tips
--------------

* Keep matrix format in CSR/CSC for sparse operations.
* Start with ``method='auto'`` to get a good baseline.
* Reduce ``max_iter`` when rough estimates are acceptable.
* Use ``return_dict=True`` to inspect convergence metadata and debug bottlenecks.

Benchmark template
------------------

.. code-block:: python

   import time

   candidates = ['svds', 'lanczos', 'golub-kahan']
   for name in candidates:
       t0 = time.time()
       cond = cond_estimate(A, norm=2, method=name)
       dt = time.time() - t0
       print(f"{name:12s}  cond={cond:.3e}  time={dt:.3f}s")
