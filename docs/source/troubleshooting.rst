Troubleshooting
===============

Import errors
-------------

**Problem:** missing runtime dependencies such as ``torch``.

**Fix:** install project dependencies first.

.. code-block:: bash

   pip install -e .

Convergence issues
------------------

**Problem:** iterative method does not converge or returns unstable values.

**Fixes:**

* Increase ``max_iter`` and tighten/relax ``tol`` as needed.
* Switch method (for example from ``power`` to ``svds`` or ``golub-kahan``).
* For 1-norm estimation, prefer ``solver='lu'`` when memory allows.

Very large condition numbers
----------------------------

A very large output can be expected for nearly singular matrices.
Validate with another method and inspect matrix rank/structure before treating it as an error.

GPU / device mismatch
---------------------

If tensors or sparse matrices are on different devices, move data to a consistent device before estimation.
