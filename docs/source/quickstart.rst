Quick Start Guide
=================

This guide will get you started with Sparse Kappa in 5 minutes.

Basic Usage
-----------

Estimate Condition Number
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sparse_kappa.backend import sparse as sp
   from sparse_kappa import cond_estimate
   
   # Create sparse matrix on GPU
   A = sp.random(1000, 1000, density=0.01, format='csr')
   
   # Estimate condition number (automatic method)
   cond = cond_estimate(A)
   print(f"κ(A) = {cond:.4e}")

Choose Specific Method
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1-norm with Hager-Higham algorithm
   cond_1 = cond_estimate(A, norm=1, method='hager-higham')
   
   # 2-norm with SVDS
   cond_2 = cond_estimate(A, norm=2, method='svds')
   
   print(f"κ₁(A) = {cond_1:.4e}")
   print(f"κ₂(A) = {cond_2:.4e}")

Get Detailed Results
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Return full dictionary of results
   result = cond_estimate(A, norm=2, method='svds', return_dict=True)
   
   print(f"Method: {result['method']}")
   print(f"Condition number: {result['condition_number']:.4e}")
   print(f"Iterations: {result['iterations']}")
   print(f"σ_max: {result['sigma_max']:.4e}")
   print(f"σ_min: {result['sigma_min']:.4e}")

Use Fast LU Solver
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Fastest for 1-norm (10-20x speedup)
   cond = cond_estimate(A, norm=1, method='hager-higham', solver='lu')

Common Workflows
----------------

Check Matrix Conditioning
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sparse_kappa.backend import sparse as sp
   from sparse_kappa import cond_estimate
   
   A = sp.random(2000, 2000, density=0.005, format='csr')
   cond = cond_estimate(A)
   
   if cond < 100:
       print("Matrix is well-conditioned ✓")
   elif cond < 1000:
       print("Matrix is moderately conditioned")
   else:
       print("Matrix is ill-conditioned ⚠")

Compare Methods
~~~~~~~~~~~~~~~

.. code-block:: python

   methods_1norm = ['hager-higham', 'power', 'oettli-prager']
   methods_2norm = ['svds', 'lanczos', 'golub-kahan']
   
   print("1-norm methods:")
   for method in methods_1norm:
       result = cond_estimate(A, norm=1, method=method, 
                             solver='lu', return_dict=True)
       print(f"  {method:15s}: κ={result['condition_number']:.4e}, "
             f"iter={result['iterations']}")
   
   print("\n2-norm methods:")
   for method in methods_2norm:
       cond = cond_estimate(A, norm=2, method=method)
       print(f"  {method:15s}: κ={cond:.4e}")

Analyze Large Matrix
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Large sparse matrix
   A = sp.random(50000, 50000, density=0.0001, format='csr')
   
   # Use Golub-Kahan for speed
   import time
   start = time.time()
   cond = cond_estimate(A, norm=2, method='golub-kahan', 
                       num_values=6, max_iter=30)
   elapsed = time.time() - start
   
   print(f"Condition number: {cond:.4e}")
   print(f"Time: {elapsed:.2f}s")

Method Selection Guide
----------------------

When to Use Each Norm
~~~~~~~~~~~~~~~~~~~~~

**1-norm** (``norm=1``):

* When you need L₁ condition number
* Faster than 2-norm for sparse matrices
* Use with ``solver='lu'`` for best performance

**2-norm** (``norm=2``):

* Most common in numerical analysis
* Natural for symmetric matrices
* Better stability analysis

When to Use Each Method
~~~~~~~~~~~~~~~~~~~~~~~

**1-norm methods**:

.. code-block:: python

   # Hager-Higham: Industry standard, accurate
   cond_estimate(A, norm=1, method='hager-higham', solver='lu')
   
   # Power: Fast rough estimate
   cond_estimate(A, norm=1, method='power', solver='lu')
   
   # Oettli-Prager: Multiple sampling strategies
   cond_estimate(A, norm=1, method='oettli-prager', solver='lu', 
                variant='adaptive')

**2-norm methods**:

.. code-block:: python

   # SVDS: Most accurate (small-medium matrices)
   cond_estimate(A, norm=2, method='svds', num_values=10)
   
   # Lanczos: Good for symmetric
   cond_estimate(A, norm=2, method='lanczos', num_values=6)
   
   # Golub-Kahan: Fast for large matrices
   cond_estimate(A, norm=2, method='golub-kahan', num_values=6)

Solver Selection (1-norm only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # LU: Fastest (recommended)
   cond_estimate(A, norm=1, method='hager-higham', solver='lu')
   
   # LSMR: For very large matrices
   cond_estimate(A, norm=1, method='hager-higham', solver='lsmr',
                solver_kwargs={'atol': 1e-3, 'maxiter': 20})
   
   # CG: For symmetric positive definite
   A_spd = A @ A.T + sp.eye(A.shape[0]) * 10
   cond_estimate(A_spd, norm=1, method='hager-higham', solver='cg')

Performance Tips
----------------

1. **Use LU solver for 1-norm**

   .. code-block:: python
   
      # 10-20x faster
      cond_estimate(A, norm=1, method='hager-higham', solver='lu')

2. **Reduce iterations for quick estimates**

   .. code-block:: python
   
      cond_estimate(A, norm=2, method='lanczos', max_iter=10, num_values=3)

3. **Use auto-selection**

   .. code-block:: python
   
      # Library picks best method
      cond_estimate(A, method='auto')

4. **Warm up GPU first**

   .. code-block:: python
   
      # First call compiles kernels
      _ = cond_estimate(A, norm=2, method='svds')
      # Subsequent calls are fast
      cond = cond_estimate(A, norm=2, method='svds')

Next Steps
----------

* :doc:`user_guide` - Comprehensive documentation
* :doc:`api/main` - Full API reference
* :doc:`examples` - More examples
* :doc:`performance` - Performance optimization