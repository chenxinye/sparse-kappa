Sparse Kappa Documentation
===========================

**GPU-Accelerated Sparse Matrix Condition Number Estimation**

Sparse Kappa is a high-performance library for estimating condition numbers of sparse matrices on NVIDIA GPUs using PyTorch.

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/inEXASCALE/sparse-kappa/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/CUDA-11.0+-orange.svg
   :target: https://developer.nvidia.com/cuda-toolkit
   :alt: CUDA

Features
--------

* **⚡ GPU-Accelerated**: All computations run on NVIDIA GPUs via PyTorch
* **🎯 Multiple Norms**: Support for 1-norm and 2-norm condition numbers
* **🧮 Rich Algorithms**: 11+ methods including Hager-Higham, SVDS, Lanczos
* **🤖 GNN Prediction**: Train direct condition-number or inverse-norm predictors
* **🔧 Flexible Solvers**: LU, LSMR, CG, GMRES with smart caching
* **📊 10-100x Faster**: Than dense methods for sparse matrices
* **💾 Memory Efficient**: Designed for large sparse systems

Quick Start
-----------

.. code-block:: python

   from sparse_kappa.backend import sparse as sp
   from sparse_kappa import cond_estimate

   # Create sparse matrix
   A = sp.random(10000, 10000, density=0.01, format='csr')

   # Estimate condition number
   cond = cond_estimate(A)
   print(f"κ(A) = {cond:.2e}")

   # Use specific method with LU solver (fastest!)
   cond = cond_estimate(A, norm=1, method='hager-higham', solver='lu')

Performance
-----------

**Matrix: 3000×3000, density=0.005**

+---------------------------+----------+---------------+
| Method                    | Time     | vs PyTorch    |
+===========================+==========+===============+
| PyTorch (dense)           | 2.566s   | 1.0x          |
+---------------------------+----------+---------------+
| **Sparse SVDS**           | **0.083s**| **30.8x**    |
+---------------------------+----------+---------------+
| **Sparse Hager-Higham**   | **0.038s**| **67.5x**    |
+---------------------------+----------+---------------+

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   user_guide
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/main
   api/solvers
   api/norm1
   api/norm2
   api/gnn

.. toctree::
   :maxdepth: 2
   :caption: Advanced

   algorithms
   performance
   troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
