Installation
============

This guide covers installation of Sparse Kappa and its dependencies.

Prerequisites
-------------

Required
~~~~~~~~

* Python 3.8 or later
* NVIDIA GPU with compute capability 6.0 or higher
* CUDA Toolkit 11.0 or later (11.x or 12.x)

Recommended
~~~~~~~~~~~

* conda or virtualenv for environment management
* 8GB+ GPU memory for large matrices

Check Your System
-----------------

CUDA Version
~~~~~~~~~~~~

Check your CUDA version:

.. code-block:: bash

   nvcc --version

If you don't have CUDA installed, download it from `NVIDIA's website <https://developer.nvidia.com/cuda-toolkit>`_.

GPU Information
~~~~~~~~~~~~~~~

Check your GPU:

.. code-block:: bash

   nvidia-smi

Installing PyTorch
---------------

PyTorch is the main dependency. Install the version matching your CUDA:

For CUDA 11.x
~~~~~~~~~~~~~

.. code-block:: bash

   pip install torch

For CUDA 12.x
~~~~~~~~~~~~~

.. code-block:: bash

   pip install torch

Verify Installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sparse_kappa.backend import torch_api as cp
   print(f"PyTorch version: {cp.__version__}")
   print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
   
   # Test GPU
   x = cp.array([1, 2, 3])
   print(f"GPU test: {cp.sum(x)}")

Installing Sparse Kappa
-----------------------

From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/inEXASCALE/sparse-kappa.git
   cd sparse-kappa
   pip install -e .

This installs in editable mode for development.

From PyPI (Coming Soon)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install sparse-kappa

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development with testing tools:

.. code-block:: bash

   git clone https://github.com/inEXASCALE/sparse-kappa.git
   cd sparse-kappa
   pip install -e ".[dev]"

This installs additional packages:

* pytest - for running tests
* pytest-cov - for coverage reports
* black - for code formatting
* flake8 - for linting

Verify Installation
-------------------

.. code-block:: python

   from sparse_kappa import cond_estimate
   from sparse_kappa.backend import sparse as sp
   
   # Create test matrix
   A = sp.random(100, 100, density=0.1, format='csr')
   
   # Estimate condition number
   cond = cond_estimate(A)
   print(f"Condition number: {cond:.2e}")
   
   # Check version
   import sparse_kappa
   print(f"Sparse Kappa version: {sparse_kappa.__version__}")

Common Issues
-------------

Issue: "No module named 'PyTorch'"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: Install PyTorch matching your CUDA version

.. code-block:: bash

   pip install torch

Issue: "CUDA driver version is insufficient"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: Update your NVIDIA driver

1. Check required driver version for your CUDA
2. Download from `NVIDIA Driver Downloads <https://www.nvidia.com/Download/index.aspx>`_
3. Install and reboot

Issue: ImportError with cuSOLVER
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: This is expected - the library handles this gracefully and falls back to alternative methods.

Issue: Out of memory
~~~~~~~~~~~~~~~~~~~~

**Solution**: 

* Use iterative solvers instead of LU
* Reduce matrix size for testing
* Use methods with lower memory footprint

.. code-block:: python

   # Instead of LU
   cond_estimate(A, norm=1, method='hager-higham', solver='lsmr')

Next Steps
----------

* :doc:`quickstart` - Learn basic usage
* :doc:`user_guide` - Comprehensive guide
* :doc:`examples` - Code examples