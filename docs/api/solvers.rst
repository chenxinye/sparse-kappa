Solver API
==========

.. automodule:: sparse_kappa.solvers
   :members:
   :undoc-members:
   :show-inheritance:

Solver Classes
--------------

LUSolver
~~~~~~~~

.. autoclass:: sparse_kappa.solvers.LUSolver
   :members:
   :undoc-members:
   :show-inheritance:

   **Example**:

   .. code-block:: python

      from sparse_kappa.solvers import LUSolver

      solver = LUSolver(A)
      x = solver.solve(b)
      print(solver.info())

LSMRSolver
~~~~~~~~~~

.. autoclass:: sparse_kappa.solvers.LSMRSolver
   :members:
   :undoc-members:
   :show-inheritance:

CGSolver
~~~~~~~~

.. autoclass:: sparse_kappa.solvers.CGSolver
   :members:
   :undoc-members:
   :show-inheritance:

DirectSolver
~~~~~~~~~~~~

.. autoclass:: sparse_kappa.solvers.DirectSolver
   :members:
   :undoc-members:
   :show-inheritance:

GMRESSolver
~~~~~~~~~~~

.. autoclass:: sparse_kappa.solvers.GMRESSolver
   :members:
   :undoc-members:
   :show-inheritance:

AutoSolver
~~~~~~~~~~

.. autoclass:: sparse_kappa.solvers.AutoSolver
   :members:
   :undoc-members:
   :show-inheritance:

Factory Function
----------------

create_solver
~~~~~~~~~~~~~

.. autofunction:: sparse_kappa.solvers.create_solver

   **Example**:

   .. code-block:: python

      from sparse_kappa import create_solver

      # LU solver with caching
      solver = create_solver(A, 'lu')
      x1 = solver.solve(b1)
      x2 = solver.solve(b2)  # Reuses LU factors

      # LSMR with custom parameters
      solver = create_solver(A, 'lsmr', atol=1e-4, maxiter=50)