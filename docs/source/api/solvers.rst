Solver API
==========

Supported solver families
-------------------------

The solver subsystem is used mainly by 1-norm estimators:

* ``LUSolver``: cached LU-style solves for repeated right-hand sides.
* ``LSMRSolver``: iterative least-squares minimal residual method.
* ``CGSolver``: conjugate gradient for SPD-like systems.
* ``BiCGSTABSolver``: stabilized BiCG for non-symmetric systems.
* ``GMRESSolver``: robust Krylov fallback for difficult non-symmetric systems.
* ``DirectSolver`` and ``AutoSolver``: practical wrappers for direct/automatic selection.

Factory function
----------------

Use ``create_solver`` for method-driven selection.

.. code-block:: python

   from sparse_kappa import create_solver

   solver = create_solver(A, 'lu')
   x = solver.solve(b)
   print(solver.info())

   iterative = create_solver(A, 'lsmr', atol=1e-4, maxiter=80)
   x2 = iterative.solve(b)
