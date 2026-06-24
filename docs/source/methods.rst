Methods and Algorithms
======================

This page provides mathematical details and algorithmic descriptions of the methods implemented in sparse-kappa.

1-Norm Methods
--------------

Hager-Higham Algorithm
^^^^^^^^^^^^^^^^^^^^^^

**Mathematical Background**

The 1-norm of a matrix :math:`A \in \mathbb{R}^{n \times n}` is defined as:

.. math::

   \|A\|_1 = \max_{1 \leq j \leq n} \sum_{i=1}^n |a_{ij}|

The 1-norm condition number is:

.. math::

   \kappa_1(A) = \|A\|_1 \cdot \|A^{-1}\|_1

**Algorithm**

The Hager-Higham algorithm estimates :math:`\|A^{-1}\|_1` via iterative refinement:

1. Start with initial vector :math:`x = \frac{1}{n} \mathbf{1}`
2. For :math:`k = 1, 2, \ldots` until convergence:
   
   a. Solve :math:`A^T z = \text{sign}(x)`
   b. Find :math:`j = \arg\max_i |z_i|`
   c. If converged, stop
   d. Solve :math:`A x = e_j` where :math:`e_j` is the :math:`j`-th unit vector
   e. Compute :math:`\|x\|_1`

3. Return :math:`\|A^{-1}\|_1 \approx \|x\|_1`

**Complexity**: :math:`O(k \cdot \text{nnz}(A))` where :math:`k` is the number of iterations (typically 3-5)

**References**:

* Hager, W. W. (1984). "Condition estimates." SIAM J. Sci. Stat. Comput., 5(2), 311-316.
* Higham, N. J., & Tisseur, F. (2000). "A block algorithm for matrix 1-norm estimation." SIAM J. Matrix Anal. Appl., 21(4), 1185-1201.

2-Norm Methods
--------------

Power Method
^^^^^^^^^^^^

**Mathematical Background**

The power method finds the largest singular value :math:`\sigma_{\max}(A)` by iterating:

.. math::

   v_{k+1} = \frac{A^T A v_k}{\|A^T A v_k\|}

For the smallest singular value, use inverse iteration on :math:`A^T A`.

**Algorithm**

1. **For** :math:`\sigma_{\max}`:
   
   a. Initialize random :math:`v_0`, normalize
   b. Iterate :math:`v_{k+1} = \frac{A^T A v_k}{\|A^T A v_k\|}`
   c. Rayleigh quotient: :math:`\sigma^2 = v_k^T (A^T A) v_k`

2. **For** :math:`\sigma_{\min}`:
   
   Use shifted inverse iteration

3. **Return** :math:`\kappa_2(A) = \frac{\sigma_{\max}}{\sigma_{\min}}`

**Complexity**: :math:`O(k \cdot \text{nnz}(A))` per iteration

**Convergence**: Linear, rate depends on ratio :math:`\frac{\sigma_2}{\sigma_1}`

Lanczos Method
^^^^^^^^^^^^^^

**Mathematical Background**

The Lanczos algorithm tridiagonalizes a symmetric matrix :math:`B = A^T A`:

.. math::

   Q^T B Q = T

where :math:`T` is tridiagonal and :math:`Q` is orthogonal.

Eigenvalues of :math:`T` approximate eigenvalues of :math:`B`, and :math:`\sigma(A) = \sqrt{\lambda(A^T A)}`.

**Algorithm**

1. Initialize :math:`q_1` random, :math:`\beta_0 = 0`, :math:`q_0 = 0`
2. For :math:`j = 1, \ldots, m`:
   
   a. :math:`v = A^T (A q_j)`
   b. :math:`\alpha_j = q_j^T v`
   c. :math:`v = v - \alpha_j q_j - \beta_{j-1} q_{j-1}`
   d. :math:`\beta_j = \|v\|`
   e. :math:`q_{j+1} = v / \beta_j`

3. Form tridiagonal matrix :math:`T` with diagonals :math:`\alpha` and off-diagonals :math:`\beta`
4. Compute eigenvalues of :math:`T`
5. Return :math:`\kappa_2(A) = \sqrt{\frac{\lambda_{\max}}{\lambda_{\min}}}`

**Complexity**: :math:`O(m^2 \cdot \text{nnz}(A))` for :math:`m` iterations

**Advantages**:

* Fast convergence for well-separated eigenvalues
* Numerically stable with reorthogonalization

**References**:

* Lanczos, C. (1950). "An iteration method for the solution of the eigenvalue problem."
* Golub, G. H., & Van Loan, C. F. (2013). "Matrix Computations" (4th ed.), Chapter 10.

Arnoldi Method
^^^^^^^^^^^^^^

**Mathematical Background**

The Arnoldi iteration is a generalization of Lanczos for non-symmetric matrices. It builds an orthonormal basis for the Krylov subspace:

.. math::

   \mathcal{K}_m(A^T A, v) = \text{span}\{v, A^T A v, (A^T A)^2 v, \ldots, (A^T A)^{m-1} v\}

**Algorithm**

1. Initialize :math:`q_1 = v / \|v\|`
2. For :math:`j = 1, \ldots, m`:
   
   a. :math:`v = A^T (A q_j)`
   b. For :math:`i = 1, \ldots, j`:
      
      * :math:`h_{ij} = q_i^T v`
      * :math:`v = v - h_{ij} q_i`
   
   c. :math:`h_{j+1,j} = \|v\|`
   d. :math:`q_{j+1} = v / h_{j+1,j}`

3. Compute eigenvalues of Hessenberg matrix :math:`H = [h_{ij}]`
4. Return condition number from extreme eigenvalues

**Complexity**: :math:`O(m^2 n + m \cdot \text{nnz}(A))` for :math:`m` iterations

**References**:

* Arnoldi, W. E. (1951). "The principle of minimized iterations."
* Saad, Y. (2011). "Numerical Methods for Large Eigenvalue Problems" (2nd ed.).

Golub-Kahan Bidiagonalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Mathematical Background**

Instead of forming :math:`A^T A`, the Golub-Kahan algorithm directly bidiagonalizes :math:`A`:

.. math::

   U^T A V = B

where :math:`B` is bidiagonal and :math:`U, V` are orthogonal.

Singular values of :math:`B` equal singular values of :math:`A` (numerically more stable).

**Algorithm**

1. Initialize :math:`u_1 = 0`, :math:`\beta_1 v_1 = b` (random :math:`b`)
2. For :math:`j = 1, \ldots, m`:
   
   a. :math:`u_j = A v_j - \beta_j u_{j-1}`
   b. :math:`\alpha_j = \|u_j\|`, :math:`u_j = u_j / \alpha_j`
   c. :math:`v_{j+1} = A^T u_j - \alpha_j v_j`
   d. :math:`\beta_{j+1} = \|v_{j+1}\|`, :math:`v_{j+1} = v_{j+1} / \beta_{j+1}`

3. Form bidiagonal matrix :math:`B` with diagonals :math:`\alpha` and superdiagonals :math:`\beta`
4. Compute singular values of :math:`B` (via QR iteration)
5. Return :math:`\kappa_2(A) = \frac{\sigma_{\max}}{\sigma_{\min}}`

**Complexity**: :math:`O(m \cdot \text{nnz}(A))` for :math:`m` iterations

**Advantages**:

* More numerically stable than forming :math:`A^T A`
* Direct computation of singular values
* Better for ill-conditioned matrices

**References**:

* Golub, G. H., & Kahan, W. (1965). "Calculating the singular values and pseudo-inverse of a matrix."
* Golub, G. H., & Van Loan, C. F. (2013). "Matrix Computations" (4th ed.), Section 8.6.

PyTorch Wrapper Methods
--------------------

SVDS
^^^^

Wraps PyTorch's ``sparse_kappa.backend.sparse.linalg.svds``, which implements:

* Implicitly restarted Lanczos method
* Computes :math:`k` largest or smallest singular values
* Most accurate method for moderate-sized problems

**Usage**: Best for matrices with :math:`n < 10{,}000` when high accuracy is required.

EIGSH
^^^^^

Wraps PyTorch's ``sparse_kappa.backend.sparse.linalg.eigsh`` for symmetric matrices:

* Thick-restart Lanczos method (ARPACK-style)
* Computes :math:`k` largest/smallest eigenvalues
* Optimized for Hermitian/symmetric matrices

**Usage**: Symmetric or Hermitian matrices, faster than general methods.

LOBPCG
^^^^^^

Locally Optimal Block Preconditioned Conjugate Gradient:

* Block algorithm: computes multiple eigenvalues simultaneously
* Can incorporate preconditioners
* Efficient for very large matrices

**Algorithm**:

1. Start with block of random vectors :math:`X`
2. Iterate:
   
   a. Compute :math:`W = A X`
   b. Rayleigh-Ritz procedure on subspace
   c. Update :math:`X` with refined eigenvectors

**Usage**: Large matrices (:math:`n > 50{,}000`), especially with good preconditioners.

**References**:

* Knyazev, A. V. (2001). "Toward the optimal preconditioned eigensolver."

Algorithm Comparison
--------------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 20 35

   * - Method
     - Complexity
     - Memory
     - Best For
     - Notes
   * - Hager-Higham
     - :math:`O(k \cdot \text{nnz})`
     - Low
     - 1-norm estimation
     - Fast, 3-5 iterations
   * - Power
     - :math:`O(k \cdot \text{nnz})`
     - Low
     - Quick estimates
     - Simple, robust
   * - Lanczos
     - :math:`O(m^2 \cdot \text{nnz})`
     - Medium
     - General purpose
     - Good balance
   * - Arnoldi
     - :math:`O(m^2 n)`
     - High
     - Non-symmetric
     - More memory
   * - Golub-Kahan
     - :math:`O(m \cdot \text{nnz})`
     - Low
     - Ill-conditioned
     - Most stable
   * - SVDS
     - :math:`O(k \cdot \text{nnz})`
     - Medium
     - High accuracy
     - Most accurate
   * - EIGSH
     - :math:`O(k \cdot \text{nnz})`
     - Medium
     - Symmetric matrices
     - Optimized
   * - LOBPCG
     - :math:`O(k \cdot \text{nnz})`
     - Medium
     - Very large
     - Preconditionable

Convergence Theory
------------------

For power iteration methods, convergence rate depends on the eigenvalue gap:

.. math::

   \|v_k - v_*\| \leq C \left(\frac{|\lambda_2|}{|\lambda_1|}\right)^k

For Krylov methods (Lanczos, Arnoldi), convergence is faster and depends on polynomial approximation properties.

Numerical Stability
-------------------

**Loss of Orthogonality**

In Lanczos/Arnoldi, vectors can lose orthogonality due to rounding errors. Solutions:

1. Full reorthogonalization (expensive)
2. Selective reorthogonalization (adaptive)
3. Partial reorthogonalization (sparse-kappa default)

**Condition Number Effects**

For :math:`\kappa(A) \approx 10^d`, expect :math:`d` digits of accuracy loss in IEEE double precision.

Practical Recommendations
--------------------------

1. **Start with ``auto``** method selection
2. **For symmetric matrices**: Use ``eigsh``
3. **For high accuracy**: Use ``svds`` (if :math:`n < 10{,}000`)
4. **For large sparse**: Use ``golub-kahan`` or ``lobpcg``
5. **For ill-conditioned**: Use ``golub-kahan`` (more stable)
6. **For quick estimates**: Use ``power`` method