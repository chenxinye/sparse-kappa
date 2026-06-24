"""
Sparse linear system solvers with caching support.

Provides unified interface for different solver types with optional factorization caching.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from sparse_kappa.backend.sparse import linalg as splinalg
from typing import Dict, Any
import warnings
import torch


class SparseSolver:
    """
    Base class for sparse linear system solvers.
    
    Solves Ax = b for various right-hand sides b.
    """
    
    def __init__(self, A: sp.spmatrix, **kwargs):
        self.A = A
        self.n = A.shape[0]
        self.kwargs = kwargs
        self.solve_count = 0
    
    def solve(self, b: cp.ndarray) -> cp.ndarray:
        """Solve Ax = b"""
        raise NotImplementedError
    
    def info(self) -> Dict[str, Any]:
        """Return solver information"""
        return {
            'solver_type': self.__class__.__name__,
            'matrix_size': self.n,
            'solve_count': self.solve_count,
        }


class LUSolver(SparseSolver):
    """
    LU factorization solver with caching.
    
    Uses dense LU decomposition for small-medium matrices.
    PyTorch does not provide a native sparse LU factorization here,
    so we use cached dense LU for matrices that fit in memory.
    
    Best for: Small to medium matrices (n < 5000), multiple solves
    """
    
    def __init__(self, A: sp.spmatrix, use_dense_threshold: float = 0.001, **kwargs):
        super().__init__(A, **kwargs)
        
        n = A.shape[0]
        density = A.nnz / (n * n)
        
        # Decide whether to use dense LU
        # Use dense if: small matrix OR not too sparse
        self.use_dense = (n < 5000) or (density > use_dense_threshold)
        
        try:
            if self.use_dense:
                # Convert to dense and compute LU
                A_dense = A.toarray()
                
                try:
                    self.lu, self.piv = torch.linalg.lu_factor(A_dense)
                    self.has_lu_factor = True
                except (RuntimeError, AttributeError):
                    # Fallback: just store dense matrix and use solve
                    self.A_dense = A_dense
                    self.has_lu_factor = False
                
                self.factorized = True
                self.method = 'dense_lu'
                
            else:
                # Try sparse LU (may not be available or may be slow)
                try:
                    A_csc = A.tocsc()
                    self.lu_sparse = splinalg.splu(A_csc)
                    self.factorized = True
                    self.method = 'sparse_lu'
                except:
                    raise RuntimeError("Sparse LU not available")
                    
        except Exception as e:
            if self.use_dense:
                warnings.warn(f"Dense LU failed: {e}. Falling back to iterative solver.")
            else:
                warnings.warn(f"Sparse LU failed: {e}. Falling back to iterative solver.")
            self.factorized = False
            self.backup_solver = LSMRSolver(A, **kwargs)
            self.method = 'lsmr_backup'
    
    def solve(self, b: cp.ndarray) -> cp.ndarray:
        self.solve_count += 1
        
        if not self.factorized:
            return self.backup_solver.solve(b)
        
        if self.method == 'dense_lu':
            if self.has_lu_factor:
                rhs_was_vector = b.ndim == 1
                rhs = b.unsqueeze(1) if rhs_was_vector else b
                x = torch.linalg.lu_solve(self.lu, self.piv, rhs)
                return x.squeeze(1) if rhs_was_vector else x
            else:
                return cp.linalg.solve(self.A_dense, b)
        
        elif self.method == 'sparse_lu':
            return self.lu_sparse.solve(b)
        
        else:
            return self.backup_solver.solve(b)
    
    def info(self) -> Dict[str, Any]:
        info = super().info()
        info.update({
            'factorized': self.factorized,
            'method': getattr(self, 'method', 'unknown'),
            'use_dense': self.use_dense,
        })
        return info


class LSMRSolver(SparseSolver):
    """
    LSMR iterative solver.
    
    Least-squares minimal residual method.
    Good general-purpose iterative solver.
    
    Best for: Large sparse matrices, approximate solves
    """
    
    def __init__(
        self, 
        A: sp.spmatrix, 
        atol: float = 1e-3,
        btol: float = 1e-3,
        maxiter: int = 50,
        **kwargs
    ):
        super().__init__(A, **kwargs)
        self.atol = atol
        self.btol = btol
        self.maxiter = maxiter
    
    def solve(self, b: cp.ndarray) -> cp.ndarray:
        self.solve_count += 1
        result = splinalg.lsmr(
            self.A, b,
            atol=self.atol,
            btol=self.btol,
            maxiter=self.maxiter
        )
        return result[0]
    
    def info(self) -> Dict[str, Any]:
        info = super().info()
        info.update({
            'atol': self.atol,
            'btol': self.btol,
            'maxiter': self.maxiter,
        })
        return info


class CGSolver(SparseSolver):
    """
    Conjugate Gradient solver.
    
    Best for symmetric positive definite matrices.
    
    Best for: SPD matrices, faster than LSMR for SPD
    """
    
    def __init__(
        self,
        A: sp.spmatrix,
        atol: float = 1e-3,
        maxiter: int = 50,  
        **kwargs
    ):
        super().__init__(A, **kwargs)
        self.atol = atol
        self.maxiter = maxiter
    
    def solve(self, b: cp.ndarray) -> cp.ndarray:
        self.solve_count += 1
        x, info = splinalg.cg(
            self.A, b,
            atol=self.atol,
            maxiter=self.maxiter
        )
        if info > 0:
            if info == self.maxiter:
                warnings.warn(
                    f"CG reached max iterations ({self.maxiter}). "
                    f"Consider increasing maxiter or using a different solver.",
                    category=UserWarning
                )
            else:
                warnings.warn(f"CG did not converge: info={info}")
        return x


class BiCGSTABSolver(SparseSolver):
    """
    BiCGSTAB (Biconjugate Gradient Stabilized) solver.
    
    Stabilized version of BiCG, good for non-symmetric matrices.
    Often faster and more stable than GMRES for moderately ill-conditioned systems.
    
    Best for: Non-symmetric matrices, faster convergence than GMRES
    
    Reference
    ---------
    Van der Vorst, H. A. (1992). "Bi-CGSTAB: A fast and smoothly converging 
    variant of Bi-CG for the solution of nonsymmetric linear systems."
    SIAM J. Sci. Stat. Comput., 13(2), 631-644.
    """
    
    def __init__(
        self,
        A: sp.spmatrix,
        atol: float = 1e-3,
        maxiter: int = 50,
        **kwargs
    ):
        super().__init__(A, **kwargs)
        self.atol = atol
        self.maxiter = maxiter
    
    def solve(self, b: cp.ndarray) -> cp.ndarray:
        self.solve_count += 1
        
        try:
            # BiCGSTAB is available in PyTorch backend sparse.linalg
            x, info = splinalg.bicgstab(
                self.A, b,
                atol=self.atol,
                maxiter=self.maxiter
            )
        except:
            warnings.warn("BiCGSTAB not available. Falling back to our implementation.")
            from .notimplement import bicgstab
            
            x, info = bicgstab(
                self.A, b,
                atol=self.atol,
                maxiter=self.maxiter
            )
        
        if info > 0:
            warnings.warn(f"BiCGSTAB did not converge after {info} iterations")
        elif info < 0:
            warnings.warn(f"BiCGSTAB illegal input or breakdown (info={info})")
        
        return x
    
    def info(self) -> Dict[str, Any]:
        info = super().info()
        info.update({
            'atol': self.atol,
            'maxiter': self.maxiter,
        })
        return info


class DirectSolver(SparseSolver):
    """
    Direct sparse solver (no caching).
    
    Uses spsolve for each solve. Good for one-time solves.
    
    Best for: Small matrices, single solve
    """
    
    def __init__(self, A: sp.spmatrix, **kwargs):
        super().__init__(A, **kwargs)
        # 预先转换为CSR格式以避免警告
        if not sp.isspmatrix_csr(A):
            self.A_csr = A.tocsr()
        else:
            self.A_csr = A
    
    def solve(self, b: cp.ndarray) -> cp.ndarray:
        self.solve_count += 1
        return splinalg.spsolve(self.A_csr, b)


class GMRESSolver(SparseSolver):
    """
    GMRES iterative solver.
    
    Generalized Minimal Residual method.
    Good for non-symmetric systems.
    
    Best for: Non-symmetric matrices, when BiCGSTAB fails
    """
    
    def __init__(
        self,
        A: sp.spmatrix,
        atol: float = 1e-3,
        maxiter: int = 50,  
        restart: int = 30,  
        **kwargs
    ):
        super().__init__(A, **kwargs)
        self.atol = atol
        self.maxiter = maxiter
        self.restart = restart
    
    def solve(self, b: cp.ndarray) -> cp.ndarray:
        self.solve_count += 1
        x, info = splinalg.gmres(
            self.A, b,
            atol=self.atol,
            maxiter=self.maxiter,
            restart=self.restart
        )
        if info > 0:
            if info == self.maxiter:
                warnings.warn(
                    f"GMRES reached max iterations ({self.maxiter}). "
                    f"Consider increasing maxiter or restart.",
                    category=UserWarning
                )
            else:
                warnings.warn(f"GMRES did not converge: info={info}")
        return x
    
    def info(self) -> Dict[str, Any]:
        info = super().info()
        info.update({
            'atol': self.atol,
            'maxiter': self.maxiter,
            'restart': self.restart,
        })
        return info


class AutoSolver(SparseSolver):
    """
    Automatically select best solver based on matrix properties.
    
    Selection criteria:
    - Small matrices (n < 5000): LU with caching (dense)
    - Medium matrices: LSMR or BiCGSTAB
    - Large matrices: LSMR with relaxed tolerance
    """
    
    def __init__(self, A: sp.spmatrix, **kwargs):
        super().__init__(A, **kwargs)
        
        n = A.shape[0]
        nnz = A.nnz
        density = nnz / (n * n)
        
        # 使用更可靠的对称性检查
        is_symmetric = self._is_symmetric_fast(A)
        
        # Select solver
        if n < 5000:
            # Small matrices: use LU
            self.solver = LUSolver(A, **kwargs)
            self.solver_name = 'LU (dense)'
        elif is_symmetric and n < 20000:
            # Symmetric medium: use CG with more iterations
            self.solver = CGSolver(A, atol=1e-3, maxiter=50)
            self.solver_name = 'CG'
        elif n < 20000:
            # Non-symmetric medium: use BiCGSTAB
            self.solver = BiCGSTABSolver(A, atol=1e-3, maxiter=50)
            self.solver_name = 'BiCGSTAB'
        else:
            # Large matrices: use LSMR with relaxed tolerance
            self.solver = LSMRSolver(A, atol=1e-2, btol=1e-2, maxiter=20)
            self.solver_name = 'LSMR (relaxed)'
    
    def _is_symmetric_fast(self, A: sp.spmatrix) -> bool:
        """Fast symmetry check with sampling for large matrices."""
        if A.shape[0] != A.shape[1]:
            return False
        
        # Quick check: compare nnz
        if A.nnz != A.T.nnz:
            return False
        
        n = A.shape[0]
        
        # For large matrices, sample
        if n > 500:
            num_samples = min(20, n)
            indices = cp.random.choice(n, size=num_samples, replace=False)
            
            for i in indices:
                for j in indices:
                    if i != j:
                        val_ij = float(A[i, j])
                        val_ji = float(A[j, i])
                        if abs(val_ij - val_ji) > 1e-6 * (abs(val_ij) + abs(val_ji) + 1e-10):
                            return False
            return True
        else:
            # Small matrix: full check
            try:
                diff = A - A.T
                return splinalg.norm(diff) < 1e-6 * splinalg.norm(A)
            except:
                return False
    
    def solve(self, b: cp.ndarray) -> cp.ndarray:
        self.solve_count += 1
        return self.solver.solve(b)
    
    def info(self) -> Dict[str, Any]:
        info = super().info()
        info['selected_solver'] = self.solver_name
        info['inner_solver_info'] = self.solver.info()
        return info


# Solver factory
SOLVER_MAP = {
    'lu': LUSolver,
    'lsmr': LSMRSolver,
    'cg': CGSolver,
    'bicgstab': BiCGSTABSolver,
    'direct': DirectSolver,
    'gmres': GMRESSolver,
    'auto': AutoSolver,
}


def create_solver(
    A: sp.spmatrix,
    solver_type: str = 'auto',
    **kwargs
) -> SparseSolver:
    """
    Create a solver for matrix A.
    
    Parameters
    ----------
    A : sparse matrix
        Input matrix
    solver_type : str, default='auto'
        Solver type: 'lu', 'lsmr', 'cg', 'bicgstab', 'direct', 'gmres', 'auto'
    **kwargs : dict
        Additional solver parameters
    
    Returns
    -------
    solver : SparseSolver
        Configured solver instance
    
    Examples
    --------
    >>> from sparse_kappa import create_solver
    >>> from sparse_kappa.backend import sparse as sp
    >>> 
    >>> A = sp.random(1000, 1000, density=0.01, format='csr')
    >>> 
    >>> # BiCGSTAB for non-symmetric matrices
    >>> solver = create_solver(A, 'bicgstab', atol=1e-4, maxiter=30)
    >>> x = solver.solve(b)
    >>> 
    >>> # Auto-selection
    >>> solver = create_solver(A, 'auto')
    """
    if solver_type not in SOLVER_MAP:
        raise ValueError(
            f"Unknown solver type: {solver_type}. "
            f"Available: {list(SOLVER_MAP.keys())}"
        )
    
    return SOLVER_MAP[solver_type](A, **kwargs)
