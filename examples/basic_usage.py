"""Basic usage examples for sparse-kappa."""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from sparse_kappa import cond_estimate


def example_1_simple():
    """Simple condition number estimation."""
    print("=" * 60)
    print("Example 1: Simple condition number estimation")
    print("=" * 60)
    
    # Create random sparse matrix
    n = 1000
    A = sp.random(n, n, density=0.01, format='csr')
    
    # Estimate condition number (auto method selection)
    cond = cond_estimate(A)
    print(f"\nCondition number: {cond:.2e}")


def example_2_compare_methods():
    """Compare different methods."""
    print("\n" + "=" * 60)
    print("Example 2: Compare different methods")
    print("=" * 60)
    
    # Create test matrix
    n = 500
    A = sp.random(n, n, density=0.02, format='csr')
    
    methods = ['power', 'lanczos', 'svds', 'golub-kahan']
    
    for method in methods:
        result = cond_estimate(A, norm=2, method=method, verbose=True)
        print(f"\n{method}: κ = {result['condition_number']:.4e}, "
              f"iters = {result['iterations']}")


def example_3_1norm():
    """1-norm condition number estimation."""
    print("\n" + "=" * 60)
    print("Example 3: 1-norm condition number")
    print("=" * 60)
    
    # Create matrix
    n = 800
    A = sp.random(n, n, density=0.015, format='csr')
    
    # Estimate 1-norm condition number
    result = cond_estimate(A, norm=1, method='hager-higham', verbose=True)
    
    print(f"\nκ₁(A) = {result['condition_number']:.4e}")
    print(f"||A||₁ = {result['norm_A']:.4e}")
    print(f"||A⁻¹||₁ = {result['norm_Ainv']:.4e}")


def example_4_ill_conditioned():
    """Test on ill-conditioned matrix."""
    print("\n" + "=" * 60)
    print("Example 4: Ill-conditioned matrix")
    print("=" * 60)
    
    # Create ill-conditioned matrix (diagonal with large condition number)
    n = 500
    diag = cp.logspace(0, 6, n)  # Condition number ~ 10^6
    A = sp.diags(diag, format='csr')
    
    print(f"True condition number: ~{diag.max() / diag.min():.2e}")
    
    # Estimate with different methods
    for method in ['svds', 'power', 'lanczos']:
        cond = cond_estimate(A, norm=2, method=method)
        print(f"{method:12s}: {cond:.2e}")


def example_5_symmetric():
    """Test on symmetric matrix."""
    print("\n" + "=" * 60)
    print("Example 5: Symmetric matrix")
    print("=" * 60)
    
    # Create symmetric matrix
    n = 600
    A = sp.random(n, n, density=0.01, format='csr')
    A = (A + A.T) / 2  # Make symmetric
    
    # Use eigsh (optimized for symmetric matrices)
    result = cond_estimate(A, norm=2, method='eigsh', verbose=True)
    
    print(f"\nκ₂(A) = {result['condition_number']:.4e}")


if __name__ == '__main__':
    example_1_simple()
    example_2_compare_methods()
    example_3_1norm()
    example_4_ill_conditioned()
    example_5_symmetric()
