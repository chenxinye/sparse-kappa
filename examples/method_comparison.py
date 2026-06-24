"""
Comprehensive comparison of all methods.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from sparse_kappa import cond_estimate
import time


def compare_all_methods():
    """Compare all available methods on same matrix."""
    print("=" * 80)
    print("Method Comparison")
    print("=" * 80)
    
    # Create test matrix
    n = 2000
    density = 0.005
    A = sp.random(n, n, density=density, format='csr')
    
    print(f"Matrix size: {n} x {n}")
    print(f"Density: {density:.4f}")
    print(f"NNZ: {A.nnz}")
    print()
    
    # Methods to test
    methods_2norm = [
        'power',
        'lanczos',
        'arnoldi',
        'golub-kahan',
        'svds',
        'lobpcg',
    ]
    
    results = {}
    
    # Test each method
    for method in methods_2norm:
        print(f"\nTesting {method}...")
        try:
            start = time.time()
            result = cond_estimate(A, norm=2, method=method, 
                                 max_iter=100, tol=1e-6, verbose=False)
            elapsed = time.time() - start
            
            results[method] = {
                'cond': result['condition_number'],
                'time': elapsed,
                'iters': result['iterations'],
                'converged': result['converged'],
            }
            
            print(f"  κ₂ = {results[method]['cond']:.4e}")
            print(f"  Time = {results[method]['time']:.3f}s")
            print(f"  Iterations = {results[method]['iters']}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[method] = None
    
    # Summary table
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Method':<15} {'Cond Number':>15} {'Time (s)':>12} {'Iters':>8} {'Conv':>6}")
    print("-" * 80)
    
    for method, res in results.items():
        if res:
            print(f"{method:<15} {res['cond']:15.4e} {res['time']:12.3f} "
                  f"{res['iters']:8d} {str(res['converged']):>6}")
        else:
            print(f"{method:<15} {'FAILED':>15}")


def test_1norm():
    """Test 1-norm methods."""
    print("\n\n" + "=" * 80)
    print("1-Norm Methods")
    print("=" * 80)
    
    n = 1500
    A = sp.random(n, n, density=0.01, format='csr')
    
    print(f"Matrix size: {n} x {n}\n")
    
    start = time.time()
    result = cond_estimate(A, norm=1, method='hager-higham', verbose=False)
    elapsed = time.time() - start
    
    print(f"Hager-Higham:")
    print(f"  κ₁ = {result['condition_number']:.4e}")
    print(f"  Time = {elapsed:.3f}s")
    print(f"  Iterations = {result['iterations']}")


if __name__ == '__main__':
    compare_all_methods()
    test_1norm()