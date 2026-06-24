"""
Performance benchmarking script.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from sparse_kappa import cond_estimate
import time
import json


def benchmark_scaling():
    """Benchmark performance vs matrix size."""
    print("=" * 80)
    print("Performance Scaling Benchmark")
    print("=" * 80)
    
    sizes = [500, 1000, 2000, 5000, 10000]
    density = 0.005
    method = 'golub-kahan'
    
    results = []
    
    for n in sizes:
        print(f"\nMatrix size: {n} x {n}")
        
        # Create matrix
        A = sp.random(n, n, density=density, format='csr')
        
        # Warm-up
        _ = cond_estimate(A, norm=2, method=method, max_iter=20)
        
        # Benchmark
        start = time.time()
        result = cond_estimate(A, norm=2, method=method, max_iter=50, verbose=False)
        elapsed = time.time() - start
        
        results.append({
            'size': n,
            'nnz': int(A.nnz),
            'time': elapsed,
            'cond': float(result['condition_number']),
        })
        
        print(f"  Time: {elapsed:.3f}s")
        print(f"  κ₂: {result['condition_number']:.4e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Size':>8} {'NNZ':>12} {'Time (s)':>12} {'κ₂':>15}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['size']:8d} {r['nnz']:12d} {r['time']:12.3f} {r['cond']:15.4e}")


def benchmark_sparsity():
    """Benchmark performance vs sparsity."""
    print("\n\n" + "=" * 80)
    print("Sparsity Benchmark")
    print("=" * 80)
    
    n = 2000
    densities = [0.001, 0.005, 0.01, 0.05, 0.1]
    method = 'svds'
    
    results = []
    
    for density in densities:
        print(f"\nDensity: {density:.3f}")
        
        # Create matrix
        A = sp.random(n, n, density=density, format='csr')
        
        # Benchmark
        start = time.time()
        result = cond_estimate(A, norm=2, method=method, max_iter=50, verbose=False)
        elapsed = time.time() - start
        
        results.append({
            'density': density,
            'nnz': int(A.nnz),
            'time': elapsed,
            'cond': float(result['condition_number']),
        })
        
        print(f"  NNZ: {A.nnz}")
        print(f"  Time: {elapsed:.3f}s")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Density':>10} {'NNZ':>12} {'Time (s)':>12} {'κ₂':>15}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['density']:10.4f} {r['nnz']:12d} {r['time']:12.3f} {r['cond']:15.4e}")


if __name__ == '__main__':
    benchmark_scaling()
    benchmark_sparsity()