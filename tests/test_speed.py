"""
Simple benchmark comparing sparse methods vs PyTorch dense condition number.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
import torch
import scipy.sparse as scipy_sp
import numpy as np
import time
from sparse_kappa import cond_estimate


def gpu_time(func, warmup=3, repeat=5):
    """Measure GPU time with proper warm-up and synchronization."""
    # Warm-up
    for _ in range(warmup):
        _ = func()
        cp.cuda.Stream.null.synchronize()
        torch.cuda.synchronize()
    
    # Timing
    times = []
    result = None
    for _ in range(repeat):
        cp.cuda.Stream.null.synchronize()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = func()
        
        cp.cuda.Stream.null.synchronize()
        torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return sum(times) / len(times), result


def compute_condition_pytorch(A_scipy, norm_type):
    """Compute condition number using PyTorch (dense)."""
    A_dense = torch.tensor(A_scipy.toarray(), dtype=torch.float64, device='cuda')
    
    if norm_type == 1:
        # 1-norm condition number
        # PyTorch doesn't have direct 1-norm cond, compute manually
        norm_A = torch.max(torch.sum(torch.abs(A_dense), dim=0)).item()
        
        # ||A^{-1}||_1 using SVD approach (not optimal but works)
        U, S, Vh = torch.linalg.svd(A_dense)
        A_inv = Vh.T @ torch.diag(1.0 / S) @ U.T
        norm_Ainv = torch.max(torch.sum(torch.abs(A_inv), dim=0)).item()
        
        return norm_A * norm_Ainv
    else:
        # 2-norm condition number
        return torch.linalg.cond(A_dense, p=2).item()

def benchmark_matrix(n, density):
    """Benchmark single matrix size."""
    
    print(f"\n{'='*80}")
    print(f"Matrix Size: {n}x{n}, Density: {density}")
    print(f"{'='*80}")
    
    # Create test matrix
    A_scipy = scipy_sp.random(n, n, density=density, format='csr')
    A_scipy = A_scipy + scipy_sp.eye(n) * 5.0  # Well-conditioned
    
    # Convert to PyTorch-backed sparse
    A_torch_sparse = sp.csr_matrix(cp.array(A_scipy.toarray()))
    
    results = []
    
    # ========== 1-NORM TESTS ==========
    print(f"\n{'Method':<30} {'Time (s)':>12} {'Cond Number':>15} {'Rel Error':>12} {'Speedup':>10}")
    print("-"*80)
    
    # PyTorch 1-norm (reference)
    time_torch_1, cond_torch_1 = gpu_time(
        lambda: compute_condition_pytorch(A_scipy, norm_type=1),
        warmup=2, repeat=3
    )
    print(f"{'PyTorch 1-norm (dense)':<30} {time_torch_1:>12.4f} {cond_torch_1:>15.4e} {'-':>12} {'1.00x':>10}")
    
    # Sparse Hager-Higham 1-norm
    time_hh, result_hh = gpu_time(
        lambda: cond_estimate(A_torch_sparse, norm=1, method='hager-higham', 
                             solver='lu', return_dict=True),
        warmup=2, repeat=3
    )
    cond_hh = result_hh['condition_number']
    error_hh = abs(cond_hh - cond_torch_1) / cond_torch_1
    speedup_hh = time_torch_1 / time_hh
    print(f"{'Sparse Hager-Higham 1-norm':<30} {time_hh:>12.4f} {cond_hh:>15.4e} {error_hh:>12.2%} {speedup_hh:>10.2f}x")
    
    results.append({
        'norm': 1,
        'method': 'Hager-Higham',
        'time': time_hh,
        'cond': cond_hh,
        'error': error_hh,
        'speedup': speedup_hh
    })
    
    # ========== 2-NORM TESTS ==========
    print(f"\n{'Method':<30} {'Time (s)':>12} {'Cond Number':>15} {'Rel Error':>12} {'Speedup':>10}")
    print("-"*80)
    
    # PyTorch 2-norm (reference)
    time_torch_2, cond_torch_2 = gpu_time(
        lambda: compute_condition_pytorch(A_scipy, norm_type=2),
        warmup=2, repeat=3
    )
    print(f"{'PyTorch 2-norm (dense)':<30} {time_torch_2:>12.4f} {cond_torch_2:>15.4e} {'-':>12} {'1.00x':>10}")
    
    # Sparse SVDS 2-norm
    time_svds, result_svds = gpu_time(
        lambda: cond_estimate(A_torch_sparse, norm=2, method='svds', 
                             num_values=6, return_dict=True),
        warmup=2, repeat=3
    )
    cond_svds = result_svds['condition_number']
    error_svds = abs(cond_svds - cond_torch_2) / cond_torch_2
    speedup_svds = time_torch_2 / time_svds
    print(f"{'Sparse SVDS 2-norm':<30} {time_svds:>12.4f} {cond_svds:>15.4e} {error_svds:>12.2%} {speedup_svds:>10.2f}x")
    
    results.append({
        'norm': 2,
        'method': 'SVDS',
        'time': time_svds,
        'cond': cond_svds,
        'error': error_svds,
        'speedup': speedup_svds
    })
    
    # Sparse Lanczos 2-norm
    time_lanczos, result_lanczos = gpu_time(
        lambda: cond_estimate(A_torch_sparse, norm=2, method='lanczos', 
                             num_values=6, return_dict=True),
        warmup=2, repeat=3
    )
    cond_lanczos = result_lanczos['condition_number']
    error_lanczos = abs(cond_lanczos - cond_torch_2) / cond_torch_2
    speedup_lanczos = time_torch_2 / time_lanczos
    print(f"{'Sparse Lanczos 2-norm':<30} {time_lanczos:>12.4f} {cond_lanczos:>15.4e} {error_lanczos:>12.2%} {speedup_lanczos:>10.2f}x")
    
    results.append({
        'norm': 2,
        'method': 'Lanczos',
        'time': time_lanczos,
        'cond': cond_lanczos,
        'error': error_lanczos,
        'speedup': speedup_lanczos
    })
    
    # Sparse Golub-Kahan 2-norm
    time_gk, result_gk = gpu_time(
        lambda: cond_estimate(A_torch_sparse, norm=2, method='golub-kahan', 
                             max_iter=5, return_dict=True),
        warmup=2, repeat=3
    )
    cond_gk = result_gk['condition_number']
    error_gk = abs(cond_gk - cond_torch_2) / cond_torch_2
    speedup_gk = time_torch_2 / time_gk
    print(f"{'Sparse Golub-Kahan 2-norm':<30} {time_gk:>12.4f} {cond_gk:>15.4e} {error_gk:>12.2%} {speedup_gk:>10.2f}x")
    
    results.append({
        'norm': 2,
        'method': 'Golub-Kahan',
        'time': time_gk,
        'cond': cond_gk,
        'error': error_gk,
        'speedup': speedup_gk
    })
    
    return results


def main():
    """Run benchmarks."""
    
    print("="*80)
    print("Sparse vs Dense Condition Number Benchmark")
    print("="*80)
    
    # Test configurations
    configs = [
        (1000, 0.01),
        (3000, 0.005),
    ]
    
    all_results = []
    
    for n, density in configs:
        results = benchmark_matrix(n, density)
        all_results.extend([(n, density, r) for r in results])
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Size':<10} {'Density':<10} {'Norm':<6} {'Method':<15} {'Time (s)':>12} {'Error':>10} {'Speedup':>10}")
    print("-"*80)
    
    for n, density, r in all_results:
        print(f"{n:<10} {density:<10.4f} {r['norm']:<6} {r['method']:<15} "
              f"{r['time']:>12.4f} {r['error']:>10.2%} {r['speedup']:>10.2f}x")


if __name__ == '__main__':
    main()