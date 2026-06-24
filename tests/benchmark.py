"""
Speed comparison between sparse_kappa and dense PyTorch condition number computation.

Compares:
1. sparse_kappa methods (1-norm and 2-norm) on sparse matrices
2. PyTorch dense condition number computation
3. NumPy dense condition number computation
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
import torch
import numpy as np
import time
from sparse_kappa import cond_estimate
import pandas as pd
from typing import Dict


def create_sparse_matrix(n: int, density: float, condition_number: float = 100.0):
    """
    Create a sparse matrix with controlled condition number.
    
    Parameters
    ----------
    n : int
        Matrix size
    density : float
        Sparsity (fraction of non-zero elements)
    condition_number : float
        Target condition number
    
    Returns
    -------
    A_sparse : sparse matrix
        Sparse matrix on GPU
    A_dense_torch : torch.Tensor
        Dense matrix on GPU (PyTorch)
    A_dense_numpy : np.ndarray
        Dense matrix on CPU (NumPy)
    """
    # Create random sparse matrix
    A = sp.random(n, n, density=density, format='csr', dtype=np.float64)
    
    # Make it better conditioned by adding diagonal
    A = A + sp.eye(n, dtype=np.float64) * 10.0
    
    # Convert to dense for PyTorch and NumPy
    A_dense_cp = A.toarray()
    A_dense_np = cp.asnumpy(A_dense_cp)
    A_dense_torch = torch.from_numpy(A_dense_np).cuda().double()
    
    return A, A_dense_torch, A_dense_np


def benchmark_sparse_kappa(A: sp.spmatrix, norm: int, method: str, 
                           num_trials: int = 5) -> Dict:
    """Benchmark sparse_kappa method."""
    times = []
    cond_values = []
    
    # Warm-up
    _ = cond_estimate(A, norm=norm, method=method, max_iter=50, verbose=False)
    
    for _ in range(num_trials):
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        
        result = cond_estimate(A, norm=norm, method=method, max_iter=50, verbose=False)
        
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        times.append(elapsed)
        if isinstance(result, dict):
            cond_values.append(result['condition_number'])
        else:
            cond_values.append(result)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'cond_number': np.mean(cond_values),
        'cond_std': np.std(cond_values)
    }


def benchmark_pytorch_dense(A_torch: torch.Tensor, num_trials: int = 5) -> Dict:
    """Benchmark PyTorch dense condition number (2-norm only)."""
    times = []
    cond_values = []
    
    # Warm-up
    _ = torch.linalg.cond(A_torch)
    
    for _ in range(num_trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        cond = torch.linalg.cond(A_torch)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        times.append(elapsed)
        cond_values.append(cond.item())
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'cond_number': np.mean(cond_values),
        'cond_std': np.std(cond_values)
    }


def benchmark_numpy_dense(A_numpy: np.ndarray, norm: int, 
                         num_trials: int = 5) -> Dict:
    """Benchmark NumPy dense condition number (CPU)."""
    times = []
    cond_values = []
    
    # Warm-up
    _ = np.linalg.cond(A_numpy, p=norm)
    
    for _ in range(num_trials):
        start = time.perf_counter()
        
        cond = np.linalg.cond(A_numpy, p=norm)
        
        elapsed = time.perf_counter() - start
        
        times.append(elapsed)
        cond_values.append(cond)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'cond_number': np.mean(cond_values),
        'cond_std': np.std(cond_values)
    }


def run_comparison_suite():
    """Run comprehensive comparison across different matrix sizes and densities."""
    
    print("=" * 100)
    print("SPARSE vs DENSE CONDITION NUMBER COMPUTATION BENCHMARK")
    print("=" * 100)
    print()
    
    # Test configurations
    configs = [
        {'size': 500, 'density': 0.01, 'name': 'Small-VerySprase (500x500, 1%)'},
        {'size': 500, 'density': 0.05, 'name': 'Small-Sparse (500x500, 5%)'},
        {'size': 1000, 'density': 0.01, 'name': 'Medium-VerySparse (1000x1000, 1%)'},
        {'size': 1000, 'density': 0.05, 'name': 'Medium-Sparse (1000x1000, 5%)'},
        {'size': 2000, 'density': 0.005, 'name': 'Large-VerySparse (2000x2000, 0.5%)'},
        {'size': 2000, 'density': 0.01, 'name': 'Large-Sparse (2000x2000, 1%)'},
        {'size': 5000, 'density': 0.001, 'name': 'VeryLarge-UltraSparse (5000x5000, 0.1%)'},
        {'size': 5000, 'density': 0.005, 'name': 'VeryLarge-Sparse (5000x5000, 0.5%)'},
    ]
    
    # Methods to test for sparse_kappa
    sparse_methods_2norm = ['svds', 'power', 'lanczos', 'golub-kahan']
    sparse_methods_1norm = ['hager-higham']
    
    results = []
    
    for config in configs:
        print(f"\n{'=' * 100}")
        print(f"Configuration: {config['name']}")
        print(f"{'=' * 100}")
        
        n = config['size']
        density = config['density']
        nnz = int(n * n * density)
        
        print(f"Matrix size: {n}x{n}")
        print(f"Density: {density:.3%}")
        print(f"NNZ: {nnz:,}")
        print()
        
        # Create matrices
        print("Creating matrices...")
        A_sparse, A_torch, A_numpy = create_sparse_matrix(n, density)
        print("Done.\n")
        
        # ========== 2-NORM TESTS ==========
        print(f"{'Method':<30} {'Time (s)':<12} {'Speedup':<10} {'Cond Number':<15}")
        print("-" * 100)
        
        # Baseline: PyTorch dense (GPU)
        print("Testing PyTorch dense (GPU, 2-norm)...", end=" ", flush=True)
        pytorch_result = benchmark_pytorch_dense(A_torch, num_trials=3)
        baseline_time = pytorch_result['mean_time']
        print(f"✓")
        print(f"{'PyTorch Dense (GPU)':<30} {pytorch_result['mean_time']:<12.4f} {'1.00x':<10} "
              f"{pytorch_result['cond_number']:<15.4e}")
        
        results.append({
            'config': config['name'],
            'method': 'PyTorch Dense (GPU)',
            'norm': 2,
            'time': pytorch_result['mean_time'],
            'speedup': 1.0,
            'cond': pytorch_result['cond_number'],
            'nnz': nnz
        })
        
        # NumPy dense (CPU) - 2-norm
        print("Testing NumPy dense (CPU, 2-norm)...", end=" ", flush=True)
        numpy_result = benchmark_numpy_dense(A_numpy, norm=2, num_trials=3)
        speedup = baseline_time / numpy_result['mean_time']
        print(f"✓")
        print(f"{'NumPy Dense (CPU)':<30} {numpy_result['mean_time']:<12.4f} "
              f"{speedup:<10.2f} {numpy_result['cond_number']:<15.4e}")
        
        results.append({
            'config': config['name'],
            'method': 'NumPy Dense (CPU)',
            'norm': 2,
            'time': numpy_result['mean_time'],
            'speedup': speedup,
            'cond': numpy_result['cond_number'],
            'nnz': nnz
        })
        
        # Sparse methods - 2-norm
        for method in sparse_methods_2norm:
            print(f"Testing sparse_kappa {method} (2-norm)...", end=" ", flush=True)
            try:
                sparse_result = benchmark_sparse_kappa(A_sparse, norm=2, method=method, num_trials=3)
                speedup = baseline_time / sparse_result['mean_time']
                print(f"✓")
                print(f"{'sparse_kappa-' + method + ' (2-norm)':<30} {sparse_result['mean_time']:<12.4f} "
                      f"{speedup:<10.2f} {sparse_result['cond_number']:<15.4e}")
                
                results.append({
                    'config': config['name'],
                    'method': f'sparse_kappa-{method}',
                    'norm': 2,
                    'time': sparse_result['mean_time'],
                    'speedup': speedup,
                    'cond': sparse_result['cond_number'],
                    'nnz': nnz
                })
            except Exception as e:
                print(f"✗ (Error: {e})")
        
        print()
        
        # ========== 1-NORM TESTS ==========
        print("1-NORM TESTS:")
        print(f"{'Method':<30} {'Time (s)':<12} {'Speedup':<10} {'Cond Number':<15}")
        print("-" * 100)
        
        # NumPy dense (CPU) - 1-norm (baseline for 1-norm)
        print("Testing NumPy dense (CPU, 1-norm)...", end=" ", flush=True)
        numpy_1norm_result = benchmark_numpy_dense(A_numpy, norm=1, num_trials=3)
        baseline_1norm_time = numpy_1norm_result['mean_time']
        print(f"✓")
        print(f"{'NumPy Dense (CPU)':<30} {numpy_1norm_result['mean_time']:<12.4f} "
              f"{'1.00x':<10} {numpy_1norm_result['cond_number']:<15.4e}")
        
        results.append({
            'config': config['name'],
            'method': 'NumPy Dense (CPU)',
            'norm': 1,
            'time': numpy_1norm_result['mean_time'],
            'speedup': 1.0,
            'cond': numpy_1norm_result['cond_number'],
            'nnz': nnz
        })
        
        # Sparse methods - 1-norm
        for method in sparse_methods_1norm:
            print(f"Testing sparse_kappa {method} (1-norm)...", end=" ", flush=True)
            try:
                sparse_result = benchmark_sparse_kappa(A_sparse, norm=1, method=method, num_trials=3)
                speedup = baseline_1norm_time / sparse_result['mean_time']
                print(f"✓")
                print(f"{'sparse_kappa-' + method + ' (1-norm)':<30} {sparse_result['mean_time']:<12.4f} "
                      f"{speedup:<10.2f} {sparse_result['cond_number']:<15.4e}")
                
                results.append({
                    'config': config['name'],
                    'method': f'sparse_kappa-{method}',
                    'norm': 1,
                    'time': sparse_result['mean_time'],
                    'speedup': speedup,
                    'cond': sparse_result['cond_number'],
                    'nnz': nnz
                })
            except Exception as e:
                print(f"✗ (Error: {e})")
    
    # Summary table
    print("\n\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    
    df = pd.DataFrame(results)
    
    # Group by norm and method
    print("\n--- 2-NORM AVERAGE SPEEDUP vs PyTorch Dense (GPU) ---")
    norm2_df = df[df['norm'] == 2]
    speedup_summary = norm2_df.groupby('method')['speedup'].agg(['mean', 'min', 'max'])
    print(speedup_summary.sort_values('mean', ascending=False))
    
    print("\n--- 1-NORM AVERAGE SPEEDUP vs NumPy Dense (CPU) ---")
    norm1_df = df[df['norm'] == 1]
    if len(norm1_df) > 0:
        speedup_summary_1norm = norm1_df.groupby('method')['speedup'].agg(['mean', 'min', 'max'])
        print(speedup_summary_1norm.sort_values('mean', ascending=False))
    
    # Best methods per configuration
    print("\n--- FASTEST METHOD PER CONFIGURATION (2-norm) ---")
    for config_name in df['config'].unique():
        config_df = df[(df['config'] == config_name) & (df['norm'] == 2)]
        if len(config_df) > 0:
            fastest = config_df.loc[config_df['time'].idxmin()]
            print(f"{config_name:<50} {fastest['method']:<30} "
                  f"{fastest['speedup']:.2f}x faster")
    
    print("\n--- FASTEST METHOD PER CONFIGURATION (1-norm) ---")
    for config_name in df['config'].unique():
        config_df = df[(df['config'] == config_name) & (df['norm'] == 1)]
        if len(config_df) > 1:  # More than just NumPy
            fastest = config_df.loc[config_df['time'].idxmin()]
            print(f"{config_name:<50} {fastest['method']:<30} "
                  f"{fastest['speedup']:.2f}x faster")
    
    # Save results
    df.to_csv('benchmark_results.csv', index=False)
    print("\n✓ Results saved to benchmark_results.csv")
    
    return df


def plot_results(df: pd.DataFrame):
    """Plot benchmark results (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Time vs Matrix Size (2-norm)
        norm2_df = df[df['norm'] == 2]
        for method in norm2_df['method'].unique():
            method_df = norm2_df[norm2_df['method'] == method]
            axes[0, 0].plot(method_df['nnz'], method_df['time'], 'o-', label=method)
        axes[0, 0].set_xlabel('Number of Non-zeros')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].set_title('Computation Time vs Sparsity (2-norm)')
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Speedup vs Matrix Size (2-norm)
        for method in norm2_df['method'].unique():
            if 'sparse_kappa' in method:
                method_df = norm2_df[norm2_df['method'] == method]
                axes[0, 1].plot(method_df['nnz'], method_df['speedup'], 'o-', label=method)
        axes[0, 1].axhline(y=1.0, color='r', linestyle='--', label='Baseline (PyTorch)')
        axes[0, 1].set_xlabel('Number of Non-zeros')
        axes[0, 1].set_ylabel('Speedup vs PyTorch Dense')
        axes[0, 1].set_title('Speedup vs Baseline (2-norm)')
        axes[0, 1].set_xscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Time comparison (1-norm)
        norm1_df = df[df['norm'] == 1]
        if len(norm1_df) > 0:
            configs = norm1_df['config'].unique()
            x = np.arange(len(configs))
            width = 0.35
            
            numpy_times = [norm1_df[(norm1_df['config'] == c) & 
                                   (norm1_df['method'] == 'NumPy Dense (CPU)')]['time'].values[0] 
                          for c in configs]
            sparse_times = [norm1_df[(norm1_df['config'] == c) & 
                                    (norm1_df['method'].str.contains('sparse_kappa'))]['time'].values[0] 
                           if len(norm1_df[(norm1_df['config'] == c) & 
                                          (norm1_df['method'].str.contains('sparse_kappa'))]) > 0 
                           else 0 for c in configs]
            
            axes[1, 0].bar(x - width/2, numpy_times, width, label='NumPy Dense (CPU)')
            axes[1, 0].bar(x + width/2, sparse_times, width, label='sparse_kappa')
            axes[1, 0].set_xlabel('Configuration')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].set_title('Time Comparison (1-norm)')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([c.split('(')[0] for c in configs], rotation=45, ha='right')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Method comparison heatmap
        pivot_df = norm2_df.pivot_table(values='speedup', index='method', 
                                        columns='config', aggfunc='mean')
        im = axes[1, 1].imshow(pivot_df.values, cmap='RdYlGn', aspect='auto')
        axes[1, 1].set_xticks(np.arange(len(pivot_df.columns)))
        axes[1, 1].set_yticks(np.arange(len(pivot_df.index)))
        axes[1, 1].set_xticklabels([c.split('(')[0] for c in pivot_df.columns], 
                                   rotation=45, ha='right')
        axes[1, 1].set_yticklabels(pivot_df.index)
        axes[1, 1].set_title('Speedup Heatmap (2-norm)')
        plt.colorbar(im, ax=axes[1, 1], label='Speedup')
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        print("✓ Plots saved to benchmark_results.png")
        plt.show()
        
    except ImportError:
        print("Matplotlib not available, skipping plots")


if __name__ == '__main__':
    print("Starting benchmark suite...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch version: {cp.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    df = run_comparison_suite()
    
    # Try to plot results
    plot_results(df)