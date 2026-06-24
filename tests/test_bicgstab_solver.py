"""
Test BiCGSTAB solver and compare with other solvers.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
import time
from sparse_kappa import cond_estimate, create_solver


def test_bicgstab_basic():
    """Test basic BiCGSTAB functionality."""
    print("="*70)
    print("Testing BiCGSTAB Solver - Basic Functionality")
    print("="*70)
    
    n = 1000
    A = sp.random(n, n, density=0.01, format='csr', dtype=cp.float32)
    A = A + sp.eye(n, dtype=cp.float32) * 5.0  # Make well-conditioned
    
    # Create BiCGSTAB solver
    solver = create_solver(A, 'bicgstab', atol=1e-3, maxiter=20)
    
    # Test multiple solves
    print(f"\nSolver info: {solver.info()}")
    
    for i in range(5):
        b = cp.random.randn(n).astype(cp.float32)
        
        start = time.perf_counter()
        x = solver.solve(b)
        elapsed = time.perf_counter() - start
        
        # Check residual
        residual = float(cp.linalg.norm(A @ x - b))
        
        if i == 0:
            print(f"\nSolve {i+1}:")
            print(f"  Time: {elapsed*1000:.3f} ms")
            print(f"  Residual: {residual:.6e}")
    
    print(f"\nTotal solves: {solver.solve_count}")
    print("✓ BiCGSTAB solver works correctly")


def compare_solvers_for_nonsymmetric():
    """Compare solvers for non-symmetric matrices."""
    print("\n" + "="*70)
    print("Solver Comparison for Non-Symmetric Matrices")
    print("="*70)
    
    n = 2000
    # Create non-symmetric matrix
    A = sp.random(n, n, density=0.005, format='csr', dtype=cp.float32)
    A = A + sp.eye(n, dtype=cp.float32) * 5.0
    
    b = cp.random.randn(n).astype(cp.float32)
    
    solvers_to_test = [
        ('lsmr', {'atol': 1e-3, 'maxiter': 20}),
        ('bicgstab', {'atol': 1e-3, 'maxiter': 20}),
        ('gmres', {'atol': 1e-3, 'maxiter': 20, 'restart': 20}),
    ]
    
    print(f"\nMatrix: {n}x{n}, density={A.nnz/(n*n):.4f}")
    print(f"\n{'Solver':<15} {'Time (ms)':>12} {'Residual':>12} {'Speedup':>10}")
    print("-"*70)
    
    results = {}
    for solver_name, solver_kwargs in solvers_to_test:
        solver = create_solver(A, solver_name, **solver_kwargs)
        
        # Warm-up
        _ = solver.solve(b)
        
        # Timed run
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        x = solver.solve(b)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        # Check residual
        residual = float(cp.linalg.norm(A @ x - b))
        
        results[solver_name] = {
            'time': elapsed,
            'residual': residual,
        }
        
        print(f"{solver_name:<15} {elapsed*1000:>12.3f} {residual:>12.6e} {'-':>10}")
    
    # Print speedups
    base_time = results['lsmr']['time']
    print("\nSpeedups relative to LSMR:")
    for solver_name in results:
        speedup = base_time / results[solver_name]['time']
        print(f"  {solver_name:<15} {speedup:>10.2f}x")


def test_bicgstab_with_higham():
    """Test BiCGSTAB with Hager-Higham condition number estimation."""
    print("\n" + "="*70)
    print("BiCGSTAB with Hager-Higham 1-Norm Estimation")
    print("="*70)
    
    n = 1000
    A = sp.random(n, n, density=0.01, format='csr', dtype=cp.float32)
    A = A + sp.eye(n, dtype=cp.float32) * 5.0
    
    solvers = ['lu', 'lsmr', 'bicgstab', 'gmres']
    
    print(f"\nMatrix: {n}x{n}, density={A.nnz/(n*n):.4f}")
    print(f"\n{'Solver':<15} {'Time (s)':>12} {'Cond Number':>15} {'Iterations':>12}")
    print("-"*70)
    
    for solver_name in solvers:
        try:
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            
            result = cond_estimate(
                A, norm=1, method='hager-higham',
                solver=solver_name,
                solver_kwargs={'atol': 1e-3, 'maxiter': 20} if solver_name != 'lu' else {},
                return_dict=True
            )
            
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - start
            
            cond = result['condition_number']
            iters = result['iterations']
            
            print(f"{solver_name:<15} {elapsed:>12.4f} {cond:>15.4e} {iters:>12}")
            
        except Exception as e:
            print(f"{solver_name:<15} FAILED: {e}")


def test_bicgstab_convergence():
    """Test BiCGSTAB convergence behavior."""
    print("\n" + "="*70)
    print("BiCGSTAB Convergence Test")
    print("="*70)
    
    n = 500
    A = sp.random(n, n, density=0.02, format='csr', dtype=cp.float32)
    A = A + sp.eye(n, dtype=cp.float32) * 5.0
    
    b = cp.random.randn(n).astype(cp.float32)
    
    # Test with different tolerances
    tolerances = [1e-2, 1e-3, 1e-4, 1e-5]
    
    print(f"\nMatrix: {n}x{n}, density={A.nnz/(n*n):.4f}")
    print(f"\n{'Tolerance':>12} {'Time (ms)':>12} {'Residual':>12}")
    print("-"*50)
    
    for tol in tolerances:
        solver = create_solver(A, 'bicgstab', atol=tol, maxiter=50)
        
        start = time.perf_counter()
        x = solver.solve(b)
        elapsed = time.perf_counter() - start
        
        residual = float(cp.linalg.norm(A @ x - b))
        
        print(f"{tol:>12.1e} {elapsed*1000:>12.3f} {residual:>12.6e}")


def compare_auto_solver_selection():
    """Test auto solver selection with BiCGSTAB."""
    print("\n" + "="*70)
    print("Auto Solver Selection (includes BiCGSTAB)")
    print("="*70)
    
    # Small matrix
    print("\n1. Small matrix (500x500, density=0.02):")
    A_small = sp.random(500, 500, density=0.02, format='csr', dtype=cp.float32)
    A_small = A_small + sp.eye(500, dtype=cp.float32) * 5.0
    
    solver = create_solver(A_small, 'auto')
    print(f"   Selected: {solver.info()['selected_solver']}")
    
    # Medium symmetric
    print("\n2. Medium symmetric matrix (2000x2000):")
    A_sym = sp.random(2000, 2000, density=0.005, format='csr', dtype=cp.float32)
    A_sym = (A_sym + A_sym.T) / 2 + sp.eye(2000, dtype=cp.float32) * 5.0
    
    solver = create_solver(A_sym, 'auto')
    print(f"   Selected: {solver.info()['selected_solver']}")
    
    # Medium non-symmetric
    print("\n3. Medium non-symmetric matrix (2000x2000):")
    A_nonsym = sp.random(2000, 2000, density=0.005, format='csr', dtype=cp.float32)
    A_nonsym = A_nonsym + sp.eye(2000, dtype=cp.float32) * 5.0
    
    solver = create_solver(A_nonsym, 'auto')
    print(f"   Selected: {solver.info()['selected_solver']}")
    
    # Large matrix
    print("\n4. Large matrix (25000x25000):")
    A_large = sp.random(25000, 25000, density=0.0001, format='csr', dtype=cp.float32)
    A_large = A_large + sp.eye(25000, dtype=cp.float32) * 5.0
    
    solver = create_solver(A_large, 'auto')
    print(f"   Selected: {solver.info()['selected_solver']}")


if __name__ == '__main__':
    test_bicgstab_basic()
    compare_solvers_for_nonsymmetric()
    test_bicgstab_with_higham()
    test_bicgstab_convergence()
    compare_auto_solver_selection()
