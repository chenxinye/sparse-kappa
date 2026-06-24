"""
Tests for different 1-norm condition number methods.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
import pytest
from sparse_kappa import cond_estimate

def test_all_1norm_methods():
    """Test all 1-norm methods on same matrix."""
    n = 100
    A = sp.random(n, n, density=0.05, format='csr')
    A = A + sp.eye(n) * 5.0  # Make it well-conditioned
    
    methods = ['hager', 'hager-higham', 'power', 'monte-carlo', 'block-hager']
    results = {}
    
    for method in methods:
        params = {}
        if method == 'monte-carlo':
            params['num_samples'] = 30
        
        result = cond_estimate(A, norm=1, method=method, verbose=True, **params)
        results[method] = result['condition_number']
        print(f"{method}: {results[method]:.4e}")
        assert results[method] > 0
    
    # Hager-Higham should be most accurate, use as reference
    reference = results['hager-higham']
    
    for method, cond in results.items():
        rel_diff = abs(cond - reference) / reference
        print(f"{method} vs hager-higham: {rel_diff:.2%}")


def test_1norm_identity():
    """All 1-norm methods should give ~1 for identity."""
    n = 50
    I = sp.identity(n, format='csr')
    
    methods = ['hager', 'hager-higham', 'power', 'monte-carlo', 'block-hager']
    
    for method in methods:
        cond = cond_estimate(I, norm=1, method=method)
        print(f"{method}: {cond:.6f}")
        assert abs(cond - 1.0) < 0.3


def test_1norm_diagonal():
    """Test 1-norm on diagonal matrix."""
    diag = cp.array([1.0, 2.0, 5.0, 10.0])
    A = sp.diags(diag, format='csr')
    
    true_cond = 10.0 / 1.0
    
    methods = ['hager', 'hager-higham', 'power', 'monte-carlo', 'block-hager']
    
    for method in methods:
        cond = cond_estimate(A, norm=1, method=method)
        rel_error = abs(cond - true_cond) / true_cond
        print(f"{method}: cond={cond:.4e}, error={rel_error:.2%}")
        assert rel_error < 0.3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])