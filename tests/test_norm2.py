"""
Tests for 2-norm condition number estimation.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
import pytest
from sparse_kappa import cond_estimate


def test_svds_identity():
    """Test svds on identity matrix."""
    n = 100
    I = sp.identity(n, format='csr')
    
    cond = cond_estimate(I, norm=2, method='svds')
    
    assert abs(cond - 1.0) < 0.01


def test_svds_diagonal():
    """Test svds on diagonal matrix."""
    n = 100
    diag = cp.logspace(0, 3, n)  # [1, ..., 1000]
    A = sp.diags(diag, format='csr')
    
    true_cond = 1000.0
    
    cond = cond_estimate(A, norm=2, method='svds', num_values=10)
    
    # Allow 10% error
    assert abs(cond - true_cond) / true_cond < 0.1


def test_power_method():
    """Test power method."""
    n = 150
    A = sp.random(n, n, density=0.02, format='csr')
    
    result = cond_estimate(A, norm=2, method='power', 
                          max_iter=50, verbose=True)
    
    assert result['condition_number'] > 0
    assert result['sigma_max'] > result['sigma_min']


def test_lanczos():
    """Test Lanczos method."""
    n = 150
    A = sp.random(n, n, density=0.02, format='csr')
    
    result = cond_estimate(A, norm=2, method='lanczos', verbose=True)
    
    assert result['condition_number'] > 0


def test_golub_kahan():
    """Test Golub-Kahan method."""
    n = 150
    A = sp.random(n, n, density=0.02, format='csr')
    
    result = cond_estimate(A, norm=2, method='golub-kahan', verbose=True)
    
    assert result['condition_number'] > 0


def test_eigsh_symmetric():
    """Test eigsh on symmetric matrix."""
    n = 100
    A = sp.random(n, n, density=0.03, format='csr')
    A = (A + A.T) / 2  # Make symmetric
    
    result = cond_estimate(A, norm=2, method='eigsh', verbose=True)
    
    assert result['condition_number'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])