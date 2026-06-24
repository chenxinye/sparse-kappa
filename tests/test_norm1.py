"""
Tests for 1-norm condition number estimation.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
import pytest
from sparse_kappa import cond_estimate


def test_hager_higham_identity():
    """Test on identity matrix (cond = 1)."""
    n = 100
    I = sp.identity(n, format='csr')
    
    cond = cond_estimate(I, norm=1, method='hager-higham')
    
    assert abs(cond - 1.0) < 0.1


def test_hager_higham_diagonal():
    """Test on diagonal matrix."""
    n = 100
    diag = cp.array([1.0, 2.0, 5.0, 10.0] * 25)
    A = sp.diags(diag, format='csr')
    
    # True condition number for diagonal: max/min
    true_cond = 10.0 / 1.0
    
    cond = cond_estimate(A, norm=1, method='hager-higham')
    
    # Allow 20% error
    assert abs(cond - true_cond) / true_cond < 0.2


def test_hager_higham_convergence():
    """Test that method converges."""
    n = 200
    A = sp.random(n, n, density=0.05, format='csr')
    
    result = cond_estimate(A, norm=1, method='hager-higham', 
                          max_iter=10, verbose=True)
    
    assert result['iterations'] <= 10
    assert result['condition_number'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])