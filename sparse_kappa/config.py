"""
Configuration for sparse_kappa library.
"""

import warnings

# Warning control
SHOW_CONVERGENCE_WARNINGS = False  # Set to True to see all convergence warnings
SHOW_FORMAT_WARNINGS = False       # Set to True to see format conversion warnings

def configure_warnings(
    convergence: bool = False,
    format_conversion: bool = False
):
    """
    Configure warning display.
    
    Parameters
    ----------
    convergence : bool
        Show convergence warnings from iterative solvers
    format_conversion : bool
        Show matrix format conversion warnings
    """
    global SHOW_CONVERGENCE_WARNINGS, SHOW_FORMAT_WARNINGS
    SHOW_CONVERGENCE_WARNINGS = convergence
    SHOW_FORMAT_WARNINGS = format_conversion
    
    if not format_conversion:
        # Suppress specific warnings
        warnings.filterwarnings('ignore', category=UserWarning, 
                              message='.*CSR format is required.*')
        warnings.filterwarnings('ignore', 
                              module='PyTorch backend sparse.linalg._solve')