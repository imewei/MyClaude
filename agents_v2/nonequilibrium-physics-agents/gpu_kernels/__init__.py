"""GPU Kernels for Nonequilibrium Physics Agents.

This module provides GPU-accelerated implementations of core algorithms using JAX
and optionally CUDA for maximum performance.

Modules:
- quantum_evolution: GPU-accelerated Lindblad equation solvers
- md_simulation: GPU molecular dynamics via HOOMD-blue
"""

__version__ = "4.0.0-dev"

# Try to import JAX backend
try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Try to import CuPy backend
try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

__all__ = [
    'JAX_AVAILABLE',
    'CUPY_AVAILABLE'
]

def get_available_backends():
    """Return list of available GPU backends."""
    backends = ['numpy']  # CPU fallback always available

    if JAX_AVAILABLE:
        backends.append('jax')
    if CUPY_AVAILABLE:
        backends.append('cupy')

    return backends

def get_default_backend():
    """Return default backend (prefer JAX > CuPy > NumPy)."""
    if JAX_AVAILABLE:
        return 'jax'
    elif CUPY_AVAILABLE:
        return 'cupy'
    else:
        return 'numpy'
