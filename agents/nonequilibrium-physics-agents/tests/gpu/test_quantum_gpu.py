"""Tests for GPU-accelerated quantum evolution.

Tests cover:
- Correctness vs CPU implementation
- Performance benchmarks
- Batched evolution
- Edge cases (large n_dim, long evolution)
"""

import pytest
import numpy as np
from typing import Dict, Any

# Check if GPU backend is available
try:
    from gpu_kernels.quantum_evolution import (
        solve_lindblad_gpu,
        solve_lindblad_cpu,
        solve_lindblad,
        benchmark_gpu_speedup,
        JAX_AVAILABLE
    )

    if JAX_AVAILABLE:
        import jax
        import jax.numpy as jnp
        from gpu_kernels.quantum_evolution import (
            lindblad_rhs_jax,
            compute_entropy_jax,
            compute_purity_jax,
            batch_lindblad_evolution
        )

        GPU_AVAILABLE = len(jax.devices('gpu')) > 0
    else:
        GPU_AVAILABLE = False

except ImportError:
    GPU_AVAILABLE = False
    JAX_AVAILABLE = False


# Skip all GPU tests if JAX not available
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_system():
    """Simple two-level system for testing."""
    n_dim = 2
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)  # Ground state
    H = -0.5 * np.array([[1, 0], [0, -1]], dtype=complex)  # Pauli-Z
    L = np.array([[0, 1], [0, 0]], dtype=complex)  # Lowering operator
    L_ops = [L]
    gammas = [0.1]
    t_span = np.linspace(0, 10, 100)

    return {
        'n_dim': n_dim,
        'rho0': rho0,
        'H': H,
        'L_ops': L_ops,
        'gammas': gammas,
        't_span': t_span
    }


@pytest.fixture
def medium_system():
    """Larger system (n_dim=10) for performance testing."""
    n_dim = 10
    rho0 = np.eye(n_dim, dtype=complex) / n_dim  # Maximally mixed
    H = np.diag(np.arange(n_dim, dtype=complex))  # Harmonic oscillator
    L = np.zeros((n_dim, n_dim), dtype=complex)
    L[0, 1] = 1.0  # Single decay channel
    L_ops = [L]
    gammas = [0.05]
    t_span = np.linspace(0, 10, 100)

    return {
        'n_dim': n_dim,
        'rho0': rho0,
        'H': H,
        'L_ops': L_ops,
        'gammas': gammas,
        't_span': t_span
    }


# ============================================================================
# Test 1-5: Correctness Tests
# ============================================================================

def test_lindblad_gpu_vs_cpu_simple(simple_system):
    """Test 1: GPU results match CPU for simple system."""
    # Solve on CPU
    result_cpu = solve_lindblad_cpu(**simple_system)

    # Solve on GPU
    result_gpu = solve_lindblad_gpu(**simple_system, backend='gpu')

    # Compare density matrices
    max_error = np.max(np.abs(result_cpu['rho_evolution'] - result_gpu['rho_evolution']))

    assert max_error < 1e-10, f"GPU-CPU mismatch: max error = {max_error}"

    # Compare observables
    entropy_error = np.max(np.abs(result_cpu['entropy'] - result_gpu['entropy']))
    assert entropy_error < 1e-8, f"Entropy mismatch: error = {entropy_error}"

    purity_error = np.max(np.abs(result_cpu['purity'] - result_gpu['purity']))
    assert purity_error < 1e-8, f"Purity mismatch: error = {purity_error}"


def test_lindblad_gpu_trace_preservation(simple_system):
    """Test 2: GPU solver preserves trace(ρ) = 1."""
    result = solve_lindblad_gpu(**simple_system, backend='gpu')

    traces = np.array([np.trace(rho) for rho in result['rho_evolution']])
    trace_error = np.max(np.abs(traces - 1.0))

    assert trace_error < 1e-6, f"Trace not preserved: max error = {trace_error}"


def test_lindblad_gpu_hermiticity(simple_system):
    """Test 3: GPU solver preserves Hermiticity ρ = ρ†."""
    result = solve_lindblad_gpu(**simple_system, backend='gpu')

    for i, rho in enumerate(result['rho_evolution']):
        hermiticity_error = np.max(np.abs(rho - np.conj(rho.T)))
        assert hermiticity_error < 1e-10, \
            f"Hermiticity violated at t={result['time_grid'][i]:.2f}: error = {hermiticity_error}"


def test_lindblad_gpu_positivity(simple_system):
    """Test 4: GPU solver maintains positivity (all eigenvalues ≥ 0)."""
    result = solve_lindblad_gpu(**simple_system, backend='gpu')

    for i, rho in enumerate(result['rho_evolution']):
        eigvals = np.linalg.eigvalsh(rho)
        min_eigval = np.min(eigvals)
        assert min_eigval >= -1e-8, \
            f"Negative eigenvalue at t={result['time_grid'][i]:.2f}: {min_eigval}"


def test_lindblad_gpu_entropy_monotonicity(simple_system):
    """Test 5: Entropy should increase (2nd law of thermodynamics)."""
    result = solve_lindblad_gpu(**simple_system, backend='gpu')

    entropy = result['entropy']

    # Check that entropy is non-decreasing (allow numerical tolerance)
    entropy_increase = np.diff(entropy)
    min_increase = np.min(entropy_increase)

    assert min_increase >= -1e-8, \
        f"Entropy decreased: min change = {min_increase}"


# ============================================================================
# Test 6-10: Performance Tests
# ============================================================================

@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_lindblad_gpu_speedup_medium(medium_system):
    """Test 6: GPU should be faster than CPU for n_dim=10."""
    import time

    # CPU timing
    start = time.time()
    result_cpu = solve_lindblad_cpu(**medium_system)
    cpu_time = time.time() - start

    # GPU timing
    start = time.time()
    result_gpu = solve_lindblad_gpu(**medium_system, backend='gpu')
    gpu_time = time.time() - start

    speedup = cpu_time / gpu_time

    print(f"CPU time: {cpu_time:.3f} sec")
    print(f"GPU time: {gpu_time:.3f} sec")
    print(f"Speedup: {speedup:.1f}x")

    # Target: At least 5x speedup for n_dim=10
    assert speedup > 5.0, f"GPU speedup only {speedup:.1f}x, expected > 5x"


@pytest.mark.slow
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_lindblad_gpu_large_system():
    """Test 7: GPU can handle n_dim=20 (intractable on CPU)."""
    n_dim = 20
    rho0 = np.eye(n_dim, dtype=complex) / n_dim
    H = np.diag(np.arange(n_dim, dtype=complex))
    L = np.zeros((n_dim, n_dim), dtype=complex)
    L[0, 1] = 1.0
    L_ops = [L]
    gammas = [0.01]
    t_span = np.linspace(0, 10, 50)  # Fewer steps for large system

    import time
    start = time.time()

    result = solve_lindblad_gpu(rho0, H, L_ops, gammas, t_span, backend='gpu')

    gpu_time = time.time() - start

    print(f"n_dim=20, n_steps=50: {gpu_time:.3f} sec")

    # Target: < 10 seconds
    assert gpu_time < 10.0, f"n_dim=20 took {gpu_time:.1f} sec, expected < 10 sec"

    # Verify correctness
    traces = np.array([np.trace(rho) for rho in result['rho_evolution']])
    assert np.max(np.abs(traces - 1.0)) < 1e-5


def test_lindblad_auto_backend_selection(simple_system):
    """Test 8: Auto backend selection uses GPU if available."""
    result = solve_lindblad(**simple_system, backend='auto')

    if GPU_AVAILABLE:
        assert result['backend_used'] == 'gpu', "Auto should select GPU when available"
    else:
        assert result['backend_used'] == 'cpu', "Auto should fallback to CPU"


def test_benchmark_gpu_speedup():
    """Test 9: Benchmark utility runs successfully."""
    benchmark = benchmark_gpu_speedup(n_dim=8, n_steps=50, duration=5.0)

    assert 'cpu_time' in benchmark
    assert 'n_dim' in benchmark
    assert benchmark['n_dim'] == 8

    if JAX_AVAILABLE:
        assert 'gpu_time' in benchmark
        assert 'speedup' in benchmark
        assert 'max_error' in benchmark

        if GPU_AVAILABLE:
            print(f"Benchmark (n_dim=8): {benchmark['speedup']:.1f}x speedup")
            assert benchmark['speedup'] > 1.0
            assert benchmark['max_error'] < 1e-8


@pytest.mark.slow
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_batched_evolution_performance():
    """Test 10: Batched evolution for multiple initial conditions."""
    n_dim = 4
    batch_size = 100

    # Create batch of initial states
    rho0_batch = jnp.array([
        np.eye(n_dim, dtype=complex) / n_dim for _ in range(batch_size)
    ])

    H = jnp.array(np.diag(np.arange(n_dim, dtype=complex)))
    L = jnp.zeros((n_dim, n_dim), dtype=complex)
    L = L.at[0, 1].set(1.0)
    L_ops = [L]
    gammas = [0.1]
    t_span = jnp.linspace(0, 10, 50)

    import time
    start = time.time()

    result_batch = batch_lindblad_evolution(rho0_batch, H, L_ops, gammas, t_span)

    batch_time = time.time() - start

    print(f"Batch size {batch_size}, n_dim={n_dim}: {batch_time:.3f} sec")
    print(f"Time per trajectory: {batch_time / batch_size * 1000:.1f} ms")

    assert result_batch.shape == (batch_size, len(t_span), n_dim, n_dim)

    # Target: < 1 second for 100 trajectories
    assert batch_time < 1.0, f"Batched evolution too slow: {batch_time:.1f} sec"


# ============================================================================
# Test 11-15: Edge Cases
# ============================================================================

def test_lindblad_zero_decay(simple_system):
    """Test 11: Zero decay rate should give unitary evolution."""
    system = simple_system.copy()
    system['gammas'] = [0.0]  # No dissipation

    result = solve_lindblad_gpu(**system, backend='gpu' if GPU_AVAILABLE else 'cpu')

    # Purity should remain 1.0 (pure state)
    purity = result['purity']
    assert np.allclose(purity, 1.0, atol=1e-5), "Purity not preserved in unitary evolution"


def test_lindblad_high_decay(simple_system):
    """Test 12: High decay rate causes fast relaxation."""
    system = simple_system.copy()
    system['gammas'] = [10.0]  # Strong dissipation

    result = solve_lindblad_gpu(**system, backend='gpu' if GPU_AVAILABLE else 'cpu')

    # Should reach steady state quickly
    rho_final = result['rho_evolution'][-1]
    rho_steady = result['rho_evolution'][-1]

    # For strong decay, should approach ground state
    ground_state_pop = np.real(rho_final[0, 0])
    assert ground_state_pop > 0.95, f"Did not relax to ground state: pop = {ground_state_pop}"


def test_lindblad_long_evolution(simple_system):
    """Test 13: Long evolution time remains stable."""
    system = simple_system.copy()
    system['t_span'] = np.linspace(0, 100, 500)  # Long time

    result = solve_lindblad_gpu(**system, backend='gpu' if GPU_AVAILABLE else 'cpu')

    # Check stability
    traces = np.array([np.trace(rho) for rho in result['rho_evolution']])
    trace_error = np.max(np.abs(traces - 1.0))

    assert trace_error < 1e-5, f"Long evolution unstable: trace error = {trace_error}"


def test_lindblad_multiple_jump_operators(simple_system):
    """Test 14: Multiple jump operators (decay + dephasing)."""
    system = simple_system.copy()

    # Add dephasing operator
    L_dephase = np.array([[1, 0], [0, -1]], dtype=complex)
    system['L_ops'] = system['L_ops'] + [L_dephase]
    system['gammas'] = [0.1, 0.05]

    result = solve_lindblad_gpu(**system, backend='gpu' if GPU_AVAILABLE else 'cpu')

    # Should still preserve trace and positivity
    traces = np.array([np.trace(rho) for rho in result['rho_evolution']])
    assert np.max(np.abs(traces - 1.0)) < 1e-6

    # Entropy should increase due to dephasing
    assert result['entropy'][-1] > result['entropy'][0]


def test_lindblad_excited_initial_state(simple_system):
    """Test 15: Starting from excited state."""
    system = simple_system.copy()
    system['rho0'] = np.array([[0, 0], [0, 1]], dtype=complex)  # Excited state

    result = solve_lindblad_gpu(**system, backend='gpu' if GPU_AVAILABLE else 'cpu')

    # Should decay to ground state
    rho_final = result['rho_evolution'][-1]
    ground_pop_final = np.real(rho_final[0, 0])

    assert ground_pop_final > 0.9, \
        f"Did not decay to ground state: final ground pop = {ground_pop_final}"


# ============================================================================
# Test 16-20: Observable Computation Tests
# ============================================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_entropy_computation():
    """Test 16: Entropy computation correctness."""
    # Pure state: entropy should be 0
    rho_pure = jnp.array([[1, 0], [0, 0]], dtype=complex)
    entropy_pure = compute_entropy_jax(rho_pure)
    assert abs(entropy_pure) < 1e-10, f"Pure state entropy = {entropy_pure}, expected 0"

    # Maximally mixed state: entropy should be ln(2)
    rho_mixed = jnp.array([[0.5, 0], [0, 0.5]], dtype=complex)
    entropy_mixed = compute_entropy_jax(rho_mixed)
    expected_entropy = np.log(2)
    assert abs(entropy_mixed - expected_entropy) < 1e-6, \
        f"Mixed state entropy = {entropy_mixed}, expected {expected_entropy}"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_purity_computation():
    """Test 17: Purity computation correctness."""
    # Pure state: purity = 1
    rho_pure = jnp.array([[1, 0], [0, 0]], dtype=complex)
    purity_pure = compute_purity_jax(rho_pure)
    assert abs(purity_pure - 1.0) < 1e-10, f"Pure state purity = {purity_pure}, expected 1"

    # Maximally mixed: purity = 1/2
    rho_mixed = jnp.array([[0.5, 0], [0, 0.5]], dtype=complex)
    purity_mixed = compute_purity_jax(rho_mixed)
    assert abs(purity_mixed - 0.5) < 1e-6, f"Mixed state purity = {purity_mixed}, expected 0.5"


def test_observables_returned_correctly(simple_system):
    """Test 18: Observables returned in results dict."""
    result = solve_lindblad_gpu(**simple_system, backend='gpu' if GPU_AVAILABLE else 'cpu',
                                return_observables=True)

    assert 'entropy' in result
    assert 'purity' in result
    assert 'populations' in result

    assert len(result['entropy']) == len(simple_system['t_span'])
    assert len(result['purity']) == len(simple_system['t_span'])
    assert result['populations'].shape == (len(simple_system['t_span']), simple_system['n_dim'])


def test_observables_optional(simple_system):
    """Test 19: Observables can be disabled."""
    result = solve_lindblad_gpu(**simple_system, backend='gpu' if GPU_AVAILABLE else 'cpu',
                                return_observables=False)

    assert 'entropy' not in result
    assert 'purity' not in result
    assert 'populations' not in result

    assert 'rho_evolution' in result
    assert 'time_grid' in result


def test_backend_reporting(simple_system):
    """Test 20: Backend used is correctly reported."""
    result = solve_lindblad(**simple_system, backend='auto')

    assert 'backend_used' in result
    assert result['backend_used'] in ['cpu', 'gpu']

    if GPU_AVAILABLE:
        result_gpu = solve_lindblad(**simple_system, backend='gpu')
        assert result_gpu['backend_used'] == 'gpu'

    result_cpu = solve_lindblad(**simple_system, backend='cpu')
    assert result_cpu['backend_used'] == 'cpu'


# ============================================================================
# Benchmark Summary
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_comprehensive_benchmark():
    """Comprehensive benchmark across different problem sizes."""
    results = []

    for n_dim in [4, 8, 10, 15, 20]:
        print(f"\nBenchmarking n_dim={n_dim}...")
        benchmark = benchmark_gpu_speedup(n_dim=n_dim, n_steps=100, duration=10.0)

        if benchmark['speedup'] is not None:
            print(f"  CPU time: {benchmark['cpu_time']:.3f} sec")
            print(f"  GPU time: {benchmark['gpu_time']:.3f} sec")
            print(f"  Speedup: {benchmark['speedup']:.1f}x")
            print(f"  Max error: {benchmark['max_error']:.2e}")

            results.append(benchmark)

    # Summary
    if results:
        speedups = [r['speedup'] for r in results]
        print(f"\nAverage speedup: {np.mean(speedups):.1f}x")
        print(f"Best speedup: {np.max(speedups):.1f}x (n_dim={results[np.argmax(speedups)]['n_dim']})")
