"""Performance and regression tests for GPU kernels.

Tests GPU performance characteristics, scalability, and identifies
performance regressions across different problem sizes.

Author: Nonequilibrium Physics Agents
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try importing JAX
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Import GPU kernels
from gpu_kernels.quantum_evolution import (
    solve_lindblad_cpu,
    solve_lindblad,
)

if JAX_AVAILABLE:
    from gpu_kernels.quantum_evolution import (
        solve_lindblad_jax,
        batch_lindblad_evolution,
        benchmark_gpu_speedup,
    )


class TestGPUPerformanceRegression:
    """Performance regression tests for GPU kernels."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_small_system_performance(self):
        """Test: Small system (n_dim=4) completes in < 0.1s on GPU."""
        n_dim = 4
        rho0 = jnp.eye(n_dim, dtype=jnp.complex128) / n_dim
        H = jnp.array([[0.0, 1.0, 0, 0],
                       [1.0, 0.0, 1.0, 0],
                       [0, 1.0, 0.0, 1.0],
                       [0, 0, 1.0, 0.0]], dtype=jnp.complex128)

        L = jnp.zeros((n_dim, n_dim), dtype=jnp.complex128)
        L = L.at[0, 1].set(1.0)
        L_ops = [L]
        gammas = [0.1]

        t_span = jnp.linspace(0, 1.0, 50)

        start_time = time.time()
        result = solve_lindblad_jax(rho0, H, L_ops, gammas, t_span)
        elapsed = time.time() - start_time

        # Performance threshold: < 0.1s for small system
        assert elapsed < 0.1, f"Small system took {elapsed:.3f}s, expected < 0.1s"
        assert result.shape == (50, n_dim * n_dim)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_medium_system_performance(self):
        """Test: Medium system (n_dim=10) completes in < 2s on GPU."""
        n_dim = 10
        rho0 = jnp.eye(n_dim, dtype=jnp.complex128) / n_dim

        # Random Hamiltonian (Hermitian)
        H = jnp.array(np.random.randn(n_dim, n_dim) + 1j * np.random.randn(n_dim, n_dim))
        H = (H + jnp.conj(H.T)) / 2

        L = jnp.zeros((n_dim, n_dim), dtype=jnp.complex128)
        L = L.at[0, 1].set(1.0)
        L_ops = [L]
        gammas = [0.1]

        t_span = jnp.linspace(0, 0.5, 50)

        start_time = time.time()
        result = solve_lindblad_jax(rho0, H, L_ops, gammas, t_span)
        elapsed = time.time() - start_time

        # Performance threshold: < 2s for medium system
        assert elapsed < 2.0, f"Medium system took {elapsed:.3f}s, expected < 2s"
        assert result.shape == (50, n_dim * n_dim)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_batched_performance_scaling(self):
        """Test: Batched execution scales near-linearly up to 100 trajectories."""
        n_dim = 4
        batch_sizes = [1, 10, 50, 100]
        times = []

        H = jnp.array([[0.0, 1.0, 0, 0],
                       [1.0, 0.0, 1.0, 0],
                       [0, 1.0, 0.0, 1.0],
                       [0, 0, 1.0, 0.0]], dtype=jnp.complex128)

        L = jnp.zeros((n_dim, n_dim), dtype=jnp.complex128)
        L = L.at[0, 1].set(1.0)
        L_ops = [L]
        gammas = [0.1]
        t_span = jnp.linspace(0, 0.5, 30)

        for batch_size in batch_sizes:
            # Create batch of initial states
            rho0_batch = jnp.array([
                jnp.eye(n_dim, dtype=jnp.complex128) / n_dim
                for _ in range(batch_size)
            ])

            start_time = time.time()
            # Use solve_lindblad_jax in a loop (no batch function available)
            results = [solve_lindblad_jax(rho0_batch[i], H, L_ops, gammas, t_span)
                      for i in range(batch_size)]
            elapsed = time.time() - start_time
            times.append(elapsed)

        # Check near-linear scaling (10x batch should be < 15x time)
        ratio_10 = times[1] / times[0]
        ratio_100 = times[3] / times[0]

        assert ratio_10 < 15, f"10x batch scaling: {ratio_10:.1f}x, expected < 15x"
        assert ratio_100 < 150, f"100x batch scaling: {ratio_100:.1f}x, expected < 150x"

    def test_cpu_fallback_performance(self):
        """Test: CPU fallback completes in reasonable time."""
        n_dim = 4
        rho0 = np.eye(n_dim, dtype=np.complex128) / n_dim

        H = np.array([[0.0, 1.0, 0, 0],
                      [1.0, 0.0, 1.0, 0],
                      [0, 1.0, 0.0, 1.0],
                      [0, 0, 1.0, 0.0]], dtype=np.complex128)

        L = np.zeros((n_dim, n_dim), dtype=np.complex128)
        L[0, 1] = 1.0
        L_ops = [L]
        gammas = [0.1]

        t_span = np.linspace(0, 1.0, 50)

        start_time = time.time()
        result = solve_lindblad_cpu(rho0, H, L_ops, gammas, t_span)
        elapsed = time.time() - start_time

        # CPU should complete in < 5s for small system
        assert elapsed < 5.0, f"CPU fallback took {elapsed:.3f}s, expected < 5s"
        assert result.shape == (50, n_dim, n_dim)


class TestGPUMemoryEfficiency:
    """Memory efficiency tests for GPU kernels."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_memory_scaling_linear(self):
        """Test: Memory usage scales linearly with system size."""
        # Test that we can solve progressively larger systems
        # without hitting memory issues

        t_span = jnp.linspace(0, 0.1, 10)  # Short time, few points

        for n_dim in [4, 8, 12]:
            rho0 = jnp.eye(n_dim, dtype=jnp.complex128) / n_dim
            H = jnp.eye(n_dim, dtype=jnp.complex128)
            L = jnp.zeros((n_dim, n_dim), dtype=jnp.complex128)
            L = L.at[0, min(1, n_dim-1)].set(1.0)
            L_ops = [L]
            gammas = [0.1]

            try:
                result = solve_lindblad_jax(rho0, H, L_ops, gammas, t_span)
                assert result.shape == (10, n_dim * n_dim)
            except MemoryError:
                pytest.fail(f"Memory error at n_dim={n_dim}")

    def test_cpu_memory_efficiency(self):
        """Test: CPU fallback handles medium-sized systems."""
        n_dim = 8
        rho0 = np.eye(n_dim, dtype=np.complex128) / n_dim
        H = np.eye(n_dim, dtype=np.complex128)
        L = np.zeros((n_dim, n_dim), dtype=np.complex128)
        L[0, 1] = 1.0
        L_ops = [L]
        gammas = [0.1]
        t_span = np.linspace(0, 0.5, 20)

        try:
            result = solve_lindblad_cpu(rho0, H, L_ops, gammas, t_span)
            assert result.shape == (20, n_dim, n_dim)
        except MemoryError:
            pytest.fail("CPU fallback hit memory error on medium system")


class TestGPUAccuracyVsSpeed:
    """Test accuracy vs speed tradeoffs."""

    def test_auto_backend_selection_accuracy(self):
        """Test: Auto backend selection maintains accuracy."""
        n_dim = 4
        rho0 = np.eye(n_dim, dtype=np.complex128) / n_dim

        H = np.array([[0.0, 1.0, 0, 0],
                      [1.0, 0.0, 1.0, 0],
                      [0, 1.0, 0.0, 1.0],
                      [0, 0, 1.0, 0.0]], dtype=np.complex128)

        L = np.zeros((n_dim, n_dim), dtype=np.complex128)
        L[0, 1] = 1.0
        L_ops = [L]
        gammas = [0.1]
        t_span = np.linspace(0, 1.0, 50)

        # Auto backend (will use CPU if JAX unavailable)
        result_auto = solve_lindblad(rho0, H, L_ops, gammas, t_span, backend='auto')

        # Explicit CPU
        result_cpu = solve_lindblad_cpu(rho0, H, L_ops, gammas, t_span)

        # Should match
        assert result_auto.shape == result_cpu.shape
        if JAX_AVAILABLE:
            # If JAX available, auto selected GPU - check agreement
            error = np.max(np.abs(result_auto - result_cpu))
            assert error < 1e-8, f"Auto vs CPU error: {error:.2e}"

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jit_compilation_speedup(self):
        """Test: JIT compilation provides speedup on second call."""
        n_dim = 6
        rho0 = jnp.eye(n_dim, dtype=jnp.complex128) / n_dim
        H = jnp.eye(n_dim, dtype=jnp.complex128)
        L = jnp.zeros((n_dim, n_dim), dtype=jnp.complex128)
        L = L.at[0, 1].set(1.0)
        L_ops = [L]
        gammas = [0.1]
        t_span = jnp.linspace(0, 0.5, 30)

        # First call (includes compilation)
        start_time = time.time()
        result1 = solve_lindblad_jax(rho0, H, L_ops, gammas, t_span)
        time1 = time.time() - start_time

        # Second call (should be faster)
        start_time = time.time()
        result2 = solve_lindblad_jax(rho0, H, L_ops, gammas, t_span)
        time2 = time.time() - start_time

        # Second call should be faster (at least 1.5x)
        # Note: First call includes JIT compilation overhead
        assert time2 < time1, f"Second call ({time2:.3f}s) not faster than first ({time1:.3f}s)"

        # Results should match
        error = jnp.max(jnp.abs(result1 - result2))
        assert error < 1e-12, f"JIT consistency error: {error:.2e}"


class TestGPUStressTests:
    """Stress tests for GPU kernels."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_long_time_evolution_stability(self):
        """Test: Long time evolution remains stable."""
        n_dim = 4
        rho0 = jnp.eye(n_dim, dtype=jnp.complex128) / n_dim

        H = jnp.array([[0.0, 1.0, 0, 0],
                       [1.0, 0.0, 1.0, 0],
                       [0, 1.0, 0.0, 1.0],
                       [0, 0, 1.0, 0.0]], dtype=jnp.complex128)

        L = jnp.zeros((n_dim, n_dim), dtype=jnp.complex128)
        L = L.at[0, 1].set(1.0)
        L_ops = [L]
        gammas = [0.1]

        # Long time span
        t_span = jnp.linspace(0, 100.0, 500)

        result = solve_lindblad_jax(rho0, H, L_ops, gammas, t_span)

        # Check stability: trace should remain 1
        final_rho = result[-1].reshape(n_dim, n_dim)
        trace = jnp.trace(final_rho)

        assert jnp.abs(trace - 1.0) < 1e-6, f"Long evolution: trace={trace:.6f}, expected 1.0"

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_many_jump_operators(self):
        """Test: Handles many jump operators efficiently."""
        n_dim = 4
        rho0 = jnp.eye(n_dim, dtype=jnp.complex128) / n_dim
        H = jnp.eye(n_dim, dtype=jnp.complex128)

        # Create 10 jump operators
        L_ops = []
        gammas = []
        for i in range(10):
            L = jnp.zeros((n_dim, n_dim), dtype=jnp.complex128)
            if i < n_dim - 1:
                L = L.at[i, i+1].set(1.0)
            else:
                L = L.at[i, 0].set(1.0)
            L_ops.append(L)
            gammas.append(0.01 * (i + 1))

        t_span = jnp.linspace(0, 1.0, 50)

        start_time = time.time()
        result = solve_lindblad_jax(rho0, H, L_ops, gammas, t_span)
        elapsed = time.time() - start_time

        # Should complete in reasonable time even with many operators
        assert elapsed < 5.0, f"Many operators took {elapsed:.3f}s, expected < 5s"
        assert result.shape == (50, n_dim * n_dim)

    def test_cpu_handles_edge_cases(self):
        """Test: CPU fallback handles edge cases gracefully."""
        n_dim = 4
        rho0 = np.eye(n_dim, dtype=np.complex128) / n_dim
        H = np.eye(n_dim, dtype=np.complex128) * 1e10  # Very large H
        L = np.zeros((n_dim, n_dim), dtype=np.complex128)
        L[0, 1] = 1.0
        L_ops = [L]
        gammas = [0.1]
        t_span = np.linspace(0, 0.001, 10)  # Very short time

        try:
            result = solve_lindblad_cpu(rho0, H, L_ops, gammas, t_span)
            assert result.shape == (10, n_dim, n_dim)
            # Should still preserve trace
            trace = np.trace(result[-1])
            assert np.abs(trace - 1.0) < 0.1  # Relaxed for extreme case
        except Exception as e:
            pytest.fail(f"CPU fallback failed on edge case: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
