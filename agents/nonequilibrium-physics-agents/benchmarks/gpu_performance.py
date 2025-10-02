"""
GPU Performance Benchmarking

Compares CPU vs GPU performance for optimal control computations:
- Quantum evolution (Schrödinger equation integration)
- Matrix operations (for control problems)
- Vector operations
- Memory transfer overhead

Author: Nonequilibrium Physics Agents
Date: 2025-10-01
"""

import numpy as np
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings

# Check for JAX/GPU availability
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True

    # Check if GPU is available
    try:
        # Try to create array on GPU
        test_array = jnp.ones(10)
        _ = jax.device_put(test_array, jax.devices('gpu')[0])
        GPU_AVAILABLE = True
    except (RuntimeError, IndexError):
        GPU_AVAILABLE = False
        warnings.warn("JAX available but no GPU detected. GPU benchmarks will use CPU.")
except ImportError:
    JAX_AVAILABLE = False
    GPU_AVAILABLE = False
    warnings.warn("JAX not available. GPU benchmarks will be skipped.")

from ml_optimal_control.performance import Timer


@dataclass
class GPUBenchmarkResult:
    """GPU benchmark result.

    Attributes
    ----------
    operation : str
        Operation name
    problem_size : int
        Problem dimension
    cpu_time : float
        CPU execution time
    gpu_time : float
        GPU execution time
    speedup : float
        GPU speedup factor (cpu_time / gpu_time)
    memory_transfer_time : float
        Time for CPU<->GPU memory transfer
    """
    operation: str
    problem_size: int
    cpu_time: float
    gpu_time: float
    speedup: float
    memory_transfer_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'operation': self.operation,
            'problem_size': self.problem_size,
            'cpu_time': self.cpu_time,
            'gpu_time': self.gpu_time,
            'speedup': self.speedup,
            'memory_transfer_time': self.memory_transfer_time
        }


class MatrixOperationBenchmark:
    """Benchmark matrix operations (CPU vs GPU).

    Tests matrix multiplication and eigenvalue computation.

    Parameters
    ----------
    matrix_sizes : List[int]
        List of matrix dimensions to test
    n_iterations : int
        Number of iterations for timing
    """

    def __init__(self, matrix_sizes: List[int] = None, n_iterations: int = 10):
        if matrix_sizes is None:
            self.matrix_sizes = [100, 500, 1000, 2000]
        else:
            self.matrix_sizes = matrix_sizes

        self.n_iterations = n_iterations

    def benchmark_matmul_cpu(self, size: int) -> float:
        """Benchmark matrix multiplication on CPU.

        Parameters
        ----------
        size : int
            Matrix dimension

        Returns
        -------
        float
            Execution time
        """
        A = np.random.randn(size, size)
        B = np.random.randn(size, size)

        timer = Timer()
        timer.start()

        for _ in range(self.n_iterations):
            C = A @ B

        elapsed = timer.stop()
        return elapsed / self.n_iterations

    def benchmark_matmul_gpu(self, size: int) -> Tuple[float, float]:
        """Benchmark matrix multiplication on GPU.

        Parameters
        ----------
        size : int
            Matrix dimension

        Returns
        -------
        compute_time : float
            GPU computation time
        transfer_time : float
            Memory transfer time
        """
        if not JAX_AVAILABLE:
            return self.benchmark_matmul_cpu(size), 0.0

        # Create data on CPU
        A_cpu = np.random.randn(size, size)
        B_cpu = np.random.randn(size, size)

        # Measure transfer time
        timer = Timer()
        timer.start()
        A_gpu = jnp.array(A_cpu)
        B_gpu = jnp.array(B_cpu)
        transfer_time = timer.stop()

        # Define JIT-compiled matmul
        @jit
        def matmul(A, B):
            return A @ B

        # Warm-up
        _ = matmul(A_gpu, B_gpu).block_until_ready()

        # Benchmark
        timer.start()
        for _ in range(self.n_iterations):
            C = matmul(A_gpu, B_gpu).block_until_ready()

        compute_time = timer.stop() / self.n_iterations

        return compute_time, transfer_time

    def run_benchmark(self) -> List[GPUBenchmarkResult]:
        """Run matrix multiplication benchmark.

        Returns
        -------
        List[GPUBenchmarkResult]
            Results for each matrix size
        """
        print("\n" + "=" * 80)
        print("MATRIX MULTIPLICATION BENCHMARK (CPU vs GPU)")
        print("=" * 80)
        print()

        results = []

        for size in self.matrix_sizes:
            print(f"Matrix size: {size}x{size}")

            # CPU benchmark
            cpu_time = self.benchmark_matmul_cpu(size)
            print(f"  CPU time: {cpu_time*1000:.2f} ms")

            # GPU benchmark
            if JAX_AVAILABLE:
                gpu_time, transfer_time = self.benchmark_matmul_gpu(size)
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0.0
                print(f"  GPU time: {gpu_time*1000:.2f} ms")
                print(f"  Transfer time: {transfer_time*1000:.2f} ms")
                print(f"  Speedup: {speedup:.2f}x")
            else:
                gpu_time = cpu_time
                transfer_time = 0.0
                speedup = 1.0
                print(f"  GPU: Not available")

            result = GPUBenchmarkResult(
                operation="MatrixMultiplication",
                problem_size=size,
                cpu_time=cpu_time,
                gpu_time=gpu_time,
                speedup=speedup,
                memory_transfer_time=transfer_time
            )

            results.append(result)
            print()

        return results


class QuantumEvolutionBenchmark:
    """Benchmark quantum evolution (Schrödinger equation).

    Simulates time evolution of quantum state using matrix exponential.

    Parameters
    ----------
    system_sizes : List[int]
        List of Hilbert space dimensions
    time_steps : int
        Number of time steps
    """

    def __init__(self, system_sizes: List[int] = None, time_steps: int = 100):
        if system_sizes is None:
            self.system_sizes = [10, 50, 100, 200]
        else:
            self.system_sizes = system_sizes

        self.time_steps = time_steps

    def evolve_cpu(self, H: np.ndarray, psi0: np.ndarray, dt: float) -> float:
        """Evolve quantum state on CPU.

        Parameters
        ----------
        H : np.ndarray
            Hamiltonian matrix
        psi0 : np.ndarray
            Initial state
        dt : float
            Time step

        Returns
        -------
        float
            Execution time
        """
        timer = Timer()
        timer.start()

        psi = psi0.copy()

        for _ in range(self.time_steps):
            # Simple Euler integration: ψ(t+dt) = ψ(t) - i*dt*H*ψ(t)
            psi = psi - 1j * dt * (H @ psi)
            # Renormalize
            psi = psi / np.linalg.norm(psi)

        elapsed = timer.stop()
        return elapsed

    def evolve_gpu(self, H_cpu: np.ndarray, psi0_cpu: np.ndarray, dt: float) -> Tuple[float, float]:
        """Evolve quantum state on GPU.

        Parameters
        ----------
        H_cpu : np.ndarray
            Hamiltonian matrix
        psi0_cpu : np.ndarray
            Initial state
        dt : float
            Time step

        Returns
        -------
        compute_time : float
            GPU computation time
        transfer_time : float
            Memory transfer time
        """
        if not JAX_AVAILABLE:
            return self.evolve_cpu(H_cpu, psi0_cpu, dt), 0.0

        # Transfer to GPU
        timer = Timer()
        timer.start()
        H = jnp.array(H_cpu)
        psi0 = jnp.array(psi0_cpu)
        transfer_time = timer.stop()

        @jit
        def evolve_step(psi, H, dt):
            psi_new = psi - 1j * dt * (H @ psi)
            norm = jnp.linalg.norm(psi_new)
            return psi_new / norm

        # Warm-up
        psi = psi0
        for _ in range(5):
            psi = evolve_step(psi, H, dt)
        psi.block_until_ready()

        # Benchmark
        timer.start()
        psi = psi0
        for _ in range(self.time_steps):
            psi = evolve_step(psi, H, dt)
        psi.block_until_ready()

        compute_time = timer.stop()

        return compute_time, transfer_time

    def run_benchmark(self) -> List[GPUBenchmarkResult]:
        """Run quantum evolution benchmark.

        Returns
        -------
        List[GPUBenchmarkResult]
            Results for each system size
        """
        print("\n" + "=" * 80)
        print("QUANTUM EVOLUTION BENCHMARK (CPU vs GPU)")
        print("=" * 80)
        print(f"Time steps: {self.time_steps}")
        print()

        results = []
        dt = 0.01

        for size in self.system_sizes:
            print(f"Hilbert space dimension: {size}")

            # Create random Hamiltonian (Hermitian)
            H_rand = np.random.randn(size, size) + 1j * np.random.randn(size, size)
            H = (H_rand + H_rand.conj().T) / 2  # Make Hermitian

            # Initial state
            psi0 = np.random.randn(size) + 1j * np.random.randn(size)
            psi0 = psi0 / np.linalg.norm(psi0)

            # CPU benchmark
            cpu_time = self.evolve_cpu(H, psi0, dt)
            print(f"  CPU time: {cpu_time:.4f} s")

            # GPU benchmark
            if JAX_AVAILABLE:
                gpu_time, transfer_time = self.evolve_gpu(H, psi0, dt)
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0.0
                print(f"  GPU time: {gpu_time:.4f} s")
                print(f"  Transfer time: {transfer_time:.4f} s")
                print(f"  Speedup: {speedup:.2f}x")
            else:
                gpu_time = cpu_time
                transfer_time = 0.0
                speedup = 1.0
                print(f"  GPU: Not available")

            result = GPUBenchmarkResult(
                operation="QuantumEvolution",
                problem_size=size,
                cpu_time=cpu_time,
                gpu_time=gpu_time,
                speedup=speedup,
                memory_transfer_time=transfer_time
            )

            results.append(result)
            print()

        return results


class VectorOperationBenchmark:
    """Benchmark vectorized operations.

    Tests element-wise operations and reductions.

    Parameters
    ----------
    vector_sizes : List[int]
        List of vector lengths
    n_iterations : int
        Number of iterations
    """

    def __init__(self, vector_sizes: List[int] = None, n_iterations: int = 100):
        if vector_sizes is None:
            self.vector_sizes = [1000, 10000, 100000, 1000000]
        else:
            self.vector_sizes = vector_sizes

        self.n_iterations = n_iterations

    def benchmark_cpu(self, size: int) -> float:
        """Benchmark vector operations on CPU.

        Parameters
        ----------
        size : int
            Vector length

        Returns
        -------
        float
            Execution time
        """
        x = np.random.randn(size)
        y = np.random.randn(size)

        timer = Timer()
        timer.start()

        for _ in range(self.n_iterations):
            # Complex element-wise operations
            z = np.sin(x) * np.cos(y) + np.exp(-x**2)
            result = np.sum(z)

        elapsed = timer.stop()
        return elapsed / self.n_iterations

    def benchmark_gpu(self, size: int) -> Tuple[float, float]:
        """Benchmark vector operations on GPU.

        Parameters
        ----------
        size : int
            Vector length

        Returns
        -------
        compute_time : float
            GPU computation time
        transfer_time : float
            Memory transfer time
        """
        if not JAX_AVAILABLE:
            return self.benchmark_cpu(size), 0.0

        x_cpu = np.random.randn(size)
        y_cpu = np.random.randn(size)

        # Transfer
        timer = Timer()
        timer.start()
        x = jnp.array(x_cpu)
        y = jnp.array(y_cpu)
        transfer_time = timer.stop()

        @jit
        def compute(x, y):
            z = jnp.sin(x) * jnp.cos(y) + jnp.exp(-x**2)
            return jnp.sum(z)

        # Warm-up
        _ = compute(x, y).block_until_ready()

        # Benchmark
        timer.start()
        for _ in range(self.n_iterations):
            result = compute(x, y).block_until_ready()

        compute_time = timer.stop() / self.n_iterations

        return compute_time, transfer_time

    def run_benchmark(self) -> List[GPUBenchmarkResult]:
        """Run vector operation benchmark.

        Returns
        -------
        List[GPUBenchmarkResult]
            Results for each vector size
        """
        print("\n" + "=" * 80)
        print("VECTOR OPERATIONS BENCHMARK (CPU vs GPU)")
        print("=" * 80)
        print()

        results = []

        for size in self.vector_sizes:
            print(f"Vector size: {size}")

            cpu_time = self.benchmark_cpu(size)
            print(f"  CPU time: {cpu_time*1000:.4f} ms")

            if JAX_AVAILABLE:
                gpu_time, transfer_time = self.benchmark_gpu(size)
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0.0
                print(f"  GPU time: {gpu_time*1000:.4f} ms")
                print(f"  Speedup: {speedup:.2f}x")
            else:
                gpu_time = cpu_time
                transfer_time = 0.0
                speedup = 1.0
                print(f"  GPU: Not available")

            result = GPUBenchmarkResult(
                operation="VectorOperations",
                problem_size=size,
                cpu_time=cpu_time,
                gpu_time=gpu_time,
                speedup=speedup,
                memory_transfer_time=transfer_time
            )

            results.append(result)
            print()

        return results


def run_gpu_benchmark_suite() -> Dict[str, List[GPUBenchmarkResult]]:
    """Run complete GPU benchmark suite.

    Returns
    -------
    Dict[str, List[GPUBenchmarkResult]]
        All GPU benchmark results
    """
    if not JAX_AVAILABLE:
        print("JAX not available. GPU benchmarks skipped.")
        return {}

    results = {}

    # Matrix operations
    print("\nRunning Matrix Operation Benchmarks...")
    matmul = MatrixOperationBenchmark(matrix_sizes=[100, 500, 1000], n_iterations=10)
    results['matrix_multiplication'] = matmul.run_benchmark()

    # Quantum evolution
    print("\nRunning Quantum Evolution Benchmarks...")
    quantum = QuantumEvolutionBenchmark(system_sizes=[10, 50, 100], time_steps=100)
    results['quantum_evolution'] = quantum.run_benchmark()

    # Vector operations
    print("\nRunning Vector Operation Benchmarks...")
    vector = VectorOperationBenchmark(vector_sizes=[10000, 100000, 1000000], n_iterations=100)
    results['vector_operations'] = vector.run_benchmark()

    return results


def print_gpu_summary(results: Dict[str, List[GPUBenchmarkResult]]) -> None:
    """Print GPU benchmark summary.

    Parameters
    ----------
    results : Dict[str, List[GPUBenchmarkResult]]
        GPU benchmark results
    """
    print("\n" + "=" * 80)
    print("GPU PERFORMANCE SUMMARY")
    print("=" * 80)

    for operation_name, operation_results in results.items():
        print(f"\n{operation_name.replace('_', ' ').title()}:")
        print("-" * 80)
        print(f"{'Size':>12s} {'CPU (ms)':>12s} {'GPU (ms)':>12s} {'Speedup':>12s}")
        print("-" * 80)

        for result in operation_results:
            print(f"{result.problem_size:12d} {result.cpu_time*1000:12.4f} "
                  f"{result.gpu_time*1000:12.4f} {result.speedup:11.2f}x")

        # Average speedup
        avg_speedup = np.mean([r.speedup for r in operation_results])
        print(f"\nAverage speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    results = run_gpu_benchmark_suite()
    if results:
        print_gpu_summary(results)
    else:
        print("GPU benchmarks not available (JAX not installed or no GPU detected)")
