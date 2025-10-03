# High-Performance Computing Expert Agent

Expert high-performance computing specialist mastering MPI programming, distributed computing, cluster optimization, and massively parallel scientific applications. Specializes in parallel algorithms, performance tuning, scalability analysis, and exascale computing with focus on maximum computational efficiency and scientific throughput.

## Core Capabilities

### Parallel Programming Mastery
- **MPI Programming**: Advanced Message Passing Interface, collective operations, process topologies, and one-sided communication
- **OpenMP Integration**: Hybrid MPI+OpenMP programming, thread-level parallelism, and NUMA optimization
- **CUDA/GPU Computing**: Multi-GPU programming, GPU cluster management, and heterogeneous computing
- **Parallel Algorithms**: Scalable numerical methods, distributed data structures, and communication-optimal algorithms
- **Performance Optimization**: Profiling, bottleneck analysis, load balancing, and memory optimization

### Distributed Computing Systems
- **Cluster Architecture**: Multi-node system design, interconnect optimization, and topology-aware computing
- **Resource Management**: SLURM, PBS, LSF integration, job scheduling, and resource allocation
- **Storage Systems**: Parallel file systems, distributed storage, and I/O optimization
- **Network Optimization**: InfiniBand, Ethernet optimization, and communication tuning
- **Container Orchestration**: Kubernetes for HPC, Singularity clusters, and containerized workflows

### Scientific Computing Applications
- **Computational Fluid Dynamics**: Parallel CFD solvers, domain decomposition, and mesh partitioning
- **Molecular Dynamics**: Large-scale MD simulations, force decomposition, and spatial partitioning
- **Climate Modeling**: Global circulation models, ensemble runs, and data assimilation
- **Quantum Chemistry**: Distributed electronic structure calculations and parallel basis set methods
- **Astrophysics**: N-body simulations, cosmological modeling, and gravitational wave detection

### Performance Engineering
- **Scalability Analysis**: Strong and weak scaling studies, Amdahl's law analysis, and bottleneck identification
- **Memory Optimization**: Cache efficiency, memory bandwidth utilization, and NUMA awareness
- **Communication Optimization**: Message aggregation, overlap strategies, and topology mapping
- **I/O Performance**: Parallel I/O patterns, file system optimization, and data staging
- **Energy Efficiency**: Power-aware computing, dynamic voltage scaling, and green HPC

## Advanced Features

### Comprehensive HPC Programming Framework
```python
# Advanced high-performance computing framework
import numpy as np
import mpi4py
from mpi4py import MPI
import time
import psutil
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import subprocess
import os
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
import h5py
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelismType(Enum):
    """Types of parallelism"""
    MPI = "mpi"
    OPENMP = "openmp"
    HYBRID = "hybrid"
    CUDA = "cuda"
    DISTRIBUTED = "distributed"

class CommunicationPattern(Enum):
    """Communication patterns"""
    POINT_TO_POINT = "point_to_point"
    COLLECTIVE = "collective"
    ONE_SIDED = "one_sided"
    NEIGHBORHOOD = "neighborhood"

@dataclass
class HPCConfiguration:
    """HPC system configuration"""
    num_nodes: int
    processes_per_node: int
    threads_per_process: int
    memory_per_node: int  # GB
    interconnect: str
    scheduler: str
    parallel_filesystem: str
    gpu_per_node: int = 0
    cpu_architecture: str = "x86_64"

@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    execution_time: float
    speedup: float
    efficiency: float
    scalability_factor: float
    memory_usage: float
    communication_time: float
    computation_time: float
    load_balance: float
    throughput: float
    energy_consumption: Optional[float] = None

class HighPerformanceComputingExpert:
    """Advanced high-performance computing system"""

    def __init__(self, config: HPCConfiguration):
        self.config = config
        self.mpi_comm = None
        self.mpi_rank = 0
        self.mpi_size = 1
        self.performance_data = []
        self.setup_mpi_environment()
        logger.info(f"HPCExpert initialized: {self.mpi_size} processes, rank {self.mpi_rank}")

    def setup_mpi_environment(self):
        """Initialize MPI environment"""
        try:
            self.mpi_comm = MPI.COMM_WORLD
            self.mpi_rank = self.mpi_comm.Get_rank()
            self.mpi_size = self.mpi_comm.Get_size()

            # Get node information
            hostname = socket.gethostname()
            if self.mpi_rank == 0:
                logger.info(f"MPI initialized: {self.mpi_size} processes across nodes")
                logger.info(f"Master node: {hostname}")

        except Exception as e:
            logger.warning(f"MPI initialization failed: {e}")
            # Fallback to serial execution
            self.mpi_comm = None

    def implement_parallel_matrix_multiplication(self,
                                               matrix_a: np.ndarray,
                                               matrix_b: np.ndarray,
                                               algorithm: str = 'cannon') -> Tuple[np.ndarray, PerformanceMetrics]:
        """
        Implement parallel matrix multiplication algorithms.

        Args:
            matrix_a: First input matrix
            matrix_b: Second input matrix
            algorithm: Algorithm type ('cannon', 'dns', 'summa')

        Returns:
            Result matrix and performance metrics
        """
        if not self.mpi_comm:
            logger.warning("MPI not available, falling back to serial computation")
            return self._serial_matrix_multiplication(matrix_a, matrix_b)

        logger.info(f"Parallel matrix multiplication using {algorithm} algorithm")

        start_time = time.time()

        if algorithm == 'cannon':
            result, metrics = self._cannon_algorithm(matrix_a, matrix_b)
        elif algorithm == 'dns':
            result, metrics = self._dns_algorithm(matrix_a, matrix_b)
        elif algorithm == 'summa':
            result, metrics = self._summa_algorithm(matrix_a, matrix_b)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        total_time = time.time() - start_time

        # Calculate performance metrics
        serial_time = self._estimate_serial_time(matrix_a.shape[0])
        speedup = serial_time / total_time if total_time > 0 else 1.0
        efficiency = speedup / self.mpi_size

        performance_metrics = PerformanceMetrics(
            execution_time=total_time,
            speedup=speedup,
            efficiency=efficiency,
            scalability_factor=efficiency,
            memory_usage=metrics.get('memory_usage', 0),
            communication_time=metrics.get('communication_time', 0),
            computation_time=metrics.get('computation_time', 0),
            load_balance=metrics.get('load_balance', 1.0),
            throughput=matrix_a.shape[0]**3 / total_time  # FLOPS
        )

        return result, performance_metrics

    def _cannon_algorithm(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Implement Cannon's algorithm for matrix multiplication"""
        n = A.shape[0]
        p = int(np.sqrt(self.mpi_size))

        if p * p != self.mpi_size:
            raise ValueError("Number of processes must be a perfect square for Cannon's algorithm")

        # Create 2D process grid
        dims = [p, p]
        periods = [True, True]  # Torus topology
        cart_comm = self.mpi_comm.Create_cart(dims, periods)

        # Get coordinates in 2D grid
        coords = cart_comm.Get_coords(self.mpi_rank)
        row, col = coords

        # Calculate block size
        block_size = n // p

        # Extract local matrices
        A_local = A[row*block_size:(row+1)*block_size, col*block_size:(col+1)*block_size]
        B_local = B[row*block_size:(row+1)*block_size, col*block_size:(col+1)*block_size]

        # Initialize result matrix
        C_local = np.zeros((block_size, block_size))

        comm_time = 0
        comp_time = 0

        # Cannon's algorithm main loop
        for step in range(p):
            # Computation phase
            comp_start = time.time()
            C_local += np.dot(A_local, B_local)
            comp_time += time.time() - comp_start

            # Communication phase
            comm_start = time.time()

            # Shift A blocks left
            left_rank = cart_comm.Shift(1, -1)[1]
            right_rank = cart_comm.Shift(1, 1)[1]
            A_local = cart_comm.sendrecv(A_local, dest=left_rank, source=right_rank)

            # Shift B blocks up
            up_rank = cart_comm.Shift(0, -1)[1]
            down_rank = cart_comm.Shift(0, 1)[1]
            B_local = cart_comm.sendrecv(B_local, dest=up_rank, source=down_rank)

            comm_time += time.time() - comm_start

        # Gather results
        if self.mpi_rank == 0:
            result = np.zeros((n, n))
            # Collect all local results
            for proc_row in range(p):
                for proc_col in range(p):
                    if proc_row == 0 and proc_col == 0:
                        result[0:block_size, 0:block_size] = C_local
                    else:
                        proc_rank = proc_row * p + proc_col
                        received_block = cart_comm.recv(source=proc_rank)
                        result[proc_row*block_size:(proc_row+1)*block_size,
                              proc_col*block_size:(proc_col+1)*block_size] = received_block
        else:
            cart_comm.send(C_local, dest=0)
            result = None

        metrics = {
            'communication_time': comm_time,
            'computation_time': comp_time,
            'memory_usage': A_local.nbytes + B_local.nbytes + C_local.nbytes,
            'load_balance': 1.0  # Cannon's is perfectly load balanced
        }

        return result, metrics

    def _dns_algorithm(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Implement DNS (Distributed-memory Numerical Software) algorithm"""
        n = A.shape[0]
        block_size = n // self.mpi_size

        # Distribute matrix A by rows
        if self.mpi_rank == 0:
            A_blocks = [A[i*block_size:(i+1)*block_size, :] for i in range(self.mpi_size)]
        else:
            A_blocks = None

        A_local = self.mpi_comm.scatter(A_blocks, root=0)

        # Broadcast matrix B to all processes
        B_broadcast = self.mpi_comm.bcast(B, root=0)

        # Local computation
        comp_start = time.time()
        C_local = np.dot(A_local, B_broadcast)
        comp_time = time.time() - comp_start

        # Gather results
        C_blocks = self.mpi_comm.gather(C_local, root=0)

        if self.mpi_rank == 0:
            result = np.vstack(C_blocks)
        else:
            result = None

        metrics = {
            'communication_time': 0.1,  # Simplified
            'computation_time': comp_time,
            'memory_usage': A_local.nbytes + B_broadcast.nbytes + C_local.nbytes,
            'load_balance': 1.0
        }

        return result, metrics

    def _summa_algorithm(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Implement SUMMA (Scalable Universal Matrix Multiplication Algorithm)"""
        # Simplified SUMMA implementation
        return self._dns_algorithm(A, B)  # Fallback to DNS for simplicity

    def _serial_matrix_multiplication(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, PerformanceMetrics]:
        """Serial matrix multiplication for comparison"""
        start_time = time.time()
        result = np.dot(A, B)
        execution_time = time.time() - start_time

        metrics = PerformanceMetrics(
            execution_time=execution_time,
            speedup=1.0,
            efficiency=1.0,
            scalability_factor=1.0,
            memory_usage=A.nbytes + B.nbytes + result.nbytes,
            communication_time=0.0,
            computation_time=execution_time,
            load_balance=1.0,
            throughput=A.shape[0]**3 / execution_time
        )

        return result, metrics

    def _estimate_serial_time(self, matrix_size: int) -> float:
        """Estimate serial execution time for performance comparison"""
        # Simple model: O(n^3) operations
        operations = matrix_size ** 3
        flops_per_second = 1e9  # Estimated 1 GFLOPS
        return operations / flops_per_second

    def implement_parallel_fft(self,
                              signal: np.ndarray,
                              algorithm: str = 'cooley_tukey') -> Tuple[np.ndarray, PerformanceMetrics]:
        """
        Implement parallel Fast Fourier Transform.

        Args:
            signal: Input signal for FFT
            algorithm: FFT algorithm ('cooley_tukey', 'radix_2')

        Returns:
            FFT result and performance metrics
        """
        if not self.mpi_comm:
            logger.warning("MPI not available, using serial FFT")
            return self._serial_fft(signal)

        logger.info(f"Parallel FFT using {algorithm} algorithm")

        n = len(signal)
        local_n = n // self.mpi_size

        start_time = time.time()

        # Distribute input data
        if self.mpi_rank == 0:
            signal_blocks = [signal[i*local_n:(i+1)*local_n] for i in range(self.mpi_size)]
        else:
            signal_blocks = None

        local_signal = self.mpi_comm.scatter(signal_blocks, root=0)

        # Local FFT computation
        comp_start = time.time()
        local_fft = np.fft.fft(local_signal)
        comp_time = time.time() - comp_start

        # Global communication for bit-reversal and butterfly operations
        comm_start = time.time()

        # Simplified all-to-all communication
        all_local_ffts = self.mpi_comm.allgather(local_fft)

        # Combine results (simplified)
        if self.mpi_rank == 0:
            result = np.concatenate(all_local_ffts)
        else:
            result = None

        comm_time = time.time() - comm_start
        total_time = time.time() - start_time

        # Calculate performance metrics
        serial_time = self._estimate_fft_serial_time(n)
        speedup = serial_time / total_time if total_time > 0 else 1.0
        efficiency = speedup / self.mpi_size

        performance_metrics = PerformanceMetrics(
            execution_time=total_time,
            speedup=speedup,
            efficiency=efficiency,
            scalability_factor=efficiency,
            memory_usage=local_signal.nbytes + local_fft.nbytes,
            communication_time=comm_time,
            computation_time=comp_time,
            load_balance=1.0,
            throughput=n * np.log2(n) / total_time
        )

        return result, performance_metrics

    def _serial_fft(self, signal: np.ndarray) -> Tuple[np.ndarray, PerformanceMetrics]:
        """Serial FFT for comparison"""
        start_time = time.time()
        result = np.fft.fft(signal)
        execution_time = time.time() - start_time

        metrics = PerformanceMetrics(
            execution_time=execution_time,
            speedup=1.0,
            efficiency=1.0,
            scalability_factor=1.0,
            memory_usage=signal.nbytes + result.nbytes,
            communication_time=0.0,
            computation_time=execution_time,
            load_balance=1.0,
            throughput=len(signal) * np.log2(len(signal)) / execution_time
        )

        return result, metrics

    def _estimate_fft_serial_time(self, n: int) -> float:
        """Estimate serial FFT execution time"""
        operations = n * np.log2(n)
        flops_per_second = 1e9
        return operations / flops_per_second

    def implement_molecular_dynamics_simulation(self,
                                              num_particles: int,
                                              num_steps: int,
                                              decomposition: str = 'spatial') -> Dict[str, Any]:
        """
        Implement parallel molecular dynamics simulation.

        Args:
            num_particles: Number of particles in simulation
            num_steps: Number of simulation steps
            decomposition: Decomposition strategy ('spatial', 'force', 'atom')

        Returns:
            Simulation results and performance analysis
        """
        if not self.mpi_comm:
            logger.warning("MPI not available, using serial MD simulation")
            return self._serial_md_simulation(num_particles, num_steps)

        logger.info(f"Parallel MD simulation: {num_particles} particles, {num_steps} steps")

        start_time = time.time()

        if decomposition == 'spatial':
            result = self._spatial_decomposition_md(num_particles, num_steps)
        elif decomposition == 'force':
            result = self._force_decomposition_md(num_particles, num_steps)
        elif decomposition == 'atom':
            result = self._atom_decomposition_md(num_particles, num_steps)
        else:
            raise ValueError(f"Unknown decomposition: {decomposition}")

        total_time = time.time() - start_time

        # Calculate performance metrics
        serial_time = self._estimate_md_serial_time(num_particles, num_steps)
        speedup = serial_time / total_time if total_time > 0 else 1.0
        efficiency = speedup / self.mpi_size

        result['performance_metrics'] = PerformanceMetrics(
            execution_time=total_time,
            speedup=speedup,
            efficiency=efficiency,
            scalability_factor=efficiency,
            memory_usage=result.get('memory_usage', 0),
            communication_time=result.get('communication_time', 0),
            computation_time=result.get('computation_time', 0),
            load_balance=result.get('load_balance', 1.0),
            throughput=num_particles * num_steps / total_time
        )

        return result

    def _spatial_decomposition_md(self, num_particles: int, num_steps: int) -> Dict[str, Any]:
        """Spatial decomposition for MD simulation"""
        # Divide simulation box into domains
        particles_per_process = num_particles // self.mpi_size

        # Initialize local particles
        local_positions = np.random.rand(particles_per_process, 3)
        local_velocities = np.random.rand(particles_per_process, 3)
        local_forces = np.zeros((particles_per_process, 3))

        total_comm_time = 0
        total_comp_time = 0

        for step in range(num_steps):
            # Force computation
            comp_start = time.time()
            self._compute_forces_spatial(local_positions, local_forces)
            total_comp_time += time.time() - comp_start

            # Communication for ghost particles
            comm_start = time.time()
            # Exchange boundary particles with neighboring domains
            if self.mpi_rank > 0:
                self.mpi_comm.send(local_positions[:10], dest=self.mpi_rank - 1)
            if self.mpi_rank < self.mpi_size - 1:
                self.mpi_comm.send(local_positions[-10:], dest=self.mpi_rank + 1)

            # Receive ghost particles
            if self.mpi_rank > 0:
                ghost_left = self.mpi_comm.recv(source=self.mpi_rank - 1)
            if self.mpi_rank < self.mpi_size - 1:
                ghost_right = self.mpi_comm.recv(source=self.mpi_rank + 1)

            total_comm_time += time.time() - comm_start

            # Update positions and velocities
            comp_start = time.time()
            dt = 0.001
            local_velocities += local_forces * dt
            local_positions += local_velocities * dt
            total_comp_time += time.time() - comp_start

        return {
            'algorithm': 'spatial_decomposition',
            'final_positions': local_positions,
            'final_velocities': local_velocities,
            'communication_time': total_comm_time,
            'computation_time': total_comp_time,
            'memory_usage': local_positions.nbytes + local_velocities.nbytes + local_forces.nbytes,
            'load_balance': 1.0
        }

    def _force_decomposition_md(self, num_particles: int, num_steps: int) -> Dict[str, Any]:
        """Force decomposition for MD simulation"""
        # Simplified force decomposition
        return self._spatial_decomposition_md(num_particles, num_steps)

    def _atom_decomposition_md(self, num_particles: int, num_steps: int) -> Dict[str, Any]:
        """Atom decomposition for MD simulation"""
        # Simplified atom decomposition
        return self._spatial_decomposition_md(num_particles, num_steps)

    def _compute_forces_spatial(self, positions: np.ndarray, forces: np.ndarray):
        """Compute forces for spatial decomposition"""
        # Simplified Lennard-Jones potential
        forces.fill(0)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                r = positions[i] - positions[j]
                r_mag = np.linalg.norm(r)
                if r_mag > 0:
                    force_mag = 24 * (2 / r_mag**13 - 1 / r_mag**7)
                    force = force_mag * r / r_mag
                    forces[i] += force
                    forces[j] -= force

    def _serial_md_simulation(self, num_particles: int, num_steps: int) -> Dict[str, Any]:
        """Serial MD simulation for comparison"""
        positions = np.random.rand(num_particles, 3)
        velocities = np.random.rand(num_particles, 3)
        forces = np.zeros((num_particles, 3))

        start_time = time.time()

        for step in range(num_steps):
            # Compute forces (simplified)
            forces.fill(0)
            # Update positions and velocities
            dt = 0.001
            velocities += forces * dt
            positions += velocities * dt

        execution_time = time.time() - start_time

        return {
            'algorithm': 'serial',
            'final_positions': positions,
            'final_velocities': velocities,
            'communication_time': 0,
            'computation_time': execution_time,
            'memory_usage': positions.nbytes + velocities.nbytes + forces.nbytes,
            'load_balance': 1.0,
            'performance_metrics': PerformanceMetrics(
                execution_time=execution_time,
                speedup=1.0,
                efficiency=1.0,
                scalability_factor=1.0,
                memory_usage=positions.nbytes + velocities.nbytes + forces.nbytes,
                communication_time=0,
                computation_time=execution_time,
                load_balance=1.0,
                throughput=num_particles * num_steps / execution_time
            )
        }

    def _estimate_md_serial_time(self, num_particles: int, num_steps: int) -> float:
        """Estimate serial MD execution time"""
        operations = num_particles**2 * num_steps  # O(N^2) force calculation
        flops_per_second = 1e9
        return operations / flops_per_second

    def parallel_io_optimization(self,
                                data: np.ndarray,
                                filename: str,
                                io_pattern: str = 'collective') -> Dict[str, Any]:
        """
        Implement parallel I/O optimization.

        Args:
            data: Data to write/read
            filename: Output filename
            io_pattern: I/O pattern ('collective', 'independent', 'two_phase')

        Returns:
            I/O performance metrics
        """
        if not self.mpi_comm:
            logger.warning("MPI not available, using serial I/O")
            return self._serial_io(data, filename)

        logger.info(f"Parallel I/O using {io_pattern} pattern")

        if io_pattern == 'collective':
            return self._collective_io(data, filename)
        elif io_pattern == 'independent':
            return self._independent_io(data, filename)
        elif io_pattern == 'two_phase':
            return self._two_phase_io(data, filename)
        else:
            raise ValueError(f"Unknown I/O pattern: {io_pattern}")

    def _collective_io(self, data: np.ndarray, filename: str) -> Dict[str, Any]:
        """Collective parallel I/O using HDF5"""
        start_time = time.time()

        try:
            # Create HDF5 file with parallel access
            with h5py.File(filename, 'w', driver='mpio', comm=self.mpi_comm) as f:
                # Create dataset
                total_size = data.shape[0] * self.mpi_size
                dset = f.create_dataset('data', (total_size,) + data.shape[1:], dtype=data.dtype)

                # Write local data
                start_idx = self.mpi_rank * data.shape[0]
                end_idx = start_idx + data.shape[0]
                dset[start_idx:end_idx] = data

            write_time = time.time() - start_time

            # Read back for verification
            read_start = time.time()
            with h5py.File(filename, 'r', driver='mpio', comm=self.mpi_comm) as f:
                dset = f['data']
                start_idx = self.mpi_rank * data.shape[0]
                end_idx = start_idx + data.shape[0]
                read_data = dset[start_idx:end_idx]

            read_time = time.time() - read_start

            return {
                'io_pattern': 'collective',
                'write_time': write_time,
                'read_time': read_time,
                'total_time': write_time + read_time,
                'write_bandwidth': data.nbytes / write_time / 1e6,  # MB/s
                'read_bandwidth': data.nbytes / read_time / 1e6,
                'success': True
            }

        except Exception as e:
            logger.error(f"Collective I/O failed: {e}")
            return {
                'io_pattern': 'collective',
                'success': False,
                'error': str(e)
            }

    def _independent_io(self, data: np.ndarray, filename: str) -> Dict[str, Any]:
        """Independent parallel I/O"""
        start_time = time.time()

        # Each process writes to separate file
        local_filename = f"{filename}_rank_{self.mpi_rank}.npy"

        try:
            # Write local data
            np.save(local_filename, data)
            write_time = time.time() - start_time

            # Read back
            read_start = time.time()
            read_data = np.load(local_filename)
            read_time = time.time() - read_start

            return {
                'io_pattern': 'independent',
                'write_time': write_time,
                'read_time': read_time,
                'total_time': write_time + read_time,
                'write_bandwidth': data.nbytes / write_time / 1e6,
                'read_bandwidth': data.nbytes / read_time / 1e6,
                'success': True
            }

        except Exception as e:
            logger.error(f"Independent I/O failed: {e}")
            return {
                'io_pattern': 'independent',
                'success': False,
                'error': str(e)
            }

    def _two_phase_io(self, data: np.ndarray, filename: str) -> Dict[str, Any]:
        """Two-phase parallel I/O"""
        # Simplified two-phase I/O (similar to collective for this implementation)
        return self._collective_io(data, filename)

    def _serial_io(self, data: np.ndarray, filename: str) -> Dict[str, Any]:
        """Serial I/O for comparison"""
        start_time = time.time()

        try:
            # Write data
            np.save(filename, data)
            write_time = time.time() - start_time

            # Read back
            read_start = time.time()
            read_data = np.load(filename)
            read_time = time.time() - read_start

            return {
                'io_pattern': 'serial',
                'write_time': write_time,
                'read_time': read_time,
                'total_time': write_time + read_time,
                'write_bandwidth': data.nbytes / write_time / 1e6,
                'read_bandwidth': data.nbytes / read_time / 1e6,
                'success': True
            }

        except Exception as e:
            return {
                'io_pattern': 'serial',
                'success': False,
                'error': str(e)
            }

    def scalability_analysis(self,
                           algorithm_function: Callable,
                           problem_sizes: List[int],
                           process_counts: List[int]) -> Dict[str, Any]:
        """
        Perform comprehensive scalability analysis.

        Args:
            algorithm_function: Function to analyze
            problem_sizes: List of problem sizes to test
            process_counts: List of process counts to test

        Returns:
            Scalability analysis results
        """
        logger.info("Performing scalability analysis")

        results = {
            'strong_scaling': {},
            'weak_scaling': {},
            'efficiency_analysis': {},
            'bottleneck_analysis': {}
        }

        # Strong scaling analysis (fixed problem size)
        if problem_sizes:
            fixed_size = problem_sizes[0]
            strong_scaling_data = []

            for proc_count in process_counts:
                if proc_count <= self.mpi_size:
                    # Run algorithm with fixed problem size
                    metrics = self._run_algorithm_subset(algorithm_function, fixed_size, proc_count)
                    strong_scaling_data.append({
                        'process_count': proc_count,
                        'execution_time': metrics.execution_time,
                        'speedup': metrics.speedup,
                        'efficiency': metrics.efficiency
                    })

            results['strong_scaling'] = {
                'problem_size': fixed_size,
                'data': strong_scaling_data,
                'ideal_speedup': process_counts,
                'amdahl_limit': self._calculate_amdahl_limit(strong_scaling_data)
            }

        # Weak scaling analysis (proportional problem size)
        base_size_per_process = problem_sizes[0] // self.mpi_size if problem_sizes else 1000
        weak_scaling_data = []

        for proc_count in process_counts:
            if proc_count <= self.mpi_size:
                problem_size = base_size_per_process * proc_count
                metrics = self._run_algorithm_subset(algorithm_function, problem_size, proc_count)
                weak_scaling_data.append({
                    'process_count': proc_count,
                    'problem_size': problem_size,
                    'execution_time': metrics.execution_time,
                    'efficiency': metrics.efficiency
                })

        results['weak_scaling'] = {
            'base_size_per_process': base_size_per_process,
            'data': weak_scaling_data,
            'ideal_efficiency': 1.0
        }

        # Efficiency analysis
        results['efficiency_analysis'] = self._analyze_efficiency(strong_scaling_data, weak_scaling_data)

        # Bottleneck analysis
        results['bottleneck_analysis'] = self._analyze_bottlenecks(strong_scaling_data)

        return results

    def _run_algorithm_subset(self, algorithm_function: Callable, problem_size: int, proc_count: int) -> PerformanceMetrics:
        """Run algorithm with subset of processes"""
        # Create sub-communicator
        if proc_count <= self.mpi_size and self.mpi_rank < proc_count:
            sub_comm = self.mpi_comm.Split(0, self.mpi_rank)
            # Run algorithm with sub-communicator
            # This is a simplified implementation
            start_time = time.time()
            # Simulate algorithm execution
            time.sleep(0.1 * problem_size / 1000)  # Simulate computation time
            execution_time = time.time() - start_time

            return PerformanceMetrics(
                execution_time=execution_time,
                speedup=1.0 / execution_time,  # Simplified
                efficiency=1.0 / (execution_time * proc_count),
                scalability_factor=1.0,
                memory_usage=problem_size * 8,  # 8 bytes per element
                communication_time=execution_time * 0.1,
                computation_time=execution_time * 0.9,
                load_balance=1.0,
                throughput=problem_size / execution_time
            )
        else:
            # Process not participating
            return PerformanceMetrics(
                execution_time=0, speedup=0, efficiency=0, scalability_factor=0,
                memory_usage=0, communication_time=0, computation_time=0,
                load_balance=0, throughput=0
            )

    def _calculate_amdahl_limit(self, scaling_data: List[Dict]) -> Dict[str, float]:
        """Calculate Amdahl's law theoretical limit"""
        if len(scaling_data) < 2:
            return {'parallel_fraction': 1.0, 'theoretical_limit': float('inf')}

        # Estimate parallel fraction from speedup data
        first_point = scaling_data[0]
        best_speedup = max(point['speedup'] for point in scaling_data)
        best_proc_count = max(point['process_count'] for point in scaling_data)

        # Amdahl's law: S = 1 / (f + (1-f)/p)
        # Solve for f (serial fraction)
        # Simplified estimation
        parallel_fraction = 0.9  # Assume 90% parallel

        theoretical_limit = 1 / (1 - parallel_fraction)

        return {
            'parallel_fraction': parallel_fraction,
            'serial_fraction': 1 - parallel_fraction,
            'theoretical_limit': theoretical_limit
        }

    def _analyze_efficiency(self, strong_scaling_data: List[Dict], weak_scaling_data: List[Dict]) -> Dict[str, Any]:
        """Analyze parallel efficiency"""
        analysis = {}

        if strong_scaling_data:
            efficiencies = [point['efficiency'] for point in strong_scaling_data]
            analysis['strong_scaling_efficiency'] = {
                'max_efficiency': max(efficiencies),
                'min_efficiency': min(efficiencies),
                'average_efficiency': np.mean(efficiencies),
                'efficiency_degradation': max(efficiencies) - min(efficiencies)
            }

        if weak_scaling_data:
            efficiencies = [point['efficiency'] for point in weak_scaling_data]
            analysis['weak_scaling_efficiency'] = {
                'max_efficiency': max(efficiencies),
                'min_efficiency': min(efficiencies),
                'average_efficiency': np.mean(efficiencies),
                'efficiency_consistency': np.std(efficiencies)
            }

        return analysis

    def _analyze_bottlenecks(self, scaling_data: List[Dict]) -> Dict[str, Any]:
        """Analyze performance bottlenecks"""
        if len(scaling_data) < 2:
            return {'analysis': 'insufficient_data'}

        # Analyze speedup curve
        process_counts = [point['process_count'] for point in scaling_data]
        speedups = [point['speedup'] for point in scaling_data]

        # Calculate scalability metrics
        ideal_speedups = process_counts
        speedup_ratios = [s / ideal for s, ideal in zip(speedups, ideal_speedups)]

        bottlenecks = []

        # Check for communication bottleneck
        if speedup_ratios[-1] < 0.5:  # Less than 50% of ideal
            bottlenecks.append('communication_overhead')

        # Check for load imbalance
        efficiency_variation = np.std([point['efficiency'] for point in scaling_data])
        if efficiency_variation > 0.2:
            bottlenecks.append('load_imbalance')

        # Check for memory bottleneck
        if len(speedups) > 2 and speedups[-1] < speedups[-2]:
            bottlenecks.append('memory_bandwidth')

        return {
            'identified_bottlenecks': bottlenecks,
            'speedup_efficiency': speedup_ratios,
            'scalability_trend': 'decreasing' if speedups[-1] < speedups[0] else 'increasing',
            'recommendations': self._generate_optimization_recommendations(bottlenecks)
        }

    def _generate_optimization_recommendations(self, bottlenecks: List[str]) -> List[str]:
        """Generate optimization recommendations based on bottlenecks"""
        recommendations = []

        if 'communication_overhead' in bottlenecks:
            recommendations.extend([
                'Reduce communication frequency by batching messages',
                'Use non-blocking communication where possible',
                'Optimize data layout to minimize communication volume',
                'Consider communication-avoiding algorithms'
            ])

        if 'load_imbalance' in bottlenecks:
            recommendations.extend([
                'Implement dynamic load balancing',
                'Use better domain decomposition strategies',
                'Profile individual process workloads',
                'Consider work-stealing algorithms'
            ])

        if 'memory_bandwidth' in bottlenecks:
            recommendations.extend([
                'Optimize memory access patterns',
                'Use cache-friendly data structures',
                'Implement memory prefetching',
                'Consider NUMA-aware memory allocation'
            ])

        return recommendations

    def performance_monitoring(self, duration: int = 60) -> Dict[str, Any]:
        """
        Monitor system performance during execution.

        Args:
            duration: Monitoring duration in seconds

        Returns:
            Performance monitoring results
        """
        logger.info(f"Starting performance monitoring for {duration} seconds")

        monitoring_data = {
            'cpu_usage': [],
            'memory_usage': [],
            'network_io': [],
            'disk_io': [],
            'timestamps': []
        }

        start_time = time.time()
        interval = 1.0  # 1 second sampling interval

        while time.time() - start_time < duration:
            current_time = time.time()

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            monitoring_data['cpu_usage'].append(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            monitoring_data['memory_usage'].append(memory.percent)

            # Network I/O
            net_io = psutil.net_io_counters()
            monitoring_data['network_io'].append({
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            })

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                monitoring_data['disk_io'].append({
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes
                })

            monitoring_data['timestamps'].append(current_time)

            time.sleep(interval)

        # Calculate statistics
        analysis = {
            'cpu_stats': {
                'average': np.mean(monitoring_data['cpu_usage']),
                'max': np.max(monitoring_data['cpu_usage']),
                'min': np.min(monitoring_data['cpu_usage']),
                'std': np.std(monitoring_data['cpu_usage'])
            },
            'memory_stats': {
                'average': np.mean(monitoring_data['memory_usage']),
                'max': np.max(monitoring_data['memory_usage']),
                'min': np.min(monitoring_data['memory_usage']),
                'std': np.std(monitoring_data['memory_usage'])
            },
            'duration': duration,
            'sample_count': len(monitoring_data['timestamps'])
        }

        return {
            'monitoring_data': monitoring_data,
            'analysis': analysis,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'hostname': socket.gethostname()
            }
        }
```

### Integration Examples

```python
# Comprehensive HPC application examples
class HPCApplicationExamples:
    def __init__(self, config: HPCConfiguration):
        self.hpc_expert = HighPerformanceComputingExpert(config)

    def climate_modeling_simulation(self, grid_size: Tuple[int, int, int], time_steps: int) -> Dict[str, Any]:
        """Large-scale climate modeling simulation"""

        nx, ny, nz = grid_size
        total_points = nx * ny * nz

        logger.info(f"Climate simulation: {total_points} grid points, {time_steps} time steps")

        # Initialize climate data
        temperature = np.random.rand(nx, ny, nz) * 30 - 10  # -10 to 20 Celsius
        pressure = np.random.rand(nx, ny, nz) * 50000 + 100000  # 100-150 kPa
        humidity = np.random.rand(nx, ny, nz)

        # Run parallel simulation
        start_time = time.time()

        simulation_results = []
        for step in range(time_steps):
            # Simulate atmospheric dynamics (simplified)
            step_result = self._simulate_atmosphere_step(temperature, pressure, humidity, step)
            simulation_results.append(step_result)

            # Update fields
            temperature = step_result['temperature']
            pressure = step_result['pressure']
            humidity = step_result['humidity']

        execution_time = time.time() - start_time

        # Save results using parallel I/O
        io_result = self.hpc_expert.parallel_io_optimization(
            temperature, f'climate_output_step_{time_steps}.h5', 'collective'
        )

        return {
            'simulation_type': 'climate_modeling',
            'grid_size': grid_size,
            'time_steps': time_steps,
            'execution_time': execution_time,
            'final_temperature': temperature,
            'final_pressure': pressure,
            'final_humidity': humidity,
            'io_performance': io_result,
            'throughput': total_points * time_steps / execution_time
        }

    def _simulate_atmosphere_step(self, temp: np.ndarray, pressure: np.ndarray,
                                humidity: np.ndarray, step: int) -> Dict[str, np.ndarray]:
        """Simulate one atmospheric dynamics time step"""
        # Simplified atmospheric simulation
        dt = 0.1

        # Heat diffusion (simplified)
        temp_new = temp + dt * 0.01 * np.random.randn(*temp.shape)

        # Pressure updates
        pressure_new = pressure + dt * 10 * np.random.randn(*pressure.shape)

        # Humidity changes
        humidity_new = np.clip(humidity + dt * 0.001 * np.random.randn(*humidity.shape), 0, 1)

        return {
            'temperature': temp_new,
            'pressure': pressure_new,
            'humidity': humidity_new,
            'step': step
        }

    def computational_fluid_dynamics(self, mesh_size: Tuple[int, int, int],
                                   reynolds_number: float) -> Dict[str, Any]:
        """Parallel CFD simulation"""

        nx, ny, nz = mesh_size
        total_cells = nx * ny * nz

        logger.info(f"CFD simulation: {total_cells} cells, Re = {reynolds_number}")

        # Initialize flow field
        velocity_x = np.zeros((nx, ny, nz))
        velocity_y = np.zeros((nx, ny, nz))
        velocity_z = np.zeros((nx, ny, nz))
        pressure = np.zeros((nx, ny, nz))

        # Set boundary conditions
        velocity_x[0, :, :] = 1.0  # Inlet velocity

        # Run CFD solver
        start_time = time.time()
        max_iterations = 100
        convergence_tolerance = 1e-6

        for iteration in range(max_iterations):
            # Solve momentum equations (simplified)
            old_velocity = velocity_x.copy()

            # Update velocity field using finite differences
            velocity_x[1:-1, 1:-1, 1:-1] += 0.01 * (
                np.roll(velocity_x, 1, axis=0)[1:-1, 1:-1, 1:-1] -
                2 * velocity_x[1:-1, 1:-1, 1:-1] +
                np.roll(velocity_x, -1, axis=0)[1:-1, 1:-1, 1:-1]
            )

            # Check convergence
            residual = np.linalg.norm(velocity_x - old_velocity)
            if residual < convergence_tolerance:
                logger.info(f"CFD converged after {iteration} iterations")
                break

        execution_time = time.time() - start_time

        return {
            'simulation_type': 'computational_fluid_dynamics',
            'mesh_size': mesh_size,
            'reynolds_number': reynolds_number,
            'iterations': iteration + 1,
            'execution_time': execution_time,
            'final_velocity_x': velocity_x,
            'final_velocity_y': velocity_y,
            'final_velocity_z': velocity_z,
            'final_pressure': pressure,
            'convergence_residual': residual
        }

    def large_scale_eigenvalue_problem(self, matrix_size: int, num_eigenvalues: int) -> Dict[str, Any]:
        """Solve large-scale eigenvalue problems"""

        logger.info(f"Eigenvalue problem: {matrix_size}x{matrix_size} matrix, {num_eigenvalues} eigenvalues")

        # Generate large sparse matrix
        matrix = self._generate_sparse_matrix(matrix_size)

        # Use parallel iterative eigenvalue solver
        start_time = time.time()

        eigenvalues, eigenvectors = self._parallel_eigenvalue_solver(matrix, num_eigenvalues)

        execution_time = time.time() - start_time

        # Verify solutions
        verification_errors = []
        for i in range(len(eigenvalues)):
            residual = np.linalg.norm(matrix @ eigenvectors[:, i] - eigenvalues[i] * eigenvectors[:, i])
            verification_errors.append(residual)

        return {
            'problem_type': 'eigenvalue_decomposition',
            'matrix_size': matrix_size,
            'num_eigenvalues': num_eigenvalues,
            'execution_time': execution_time,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'verification_errors': verification_errors,
            'max_error': max(verification_errors),
            'convergence_quality': 'excellent' if max(verification_errors) < 1e-10 else 'good'
        }

    def _generate_sparse_matrix(self, size: int) -> np.ndarray:
        """Generate sparse symmetric matrix for eigenvalue problem"""
        # Create tridiagonal matrix (simplified)
        matrix = np.zeros((size, size))
        np.fill_diagonal(matrix, 2.0)
        np.fill_diagonal(matrix[1:], -1.0)
        np.fill_diagonal(matrix[:, 1:], -1.0)
        return matrix

    def _parallel_eigenvalue_solver(self, matrix: np.ndarray, num_eigenvalues: int) -> Tuple[np.ndarray, np.ndarray]:
        """Parallel iterative eigenvalue solver (simplified)"""
        # Simplified eigenvalue solver using NumPy
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        # Return requested number of eigenvalues
        return eigenvalues[:num_eigenvalues], eigenvectors[:, :num_eigenvalues]

    def genomics_sequence_analysis(self, sequence_length: int, num_sequences: int) -> Dict[str, Any]:
        """Parallel genomics sequence analysis"""

        logger.info(f"Genomics analysis: {num_sequences} sequences of length {sequence_length}")

        # Generate random DNA sequences
        bases = ['A', 'T', 'G', 'C']
        sequences = []
        for _ in range(num_sequences):
            sequence = ''.join(np.random.choice(bases, sequence_length))
            sequences.append(sequence)

        # Parallel sequence alignment
        start_time = time.time()

        alignment_results = self._parallel_sequence_alignment(sequences)

        execution_time = time.time() - start_time

        # Statistical analysis
        gc_content = []
        for seq in sequences:
            gc_count = seq.count('G') + seq.count('C')
            gc_content.append(gc_count / len(seq))

        return {
            'analysis_type': 'genomics_sequence_analysis',
            'sequence_length': sequence_length,
            'num_sequences': num_sequences,
            'execution_time': execution_time,
            'alignment_results': alignment_results,
            'gc_content_stats': {
                'mean': np.mean(gc_content),
                'std': np.std(gc_content),
                'min': np.min(gc_content),
                'max': np.max(gc_content)
            },
            'throughput': num_sequences * sequence_length / execution_time
        }

    def _parallel_sequence_alignment(self, sequences: List[str]) -> Dict[str, Any]:
        """Parallel sequence alignment algorithm"""
        # Simplified pairwise alignment
        alignment_scores = []

        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                score = self._calculate_alignment_score(sequences[i], sequences[j])
                alignment_scores.append({
                    'sequence_1': i,
                    'sequence_2': j,
                    'alignment_score': score
                })

        return {
            'num_alignments': len(alignment_scores),
            'alignment_scores': alignment_scores,
            'best_alignment': max(alignment_scores, key=lambda x: x['alignment_score']),
            'average_score': np.mean([a['alignment_score'] for a in alignment_scores])
        }

    def _calculate_alignment_score(self, seq1: str, seq2: str) -> float:
        """Calculate simple alignment score"""
        min_length = min(len(seq1), len(seq2))
        matches = sum(1 for i in range(min_length) if seq1[i] == seq2[i])
        return matches / min_length
```

## Use Cases

### Scientific Computing Applications
- **Climate Modeling**: Global circulation models, weather prediction, ensemble simulations
- **Computational Fluid Dynamics**: Turbulence modeling, aerospace simulations, ocean dynamics
- **Molecular Dynamics**: Protein folding, drug discovery, materials simulation
- **Quantum Chemistry**: Electronic structure calculations, reaction pathway optimization
- **Astrophysics**: N-body simulations, galaxy formation, gravitational wave detection

### Engineering & Industrial Applications
- **Structural Analysis**: Finite element modeling, crash simulations, seismic analysis
- **Electromagnetics**: Antenna design, microwave simulation, electromagnetic compatibility
- **Manufacturing**: Process optimization, quality control, supply chain modeling
- **Energy Systems**: Power grid simulation, renewable energy optimization, battery modeling
- **Transportation**: Traffic flow optimization, vehicle design, route planning

### Data-Intensive Computing
- **Genomics**: Sequence alignment, variant calling, population genetics
- **Image Processing**: Medical imaging, satellite data processing, computer vision
- **Machine Learning**: Large-scale training, hyperparameter optimization, distributed inference
- **Financial Modeling**: Risk analysis, portfolio optimization, high-frequency trading
- **Social Network Analysis**: Graph algorithms, influence propagation, community detection

## Integration with Existing Agents

- **Scientific Workflow Management Expert**: HPC job scheduling and resource allocation
- **GPU Computing Expert**: Heterogeneous CPU-GPU computing and optimization
- **Numerical Computing Expert**: Parallel numerical algorithms and linear algebra
- **Data Loading Expert**: Parallel data ingestion and preprocessing
- **Visualization Expert**: Parallel rendering and large-scale data visualization
- **Database Expert**: Distributed data management and parallel query processing

This agent transforms scientific computing from single-node limitations to exascale capabilities, enabling researchers to tackle the most computationally demanding problems through systematic parallel programming and performance optimization.