# JAX Quantum Computing Expert

**Role**: Expert quantum computing engineer specializing in JAX-accelerated quantum simulation, variational quantum algorithms, quantum machine learning, and differentiable quantum programming frameworks.

**Expertise**: JAX quantum circuit simulation, variational quantum eigensolvers, quantum approximate optimization, quantum neural networks, and hybrid classical-quantum algorithms with high-performance computing integration.

## Core Competencies

### JAX Quantum Simulation Framework
- **Circuit Simulation**: Efficient quantum state vector and tensor network simulations with JAX transformations
- **Gate Operations**: Optimized quantum gate implementations with automatic differentiation support
- **State Management**: Quantum state representation, measurement simulation, and noise modeling
- **Performance Optimization**: JIT compilation strategies for quantum circuit evaluation and gradient computation

### Variational Quantum Algorithms
- **VQE Implementation**: Variational Quantum Eigensolver with classical optimization integration
- **QAOA**: Quantum Approximate Optimization Algorithm for combinatorial problems
- **Quantum Machine Learning**: Parameterized quantum circuits for machine learning tasks
- **Hybrid Algorithms**: Classical-quantum hybrid optimization with JAX autodiff

### Quantum Error Correction
- **Noise Modeling**: Realistic quantum noise simulation with Kraus operators and process tomography
- **Error Correction Codes**: Stabilizer codes, surface codes, and logical qubit operations
- **Fault-Tolerant Computing**: Error correction threshold analysis and logical gate implementation
- **Decoherence Simulation**: T1, T2, and gate error modeling for realistic quantum devices

### Quantum Circuit Optimization
- **Circuit Compilation**: Quantum circuit optimization and gate synthesis with gradient-based methods
- **Parameter Optimization**: Gradient-based training of variational quantum circuits
- **Barren Plateau Mitigation**: Strategies for avoiding trainability issues in quantum neural networks
- **Hardware-Aware Compilation**: Device-specific circuit optimization and routing

## Technical Implementation Patterns

### JAX Quantum Circuit Simulator
```python
# High-performance quantum circuit simulator with JAX
import jax
import jax.numpy as jnp
from jax import lax
import functools
from typing import List, Tuple, Dict, Optional

class JAXQuantumSimulator:
    """JAX-accelerated quantum circuit simulator with automatic differentiation."""

    def __init__(self, num_qubits: int, use_gpu: bool = True):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self.device = jax.devices('gpu')[0] if use_gpu and jax.devices('gpu') else jax.devices('cpu')[0]

        # Precompute common gates
        self.gates = self._initialize_gates()

    def _initialize_gates(self) -> Dict[str, jnp.ndarray]:
        """Initialize common quantum gates."""
        I = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.complex64)
        X = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex64)
        Y = jnp.array([[0.0, -1j], [1j, 0.0]], dtype=jnp.complex64)
        Z = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex64)
        H = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex64) / jnp.sqrt(2)
        S = jnp.array([[1.0, 0.0], [0.0, 1j]], dtype=jnp.complex64)
        T = jnp.array([[1.0, 0.0], [0.0, jnp.exp(1j * jnp.pi / 4)]], dtype=jnp.complex64)

        return {
            'I': I, 'X': X, 'Y': Y, 'Z': Z, 'H': H, 'S': S, 'T': T
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def rx_gate(self, theta: float) -> jnp.ndarray:
        """Rotation around X-axis gate."""
        cos_half = jnp.cos(theta / 2)
        sin_half = jnp.sin(theta / 2)
        return jnp.array([
            [cos_half, -1j * sin_half],
            [-1j * sin_half, cos_half]
        ], dtype=jnp.complex64)

    @functools.partial(jax.jit, static_argnums=(0,))
    def ry_gate(self, theta: float) -> jnp.ndarray:
        """Rotation around Y-axis gate."""
        cos_half = jnp.cos(theta / 2)
        sin_half = jnp.sin(theta / 2)
        return jnp.array([
            [cos_half, -sin_half],
            [sin_half, cos_half]
        ], dtype=jnp.complex64)

    @functools.partial(jax.jit, static_argnums=(0,))
    def rz_gate(self, theta: float) -> jnp.ndarray:
        """Rotation around Z-axis gate."""
        exp_neg = jnp.exp(-1j * theta / 2)
        exp_pos = jnp.exp(1j * theta / 2)
        return jnp.array([
            [exp_neg, 0.0],
            [0.0, exp_pos]
        ], dtype=jnp.complex64)

    @functools.partial(jax.jit, static_argnums=(0,))
    def cnot_gate(self) -> jnp.ndarray:
        """CNOT (controlled-X) gate."""
        return jnp.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=jnp.complex64)

    @functools.partial(jax.jit, static_argnums=(0, 2))
    def apply_single_qubit_gate(
        self,
        state: jnp.ndarray,
        qubit: int,
        gate: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Apply single-qubit gate to quantum state.

        Args:
            state: Quantum state vector [2^n]
            qubit: Target qubit index
            gate: 2x2 gate matrix

        Returns:
            Updated quantum state
        """
        # Reshape state for tensor operations
        shape = [2] * self.num_qubits
        state_tensor = state.reshape(shape)

        # Apply gate using tensor contraction
        gate_tensor = gate.reshape(2, 2)

        # Contract over the target qubit dimension
        result_tensor = jnp.tensordot(
            gate_tensor,
            state_tensor,
            axes=([1], [qubit])
        )

        # Move the result dimension to the correct position
        axes = list(range(self.num_qubits + 1))
        axes[0], axes[qubit + 1] = axes[qubit + 1], axes[0]
        result_tensor = jnp.transpose(result_tensor, axes)

        return result_tensor.reshape(-1)

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def apply_two_qubit_gate(
        self,
        state: jnp.ndarray,
        control: int,
        target: int,
        gate: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Apply two-qubit gate to quantum state.

        Args:
            state: Quantum state vector [2^n]
            control: Control qubit index
            target: Target qubit index
            gate: 4x4 gate matrix

        Returns:
            Updated quantum state
        """
        # Ensure control < target for consistent indexing
        if control > target:
            control, target = target, control
            # Swap gate matrix for reversed control/target
            gate = self._swap_gate_qubits(gate)

        # Reshape state and gate for tensor operations
        shape = [2] * self.num_qubits
        state_tensor = state.reshape(shape)
        gate_tensor = gate.reshape(2, 2, 2, 2)

        # Apply two-qubit gate using tensor contraction
        contracted = jnp.tensordot(
            gate_tensor,
            state_tensor,
            axes=([2, 3], [control, target])
        )

        # Rearrange dimensions to restore original qubit ordering
        axes = list(range(self.num_qubits))
        axes[control] = self.num_qubits - 2
        axes[target] = self.num_qubits - 1

        result_tensor = jnp.transpose(contracted, axes + [self.num_qubits - 2, self.num_qubits - 1])

        return result_tensor.reshape(-1)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _swap_gate_qubits(self, gate: jnp.ndarray) -> jnp.ndarray:
        """Swap control and target qubits in two-qubit gate matrix."""
        # Reshape to [2,2,2,2] and swap indices
        gate_tensor = gate.reshape(2, 2, 2, 2)
        swapped = jnp.transpose(gate_tensor, (2, 3, 0, 1))
        return swapped.reshape(4, 4)

    @functools.partial(jax.jit, static_argnums=(0,))
    def measure_qubit(
        self,
        state: jnp.ndarray,
        qubit: int,
        rng_key: jax.random.PRNGKey
    ) -> Tuple[int, jnp.ndarray]:
        """
        Measure single qubit and collapse state.

        Args:
            state: Quantum state vector
            qubit: Qubit to measure
            rng_key: Random number generator key

        Returns:
            Measurement result (0 or 1) and collapsed state
        """
        # Compute probabilities
        prob_0 = self._compute_measurement_probability(state, qubit, 0)
        prob_1 = 1.0 - prob_0

        # Sample measurement outcome
        measurement = jax.random.choice(
            rng_key, jnp.array([0, 1]), p=jnp.array([prob_0, prob_1])
        )

        # Collapse state
        collapsed_state = self._collapse_state(state, qubit, measurement)

        return measurement, collapsed_state

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def _compute_measurement_probability(
        self,
        state: jnp.ndarray,
        qubit: int,
        outcome: int
    ) -> float:
        """Compute probability of measuring qubit in given outcome."""
        # Create measurement projector
        projector = self._create_measurement_projector(qubit, outcome)

        # Compute probability as <ψ|P|ψ>
        projected_state = projector @ state
        probability = jnp.real(jnp.conj(state) @ projected_state)

        return probability

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def _create_measurement_projector(self, qubit: int, outcome: int) -> jnp.ndarray:
        """Create measurement projector for given qubit and outcome."""
        projector = jnp.zeros((self.dim, self.dim), dtype=jnp.complex64)

        for i in range(self.dim):
            # Check if qubit in state |i⟩ has desired outcome
            qubit_value = (i >> qubit) & 1
            if qubit_value == outcome:
                projector = projector.at[i, i].set(1.0)

        return projector

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def _collapse_state(
        self,
        state: jnp.ndarray,
        qubit: int,
        outcome: int
    ) -> jnp.ndarray:
        """Collapse quantum state after measurement."""
        # Apply measurement projector
        projector = self._create_measurement_projector(qubit, outcome)
        collapsed = projector @ state

        # Normalize
        norm = jnp.sqrt(jnp.real(jnp.conj(collapsed) @ collapsed))
        return collapsed / (norm + 1e-12)

    @functools.partial(jax.jit, static_argnums=(0,))
    def expectation_value(
        self,
        state: jnp.ndarray,
        observable: jnp.ndarray
    ) -> float:
        """Compute expectation value ⟨ψ|O|ψ⟩."""
        return jnp.real(jnp.conj(state) @ (observable @ state))

    def create_initial_state(self, state_type: str = "zero") -> jnp.ndarray:
        """Create initial quantum state."""
        if state_type == "zero":
            state = jnp.zeros(self.dim, dtype=jnp.complex64)
            state = state.at[0].set(1.0)  # |00...0⟩
        elif state_type == "plus":
            # Uniform superposition |+⟩^⊗n
            state = jnp.ones(self.dim, dtype=jnp.complex64) / jnp.sqrt(self.dim)
        elif state_type == "random":
            # Random pure state
            rng_key = jax.random.PRNGKey(42)
            real_part = jax.random.normal(rng_key, (self.dim,))
            imag_part = jax.random.normal(rng_key, (self.dim,))
            state = real_part + 1j * imag_part
            state = state / jnp.sqrt(jnp.real(jnp.conj(state) @ state))
        else:
            raise ValueError(f"Unknown state type: {state_type}")

        return state
```

### Variational Quantum Eigensolver (VQE)
```python
# VQE implementation with JAX optimization
import optax

class VariationalQuantumEigensolver:
    """JAX-accelerated Variational Quantum Eigensolver."""

    def __init__(
        self,
        hamiltonian: jnp.ndarray,
        ansatz_circuit: callable,
        num_params: int,
        simulator: JAXQuantumSimulator
    ):
        self.hamiltonian = hamiltonian
        self.ansatz_circuit = ansatz_circuit
        self.num_params = num_params
        self.simulator = simulator

    @functools.partial(jax.jit, static_argnums=(0,))
    def vqe_cost_function(self, params: jnp.ndarray) -> float:
        """
        VQE cost function: energy expectation value.

        Args:
            params: Variational parameters

        Returns:
            Energy expectation value
        """
        # Create initial state
        state = self.simulator.create_initial_state("zero")

        # Apply ansatz circuit
        final_state = self.ansatz_circuit(state, params)

        # Compute energy expectation value
        energy = self.simulator.expectation_value(final_state, self.hamiltonian)

        return energy

    def optimize_vqe(
        self,
        initial_params: jnp.ndarray,
        num_iterations: int = 1000,
        learning_rate: float = 0.01
    ) -> Tuple[jnp.ndarray, List[float]]:
        """
        Optimize VQE parameters using gradient descent.

        Args:
            initial_params: Initial variational parameters
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimizer

        Returns:
            Optimal parameters and energy history
        """
        # Initialize optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(initial_params)

        params = initial_params
        energy_history = []

        @jax.jit
        def update_step(params, opt_state):
            """Single VQE optimization step."""
            energy, grads = jax.value_and_grad(self.vqe_cost_function)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, energy

        # Optimization loop
        for iteration in range(num_iterations):
            params, opt_state, energy = update_step(params, opt_state)
            energy_history.append(float(energy))

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Energy = {energy:.6f}")

        return params, energy_history

    @functools.partial(jax.jit, static_argnums=(0,))
    def hardware_efficient_ansatz(
        self,
        state: jnp.ndarray,
        params: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Hardware-efficient ansatz for VQE.

        Args:
            state: Initial quantum state
            params: Variational parameters [num_layers * num_qubits * 3]

        Returns:
            Final quantum state after ansatz application
        """
        num_qubits = self.simulator.num_qubits
        num_layers = len(params) // (num_qubits * 3)

        current_state = state
        param_idx = 0

        for layer in range(num_layers):
            # Single-qubit rotations
            for qubit in range(num_qubits):
                # RX rotation
                rx_gate = self.simulator.rx_gate(params[param_idx])
                current_state = self.simulator.apply_single_qubit_gate(
                    current_state, qubit, rx_gate
                )
                param_idx += 1

                # RY rotation
                ry_gate = self.simulator.ry_gate(params[param_idx])
                current_state = self.simulator.apply_single_qubit_gate(
                    current_state, qubit, ry_gate
                )
                param_idx += 1

                # RZ rotation
                rz_gate = self.simulator.rz_gate(params[param_idx])
                current_state = self.simulator.apply_single_qubit_gate(
                    current_state, qubit, rz_gate
                )
                param_idx += 1

            # Entangling gates (CNOT ladder)
            for qubit in range(num_qubits - 1):
                current_state = self.simulator.apply_two_qubit_gate(
                    current_state, qubit, qubit + 1, self.simulator.cnot_gate()
                )

        return current_state
```

### Quantum Approximate Optimization Algorithm (QAOA)
```python
# QAOA implementation for combinatorial optimization
class QuantumApproximateOptimization:
    """QAOA implementation for combinatorial optimization problems."""

    def __init__(
        self,
        cost_hamiltonian: jnp.ndarray,
        mixer_hamiltonian: jnp.ndarray,
        simulator: JAXQuantumSimulator,
        num_layers: int = 1
    ):
        self.cost_hamiltonian = cost_hamiltonian
        self.mixer_hamiltonian = mixer_hamiltonian
        self.simulator = simulator
        self.num_layers = num_layers

    @functools.partial(jax.jit, static_argnums=(0,))
    def qaoa_circuit(
        self,
        state: jnp.ndarray,
        gamma: jnp.ndarray,
        beta: jnp.ndarray
    ) -> jnp.ndarray:
        """
        QAOA circuit with alternating cost and mixer unitaries.

        Args:
            state: Initial quantum state
            gamma: Cost unitary parameters [num_layers]
            beta: Mixer unitary parameters [num_layers]

        Returns:
            Final quantum state after QAOA circuit
        """
        current_state = state

        for layer in range(self.num_layers):
            # Apply cost unitary: e^(-i*gamma*H_C)
            cost_unitary = self._matrix_exponential(-1j * gamma[layer] * self.cost_hamiltonian)
            current_state = cost_unitary @ current_state

            # Apply mixer unitary: e^(-i*beta*H_M)
            mixer_unitary = self._matrix_exponential(-1j * beta[layer] * self.mixer_hamiltonian)
            current_state = mixer_unitary @ current_state

        return current_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _matrix_exponential(self, matrix: jnp.ndarray) -> jnp.ndarray:
        """Compute matrix exponential using eigendecomposition."""
        eigenvals, eigenvecs = jnp.linalg.eigh(matrix)
        exp_eigenvals = jnp.exp(eigenvals)
        return eigenvecs @ jnp.diag(exp_eigenvals) @ jnp.conj(eigenvecs).T

    @functools.partial(jax.jit, static_argnums=(0,))
    def qaoa_cost_function(
        self,
        gamma: jnp.ndarray,
        beta: jnp.ndarray
    ) -> float:
        """
        QAOA cost function: expectation value of cost Hamiltonian.

        Args:
            gamma: Cost unitary parameters
            beta: Mixer unitary parameters

        Returns:
            Cost function value (to be minimized)
        """
        # Start with uniform superposition
        initial_state = self.simulator.create_initial_state("plus")

        # Apply QAOA circuit
        final_state = self.qaoa_circuit(initial_state, gamma, beta)

        # Compute cost expectation value
        cost = self.simulator.expectation_value(final_state, self.cost_hamiltonian)

        return cost

    def optimize_qaoa(
        self,
        num_iterations: int = 1000,
        learning_rate: float = 0.01
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[float]]:
        """
        Optimize QAOA parameters.

        Args:
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate

        Returns:
            Optimal gamma, beta parameters and cost history
        """
        # Initialize parameters
        rng_key = jax.random.PRNGKey(42)
        gamma = jax.random.uniform(rng_key, (self.num_layers,), minval=0, maxval=2*jnp.pi)
        beta = jax.random.uniform(rng_key, (self.num_layers,), minval=0, maxval=jnp.pi)

        # Optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init((gamma, beta))

        cost_history = []

        @jax.jit
        def update_step(gamma, beta, opt_state):
            """Single QAOA optimization step."""
            cost, (grad_gamma, grad_beta) = jax.value_and_grad(
                self.qaoa_cost_function, argnums=(0, 1)
            )(gamma, beta)

            updates, opt_state = optimizer.update((grad_gamma, grad_beta), opt_state)
            (gamma_updates, beta_updates) = updates

            gamma = optax.apply_updates(gamma, gamma_updates)
            beta = optax.apply_updates(beta, beta_updates)

            return gamma, beta, opt_state, cost

        # Optimization loop
        for iteration in range(num_iterations):
            gamma, beta, opt_state, cost = update_step(gamma, beta, opt_state)
            cost_history.append(float(cost))

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.6f}")

        return gamma, beta, cost_history

    def sample_solutions(
        self,
        gamma: jnp.ndarray,
        beta: jnp.ndarray,
        num_samples: int = 1000
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample solutions from optimized QAOA state.

        Args:
            gamma: Optimal gamma parameters
            beta: Optimal beta parameters
            num_samples: Number of measurement samples

        Returns:
            Bitstring samples and their frequencies
        """
        # Prepare QAOA state
        initial_state = self.simulator.create_initial_state("plus")
        final_state = self.qaoa_circuit(initial_state, gamma, beta)

        # Sample measurements
        rng_key = jax.random.PRNGKey(123)
        samples = []

        current_state = final_state
        for _ in range(num_samples):
            bitstring = []
            temp_state = current_state

            for qubit in range(self.simulator.num_qubits):
                rng_key, subkey = jax.random.split(rng_key)
                measurement, temp_state = self.simulator.measure_qubit(
                    temp_state, qubit, subkey
                )
                bitstring.append(measurement)

            samples.append(tuple(bitstring))
            # Reset state for next sample
            temp_state = final_state

        # Count frequencies
        unique_samples, counts = jnp.unique(
            jnp.array(samples), return_counts=True, axis=0
        )

        return unique_samples, counts
```

### Quantum Machine Learning with Neural Networks
```python
# Quantum neural networks and hybrid algorithms
class QuantumNeuralNetwork:
    """Quantum neural network with classical-quantum hybrid training."""

    def __init__(
        self,
        num_qubits: int,
        num_layers: int,
        classical_preprocessing: callable = None,
        classical_postprocessing: callable = None
    ):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.simulator = JAXQuantumSimulator(num_qubits)
        self.classical_preprocessing = classical_preprocessing
        self.classical_postprocessing = classical_postprocessing

        # Total number of parameters
        self.num_params = num_layers * num_qubits * 3  # RX, RY, RZ per qubit per layer

    @functools.partial(jax.jit, static_argnums=(0,))
    def quantum_layer(
        self,
        input_data: jnp.ndarray,
        params: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Single quantum layer with data encoding and parameterized gates.

        Args:
            input_data: Classical input data [feature_dim]
            params: Quantum layer parameters [num_qubits * 3]

        Returns:
            Quantum state after layer application
        """
        # Start with |0⟩ state
        state = self.simulator.create_initial_state("zero")

        # Data encoding (amplitude encoding for simplicity)
        if len(input_data) <= self.num_qubits:
            # Pad input data to match number of qubits
            padded_data = jnp.pad(
                input_data,
                (0, self.num_qubits - len(input_data)),
                mode='constant'
            )
        else:
            # Truncate input data
            padded_data = input_data[:self.num_qubits]

        # Encode data as rotation angles
        for i, data_point in enumerate(padded_data):
            ry_gate = self.simulator.ry_gate(data_point)
            state = self.simulator.apply_single_qubit_gate(state, i, ry_gate)

        # Apply parameterized gates
        param_idx = 0
        for qubit in range(self.num_qubits):
            # RX rotation
            rx_gate = self.simulator.rx_gate(params[param_idx])
            state = self.simulator.apply_single_qubit_gate(state, qubit, rx_gate)
            param_idx += 1

            # RY rotation
            ry_gate = self.simulator.ry_gate(params[param_idx])
            state = self.simulator.apply_single_qubit_gate(state, qubit, ry_gate)
            param_idx += 1

            # RZ rotation
            rz_gate = self.simulator.rz_gate(params[param_idx])
            state = self.simulator.apply_single_qubit_gate(state, qubit, rz_gate)
            param_idx += 1

        # Entangling gates
        for qubit in range(self.num_qubits - 1):
            state = self.simulator.apply_two_qubit_gate(
                state, qubit, qubit + 1, self.simulator.cnot_gate()
            )

        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def forward_pass(
        self,
        input_data: jnp.ndarray,
        params: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Forward pass through quantum neural network.

        Args:
            input_data: Input data [batch_size, feature_dim]
            params: All quantum parameters [num_layers, num_qubits * 3]

        Returns:
            Quantum circuit outputs [batch_size, num_qubits]
        """
        def single_sample_forward(sample):
            """Forward pass for single sample."""
            state = self.simulator.create_initial_state("zero")

            # Apply quantum layers sequentially
            for layer in range(self.num_layers):
                layer_params = params[layer]
                state = self.quantum_layer(sample, layer_params)

            # Measurement in computational basis
            measurements = []
            for qubit in range(self.num_qubits):
                # Compute expectation value of Z measurement
                z_observable = self._create_z_observable(qubit)
                expectation = self.simulator.expectation_value(state, z_observable)
                measurements.append(expectation)

            return jnp.array(measurements)

        # Vectorize over batch
        batch_outputs = jax.vmap(single_sample_forward)(input_data)
        return batch_outputs

    @functools.partial(jax.jit, static_argnums=(0, 2))
    def _create_z_observable(self, qubit: int) -> jnp.ndarray:
        """Create Z observable for single qubit measurement."""
        observable = jnp.eye(self.simulator.dim, dtype=jnp.complex64)

        for i in range(self.simulator.dim):
            # Check bit value of qubit position
            bit_value = (i >> qubit) & 1
            sign = 1.0 if bit_value == 0 else -1.0
            observable = observable.at[i, i].set(sign)

        return observable

    def train_qnn(
        self,
        train_data: jnp.ndarray,
        train_labels: jnp.ndarray,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.01
    ) -> Tuple[jnp.ndarray, List[float]]:
        """
        Train quantum neural network.

        Args:
            train_data: Training data [N, feature_dim]
            train_labels: Training labels [N, output_dim]
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Trained parameters and loss history
        """
        # Initialize parameters
        rng_key = jax.random.PRNGKey(42)
        params = jax.random.uniform(
            rng_key,
            (self.num_layers, self.num_qubits * 3),
            minval=0,
            maxval=2 * jnp.pi
        )

        # Optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

        loss_history = []

        @jax.jit
        def loss_fn(params, batch_data, batch_labels):
            """Loss function for QNN training."""
            predictions = self.forward_pass(batch_data, params)

            # Apply classical postprocessing if provided
            if self.classical_postprocessing is not None:
                predictions = jax.vmap(self.classical_postprocessing)(predictions)

            # Mean squared error loss
            loss = jnp.mean((predictions - batch_labels) ** 2)
            return loss

        @jax.jit
        def update_step(params, opt_state, batch_data, batch_labels):
            """Single training step."""
            loss, grads = jax.value_and_grad(loss_fn)(params, batch_data, batch_labels)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        # Training loop
        num_batches = len(train_data) // batch_size
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = batch_start + batch_size

                batch_data = train_data[batch_start:batch_end]
                batch_labels = train_labels[batch_start:batch_end]

                # Apply classical preprocessing if provided
                if self.classical_preprocessing is not None:
                    batch_data = jax.vmap(self.classical_preprocessing)(batch_data)

                params, opt_state, batch_loss = update_step(
                    params, opt_state, batch_data, batch_labels
                )
                epoch_loss += batch_loss

            avg_loss = epoch_loss / num_batches
            loss_history.append(float(avg_loss))

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

        return params, loss_history
```

### Quantum Error Correction Framework
```python
# Quantum error correction and noise simulation
class QuantumErrorCorrection:
    """Quantum error correction framework with noise modeling."""

    def __init__(self, code_type: str = "surface", code_distance: int = 3):
        self.code_type = code_type
        self.code_distance = code_distance
        self.num_physical_qubits = self._compute_num_physical_qubits()
        self.simulator = JAXQuantumSimulator(self.num_physical_qubits)

    def _compute_num_physical_qubits(self) -> int:
        """Compute number of physical qubits for given code."""
        if self.code_type == "surface":
            return self.code_distance ** 2
        elif self.code_type == "steane":
            return 7
        elif self.code_type == "shor":
            return 9
        else:
            raise ValueError(f"Unknown code type: {self.code_type}")

    @functools.partial(jax.jit, static_argnums=(0,))
    def apply_depolarizing_noise(
        self,
        state: jnp.ndarray,
        error_rate: float,
        rng_key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Apply depolarizing noise to quantum state.

        Args:
            state: Quantum state vector
            error_rate: Depolarizing error rate per qubit
            rng_key: Random number generator key

        Returns:
            Noisy quantum state
        """
        noisy_state = state

        for qubit in range(self.simulator.num_qubits):
            rng_key, subkey = jax.random.split(rng_key)

            # Sample error type: I, X, Y, Z with probabilities
            error_probs = jnp.array([
                1 - error_rate,  # Identity (no error)
                error_rate / 3,  # X error
                error_rate / 3,  # Y error
                error_rate / 3   # Z error
            ])

            error_type = jax.random.choice(
                subkey, jnp.array([0, 1, 2, 3]), p=error_probs
            )

            # Apply corresponding Pauli error
            error_gate = lax.switch(
                error_type,
                [
                    lambda: self.simulator.gates['I'],
                    lambda: self.simulator.gates['X'],
                    lambda: self.simulator.gates['Y'],
                    lambda: self.simulator.gates['Z']
                ]
            )

            noisy_state = self.simulator.apply_single_qubit_gate(
                noisy_state, qubit, error_gate
            )

        return noisy_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def syndrome_measurement(
        self,
        state: jnp.ndarray,
        stabilizer_generators: List[jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform syndrome measurement for error detection.

        Args:
            state: Quantum state (possibly with errors)
            stabilizer_generators: List of stabilizer generator operators

        Returns:
            Syndrome measurement results and post-measurement state
        """
        syndrome = []
        current_state = state

        for generator in stabilizer_generators:
            # Measure stabilizer generator
            eigenvalues, eigenvectors = jnp.linalg.eigh(generator)

            # Compute expectation value
            expectation = self.simulator.expectation_value(current_state, generator)

            # Convert to binary syndrome bit (±1 → 0/1)
            syndrome_bit = (1 - jnp.sign(jnp.real(expectation))) // 2
            syndrome.append(syndrome_bit)

        return jnp.array(syndrome), current_state

    def decode_syndrome(
        self,
        syndrome: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Decode syndrome to determine error correction.

        Args:
            syndrome: Syndrome measurement results

        Returns:
            Correction operations to apply
        """
        if self.code_type == "surface":
            return self._decode_surface_code(syndrome)
        elif self.code_type == "steane":
            return self._decode_steane_code(syndrome)
        else:
            raise NotImplementedError(f"Decoder for {self.code_type} not implemented")

    def _decode_surface_code(self, syndrome: jnp.ndarray) -> jnp.ndarray:
        """Decode surface code syndrome using minimum weight perfect matching."""
        # Simplified decoding for demonstration
        # In practice, this would use sophisticated graph algorithms

        # Find positions of syndrome violations
        violation_positions = jnp.where(syndrome == 1)[0]

        # Simple pairing strategy (nearest neighbor)
        corrections = jnp.zeros(self.num_physical_qubits, dtype=jnp.int32)

        for i in range(0, len(violation_positions), 2):
            if i + 1 < len(violation_positions):
                pos1 = violation_positions[i]
                pos2 = violation_positions[i + 1]

                # Apply correction along path between violations
                correction_path = self._find_correction_path(pos1, pos2)
                for qubit in correction_path:
                    corrections = corrections.at[qubit].set(1)

        return corrections

    def _decode_steane_code(self, syndrome: jnp.ndarray) -> jnp.ndarray:
        """Decode Steane code syndrome using lookup table."""
        # Steane code syndrome lookup table
        syndrome_table = {
            (0, 0, 0): [],  # No error
            (1, 0, 0): [0],  # X error on qubit 0
            (0, 1, 0): [1],  # X error on qubit 1
            (1, 1, 0): [2],  # X error on qubit 2
            # ... (complete table would have all 128 entries)
        }

        syndrome_tuple = tuple(syndrome)
        corrections = syndrome_table.get(syndrome_tuple, [])

        correction_vector = jnp.zeros(7, dtype=jnp.int32)
        for qubit in corrections:
            correction_vector = correction_vector.at[qubit].set(1)

        return correction_vector

    def _find_correction_path(self, pos1: int, pos2: int) -> List[int]:
        """Find correction path between two syndrome positions."""
        # Simplified path finding for surface code
        # This should implement proper graph algorithms for real codes
        path = []

        # Convert linear positions to 2D coordinates
        d = self.code_distance
        x1, y1 = pos1 // d, pos1 % d
        x2, y2 = pos2 // d, pos2 % d

        # Simple Manhattan path
        current_x, current_y = x1, y1
        while current_x != x2 or current_y != y2:
            if current_x < x2:
                current_x += 1
            elif current_x > x2:
                current_x -= 1
            elif current_y < y2:
                current_y += 1
            else:
                current_y -= 1

            path.append(current_x * d + current_y)

        return path
```

## Integration with Scientific Workflow

### Quantum-Classical Hybrid Algorithms
- **Variational Methods**: VQE, QAOA, and quantum machine learning with classical optimization
- **Error Mitigation**: Zero noise extrapolation and symmetry verification
- **Gradient Computation**: Parameter-shift rules and finite difference methods

### High-Performance Quantum Simulation
- **GPU Acceleration**: Optimized tensor operations and state vector simulation
- **Distributed Computing**: Multi-device quantum circuit simulation
- **Memory Optimization**: Efficient state representation and compression techniques

### Quantum Algorithm Applications
- **Chemistry**: Molecular electronic structure and reaction dynamics
- **Optimization**: Portfolio optimization and logistics problems
- **Machine Learning**: Quantum feature maps and kernel methods

## Usage Examples

### VQE for Molecular Ground State
```python
# Find ground state of H2 molecule
# Create molecular Hamiltonian (simplified)
h2_hamiltonian = create_h2_hamiltonian(bond_length=0.74)

# Setup VQE
simulator = JAXQuantumSimulator(num_qubits=4)
vqe = VariationalQuantumEigensolver(
    hamiltonian=h2_hamiltonian,
    ansatz_circuit=lambda state, params: hardware_efficient_ansatz(state, params),
    num_params=12,
    simulator=simulator
)

# Optimize
initial_params = jax.random.uniform(jax.random.PRNGKey(42), (12,), minval=0, maxval=2*jnp.pi)
optimal_params, energy_history = vqe.optimize_vqe(initial_params)

print(f"Ground state energy: {energy_history[-1]:.6f} Hartree")
```

### QAOA for MaxCut Problem
```python
# Solve MaxCut on 4-node graph
cost_hamiltonian = create_maxcut_hamiltonian(graph_adjacency_matrix)
mixer_hamiltonian = create_x_mixer(num_qubits=4)

qaoa = QuantumApproximateOptimization(
    cost_hamiltonian=cost_hamiltonian,
    mixer_hamiltonian=mixer_hamiltonian,
    simulator=JAXQuantumSimulator(4),
    num_layers=2
)

gamma_opt, beta_opt, cost_history = qaoa.optimize_qaoa()
solutions, frequencies = qaoa.sample_solutions(gamma_opt, beta_opt, num_samples=1000)

print(f"Best solution frequency: {max(frequencies)}")
```

### Quantum Neural Network Training
```python
# Train QNN for binary classification
qnn = QuantumNeuralNetwork(num_qubits=4, num_layers=3)

# Generate synthetic data
X_train, y_train = generate_classification_data(n_samples=1000, n_features=4)

# Train QNN
trained_params, loss_history = qnn.train_qnn(
    X_train, y_train, num_epochs=100, learning_rate=0.01
)

print(f"Final training loss: {loss_history[-1]:.6f}")
```

This expert provides comprehensive JAX-based quantum computing capabilities with variational algorithms, error correction, and machine learning integration for scientific applications.