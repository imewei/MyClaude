# Quantum Computing with JAX

## QAOA (Quantum Approximate Optimization Algorithm)

Solve combinatorial optimization problems:

```python
def qaoa_circuit(params, graph, p_layers):
    """QAOA for MaxCut problem"""
    n_qubits = len(graph.nodes)
    qubits = cirq.LineQubit.range(n_qubits)

    circuit = cirq.Circuit()

    # Initial superposition
    circuit.append([cirq.H(q) for q in qubits])

    # p layers of cost + mixer Hamiltonians
    for layer in range(p_layers):
        # Cost Hamiltonian (graph edges)
        gamma = params[2*layer]
        for i, j in graph.edges:
            circuit.append(cirq.ZZ(qubits[i], qubits[j])**(gamma / jnp.pi))

        # Mixer Hamiltonian (X rotations)
        beta = params[2*layer + 1]
        circuit.append([cirq.rx(2*beta)(q) for q in qubits])

    return circuit

@jax.jit
def qaoa_objective(params, graph):
    """Expectation value of MaxCut cost"""
    circuit = qaoa_circuit(params, graph, p_layers=3)
    state = simulate_circuit(circuit)

    # Compute <C> = Σ <ZᵢZⱼ> for edges (i,j)
    cost = 0.0
    for i, j in graph.edges:
        expectation_ZiZj = measure_pauli_string(state, f'Z{i}Z{j}')
        cost += 0.5 * (1 - expectation_ZiZj)

    return cost

# Optimize with JAX
params = jax.random.uniform(key, (2*p_layers,)) * 2*jnp.pi
optimizer = optax.adam(0.01)
opt_state = optimizer.init(params)

for step in range(1000):
    cost, grads = jax.value_and_grad(qaoa_objective)(params, graph)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
```

## Quantum Machine Learning

Variational quantum classifier:

```python
class QuantumClassifier(nnx.Module):
    def __init__(self, n_qubits=4, n_layers=3, *, rngs: nnx.Rngs):
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Variational parameters
        self.theta = nnx.Param(jax.random.normal(rngs(), (n_layers, n_qubits)))
        self.phi = nnx.Param(jax.random.normal(rngs(), (n_layers, n_qubits)))

    def __call__(self, x):
        """x: classical input features"""
        qubits = cirq.LineQubit.range(self.n_qubits)
        circuit = cirq.Circuit()

        # Feature encoding (angle encoding)
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.ry(x[i % len(x)])(qubit))

        # Variational layers
        for layer in range(self.n_layers):
            # Rotation layer
            for i, qubit in enumerate(qubits):
                circuit.append(cirq.ry(self.theta[layer, i])(qubit))
                circuit.append(cirq.rz(self.phi[layer, i])(qubit))

            # Entangling layer
            for i in range(self.n_qubits - 1):
                circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))

        # Measure in Z basis
        state = simulate(circuit)
        measurement = jnp.array([measure_Z(state, qubit) for qubit in qubits])

        # Classical post-processing
        return jnp.tanh(jnp.sum(measurement))

# Training
model = QuantumClassifier(rngs=nnx.Rngs(0))

def loss_fn(model, x, y):
    pred = jax.vmap(model)(x)
    return jnp.mean((pred - y)**2)

optimizer = nnx.Optimizer(model, optax.adam(0.01))

for epoch in range(1000):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x_train, y_train)
    optimizer.update(grads)
```

## Error Mitigation

Zero-noise extrapolation:

```python
def zero_noise_extrapolation(circuit, noise_factors=[1.0, 1.5, 2.0]):
    """
    Run circuit at different noise levels and extrapolate to zero noise
    """
    expectations = []

    for factor in noise_factors:
        # Scale noise by factor (pulse stretching)
        noisy_circuit = scale_noise(circuit, factor)
        result = execute_on_hardware(noisy_circuit)
        expectations.append(result)

    # Polynomial extrapolation to zero
    poly = jnp.polyfit(noise_factors, expectations, deg=2)
    zero_noise_value = jnp.polyval(poly, 0.0)

    return zero_noise_value
```
