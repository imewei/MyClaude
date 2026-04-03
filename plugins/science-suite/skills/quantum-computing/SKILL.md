---
name: quantum-computing
description: "Implement quantum computing algorithms with Qiskit, Cirq, and PennyLane including quantum circuits, variational algorithms (VQE, QAOA), quantum simulation, and quantum machine learning. Use when building quantum circuits, implementing variational algorithms, or exploring quantum-classical hybrid methods."
---

# Quantum Computing

## Expert Agent

For quantum simulations, numerical methods, and high-performance computing, delegate to:

- **`simulation-expert`**: Expert in physics simulations, numerical methods, and HPC scaling.
  - *Location*: `plugins/science-suite/agents/simulation-expert.md`

## Qiskit Fundamentals

```python
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

sampler = StatevectorSampler()
result = sampler.run([qc], shots=1024).result()
counts = result[0].data.c.get_counts()  # {'00': ~512, '11': ~512}
```

### Gate Reference

| Gate | Qiskit | Purpose |
|------|--------|---------|
| Hadamard | `qc.h(q)` | Superposition |
| CNOT | `qc.cx(c, t)` | Entanglement |
| Pauli-X/Z | `qc.x(q)` / `qc.z(q)` | Bit/phase flip |
| Ry | `qc.ry(theta, q)` | Parameterized rotation |
| Toffoli | `qc.ccx(c1, c2, t)` | 3-qubit AND |

## VQE (Variational Quantum Eigensolver)

```python
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize
import numpy as np

hamiltonian = SparsePauliOp.from_list([
    ("II", -1.0523), ("IZ", 0.3979), ("ZI", -0.3979),
    ("ZZ", -0.0112), ("XX", 0.1809),
])
ansatz = EfficientSU2(num_qubits=2, reps=1, entanglement="linear")
estimator = StatevectorEstimator()

def cost(params):
    bound = ansatz.assign_parameters(params)
    return estimator.run([(bound, hamiltonian)]).result()[0].data.evs

x0 = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
result = minimize(cost, x0, method="COBYLA", options={"maxiter": 500})
print(f"Ground state energy: {result.fun:.6f} Ha")
```

## QAOA (MaxCut)

```python
from qiskit.circuit import QuantumCircuit, Parameter

def build_qaoa(num_qubits: int, edges: list[tuple[int, int]], p: int) -> QuantumCircuit:
    gammas = [Parameter(f"g_{i}") for i in range(p)]
    betas = [Parameter(f"b_{i}") for i in range(p)]
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    for layer in range(p):
        for i, j in edges:
            qc.cx(i, j); qc.rz(2 * gammas[layer], j); qc.cx(i, j)
        for q in range(num_qubits):
            qc.rx(2 * betas[layer], q)
    qc.measure_all()
    return qc
```

## Quantum ML (PennyLane)

```python
import pennylane as qml
from pennylane import numpy as pnp

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def quantum_classifier(inputs, weights):
    for i in range(4):
        qml.RY(inputs[i], wires=i)
    for layer_w in weights:
        for i in range(4):
            qml.RY(layer_w[i, 0], wires=i)
            qml.RZ(layer_w[i, 1], wires=i)
        for i in range(3):
            qml.CNOT(wires=[i, i + 1])
    return qml.expval(qml.PauliZ(0))

weights = pnp.random.uniform(-pnp.pi, pnp.pi, (3, 4, 2), requires_grad=True)
opt = qml.AdamOptimizer(stepsize=0.1)
```

## Error Mitigation

### Zero-Noise Extrapolation

Run the circuit at multiple noise levels (fold gates to amplify noise), then extrapolate measured expectation values to the zero-noise limit using Richardson extrapolation.

### Noise Models (Qiskit Aer)

```python
from qiskit_aer.noise import NoiseModel, depolarizing_error

noise = NoiseModel()
noise.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ["h", "rx", "ry"])
noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx"])
```

## Algorithm Selection

| Problem | Algorithm | Qubits | Advantage |
|---------|-----------|--------|-----------|
| Ground state | VQE | 2N orbitals | Noise tolerant |
| Combinatorial | QAOA | N variables | Hybrid |
| Classification | Variational QC | log N | Kernel methods |
| Simulation | Trotterization | N sites | Exponential speedup |
| Search | Grover | log N | Quadratic speedup |

## Production Checklist

- [ ] Choose simulator vs. hardware based on qubit count and noise tolerance
- [ ] Transpile circuits to target device topology
- [ ] Benchmark ansatz expressibility before committing to VQE depth
- [ ] Implement error mitigation (ZNE, measurement correction) for hardware
- [ ] Set shot budget: 1024 for prototyping, 8192+ for production
- [ ] Profile classical optimizer convergence (COBYLA, SPSA)
- [ ] Version-lock Qiskit/PennyLane/Cirq to avoid API breakage
- [ ] Log circuit metadata: depth, gate count, parameters, expectation values
