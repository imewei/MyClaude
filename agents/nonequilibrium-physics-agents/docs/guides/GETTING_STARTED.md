# Getting Started Guide

**Nonequilibrium Physics Optimal Control Framework**

Welcome! This guide will help you get started with the framework in minutes.

---

## What Can You Do?

This framework enables:
- **Classical Optimal Control**: LQR, MPC, trajectory optimization
- **ML-Based Control**: Neural networks, reinforcement learning, PINNs
- **Quantum Control**: Quantum state manipulation, gate optimization
- **HPC Scalability**: Distributed execution on clusters
- **GPU Acceleration**: 30-50x speedup for large problems

---

## Installation

### Quick Install

```bash
pip install nonequilibrium-control
```

### With GPU Support

```bash
pip install "nonequilibrium-control[gpu]"
```

### Full Installation

```bash
pip install "nonequilibrium-control[all]"
```

---

## Your First Example

### Example 1: Linear Quadratic Regulator (LQR)

```python
import numpy as np
from scipy.linalg import solve_continuous_are

# Define system: dx/dt = Ax + Bu
A = np.array([[0, 1], [-1, -0.5]])
B = np.array([[0], [1]])

# Cost matrices
Q = np.eye(2)
R = np.array([[1.0]])

# Solve Riccati equation
P = solve_continuous_are(A, B, Q, R)

# Optimal control gain
K = np.linalg.solve(R, B.T @ P)

# Control law: u = -Kx
x = np.array([1.0, 0.5])
u = -K @ x

print(f"Optimal control: {u}")
```

### Example 2: Model Predictive Control

```python
from hpc.parallel import ParameterSpec, GridSearch

# Define parameters to optimize
specs = [
    ParameterSpec("controller_gain", "continuous", 0.1, 10.0),
    ParameterSpec("horizon", "integer", 5, 20)
]

# Grid search
sweep = GridSearch(specs, points_per_dim=10)
samples = sweep.generate_samples()

# Evaluate each configuration
def evaluate_controller(params):
    gain = params["controller_gain"]
    horizon = params["horizon"]
    # Simulate and return cost
    cost = simulate_mpc(gain, horizon)
    return cost

results = [evaluate_controller(s) for s in samples]
best = samples[np.argmin(results)]

print(f"Best parameters: {best}")
```

### Example 3: Neural Network Control

```python
import numpy as np

# Simple neural controller (mock implementation)
class NeuralController:
    def __init__(self, state_dim, control_dim):
        self.state_dim = state_dim
        self.control_dim = control_dim

        # Initialize weights
        self.W1 = np.random.randn(state_dim, 32) * 0.1
        self.W2 = np.random.randn(32, control_dim) * 0.1

    def __call__(self, state):
        # Forward pass
        h = np.tanh(state @ self.W1)
        u = h @ self.W2
        return u

# Create controller
controller = NeuralController(state_dim=4, control_dim=2)

# Use controller
state = np.random.randn(4)
control = controller(state)

print(f"Control output: {control}")
```

---

## Core Concepts

### 1. Optimal Control Problem

Minimize cost function:
```
J = ∫₀ᵀ L(x, u, t) dt
```

Subject to dynamics:
```
dx/dt = f(x, u, t)
```

### 2. Solution Methods

**Classical**:
- Dynamic Programming
- Pontryagin Maximum Principle
- Direct Methods (collocation, shooting)

**Machine Learning**:
- Neural network policies
- Reinforcement learning (PPO, SAC)
- Physics-informed neural networks

### 3. Key Components

**Systems**: Dynamics models (linear, nonlinear, quantum)
**Solvers**: Optimization algorithms
**Controllers**: Control policies (classical, neural)
**HPC**: Distributed execution infrastructure

---

## Common Workflows

### Workflow 1: Parameter Tuning

```python
from hpc.parallel import RandomSearch, ParameterSpec

# 1. Define parameter space
specs = [
    ParameterSpec("learning_rate", "continuous", 0.001, 0.1, log_scale=True),
    ParameterSpec("hidden_size", "integer", 16, 128)
]

# 2. Random search
sweep = RandomSearch(specs, n_samples=50, seed=42)
samples = sweep.generate_samples()

# 3. Evaluate
results = []
for params in samples:
    performance = train_and_evaluate(params)
    results.append({"params": params, "value": performance})

# 4. Find best
best = min(results, key=lambda x: x["value"])
print(f"Best configuration: {best['params']}")
print(f"Best performance: {best['value']}")
```

### Workflow 2: Distributed Training

```python
# Requires Dask
try:
    from hpc.distributed import create_local_cluster, distribute_computation

    # Create cluster
    cluster = create_local_cluster(n_workers=4)

    # Define training function
    def train_model(config):
        # Train and return performance
        return performance

    # Distribute across workers
    configs = generate_configurations(n=100)
    results = distribute_computation(train_model, configs, cluster)

    cluster.close()

except ImportError:
    print("Dask not installed. Install with: pip install dask[distributed]")
```

### Workflow 3: GPU Acceleration

```python
# Requires JAX
try:
    import jax
    import jax.numpy as jnp

    # Define computation on GPU
    @jax.jit
    def compute_cost(state, control):
        return jnp.sum(state**2) + jnp.sum(control**2)

    # Move data to GPU
    state_gpu = jnp.array([1.0, 2.0, 3.0])
    control_gpu = jnp.array([0.5])

    # Compute
    cost = compute_cost(state_gpu, control_gpu)
    print(f"Cost: {cost}")

except ImportError:
    print("JAX not installed. Install with: pip install jax[cuda]")
```

---

## Performance Tips

### 1. Use GPU for Large Problems

```python
# Enable GPU (requires JAX)
import os
os.environ['JAX_PLATFORM_NAME'] = 'gpu'

# Verify
import jax
print(f"GPU available: {jax.devices('gpu')}")
```

### 2. Parallelize Parameter Sweeps

```python
# Use distributed execution
from hpc.distributed import create_local_cluster

cluster = create_local_cluster(n_workers=8)
# ... use cluster for parallel evaluation
cluster.close()
```

### 3. Profile Performance

```python
from ml_optimal_control.performance import Benchmarker

benchmarker = Benchmarker()
result = benchmarker.benchmark(your_function, num_iterations=100)

print(f"Mean time: {result.mean_time:.4f}s")
```

### 4. Use Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_computation(x):
    # Will be cached for repeated calls
    return complex_function(x)
```

---

## Next Steps

### Tutorials
- **Basic Tutorial**: `docs/tutorials/basic_lqr.md`
- **Advanced Tutorial**: `docs/tutorials/neural_control.md`
- **HPC Tutorial**: `docs/tutorials/distributed_sweep.md`

### Examples
Browse `examples/` directory:
- `examples/lqr_control.py` - Linear quadratic regulator
- `examples/neural_control.py` - Neural network controller
- `examples/quantum_control.py` - Quantum state control
- `examples/parameter_sweep.py` - Hyperparameter optimization

### Documentation
- **API Reference**: `docs/api/`
- **Deployment Guide**: `DEPLOYMENT.md`
- **Troubleshooting**: `DEPLOYMENT.md#troubleshooting`

### Community
- **GitHub**: https://github.com/your-org/nonequilibrium-control
- **Discussions**: https://github.com/your-org/nonequilibrium-control/discussions
- **Issues**: https://github.com/your-org/nonequilibrium-control/issues

---

## FAQ

### Q: What problems can this framework solve?

**A**: Any optimal control problem including:
- Trajectory optimization
- Model predictive control
- Reinforcement learning
- Quantum control
- Stochastic control

### Q: Do I need GPU for good performance?

**A**: No, but GPU provides 30-50x speedup for large problems (n > 100 states).

### Q: Can I use this on HPC clusters?

**A**: Yes! Full support for SLURM and PBS schedulers. See `DEPLOYMENT.md#hpc-cluster-deployment`.

### Q: How do I cite this framework?

**A**:
```bibtex
@software{nonequilibrium_control,
  title={Nonequilibrium Physics Optimal Control Framework},
  author={Nonequilibrium Physics Agents Team},
  year={2025},
  version={1.0.0},
  url={https://github.com/your-org/nonequilibrium-control}
}
```

### Q: Where can I get help?

**A**:
- Check documentation: https://docs.nonequilibrium-control.org
- Ask on GitHub Discussions
- Open an issue for bugs
- Email: support@nonequilibrium-control.org

---

## Quick Reference

### Common Imports

```python
# Core functionality
import numpy as np
from scipy.linalg import solve_continuous_are

# Parameter sweeps
from hpc.parallel import ParameterSpec, GridSearch, RandomSearch

# Performance tools
from ml_optimal_control.performance import Benchmarker, Timer

# Distributed computing (optional)
from hpc.distributed import create_local_cluster, distribute_computation

# GPU acceleration (optional)
import jax
import jax.numpy as jnp
```

### Command Line Tools

```bash
# Run parameter sweep
python -m nonequilibrium_control.hpc.sweep --config config.yaml

# Run benchmarks
python run_benchmarks.py --all --report

# Run tests
pytest tests/ -v
```

---

**Ready to dive deeper?** Check out the [tutorials](docs/tutorials/) or [API documentation](docs/api/)!

**Version**: 1.0.0
**Last Updated**: 2025-10-01
