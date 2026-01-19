# JAX Core Programming Assets

This directory contains visual assets and diagrams for JAX Core Programming workflows.

## Architecture Diagrams

### JAX Transformation Stack
```
┌─────────────────────────────────────────┐
│         User Function f(x)              │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│    grad(f) - Automatic Differentiation  │
│         ∂f/∂x via reverse-mode AD       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│    vmap(grad(f)) - Vectorization        │
│     Batch processing over inputs        │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│    jit(vmap(grad(f))) - Compilation     │
│      XLA compilation to machine code    │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│    pmap/shard - Multi-Device            │
│     Distribution across GPUs/TPUs       │
└─────────────────────────────────────────┘
```

### Training Pipeline Flow

```
Data → Preprocessing → Batching → Model → Loss → Gradients → Update
  │                      │          │       │       │          │
  └──────────────────────┼──────────┼───────┼───────┼──────────┘
                         │          │       │       │
                    [Flax NNX] [Optax Loss] [Grad] [Optax]
                         │                           │
                         └─────── Orbax Checkpoint ──┘
```

### Device Mesh (2D Parallelism)

```
           Model Dimension
              (axis 1)
                  │
      ┌───────────┼───────────┐
      │     GPU 0 │ GPU 1     │
Data  ├───────────┼───────────┤
(axis │     GPU 2 │ GPU 3     │
 0)   ├───────────┼───────────┤
      │     GPU 4 │ GPU 5     │
      ├───────────┼───────────┤
      │     GPU 6 │ GPU 7     │
      └───────────────────────┘

Data Sharding: P('data', None)
Model Sharding: P(None, 'model')
Full Sharding: P('data', 'model')
```

## Performance Charts

### Speedup by Transformation

| Transformation | Expected Speedup | Best For |
|----------------|------------------|----------|
| None (Baseline) | 1x | Reference |
| jit | 10-100x | All code paths |
| vmap | 2-10x | Batch processing |
| jit + vmap | 100-1000x | Production pipelines |
| Multi-device | N x (N devices) | Large-scale training |

### Memory Optimization

| Technique | Memory Reduction | Compute Overhead |
|-----------|------------------|------------------|
| Baseline | 1x | 0% |
| Rematerialization | 2-5x | 30-50% |
| Mixed Precision (bf16) | 2x | -50% (faster!) |
| Gradient Accumulation | 1-4x | Proportional |
| Sharding | Per-device | Communication cost |

## Workflow Diagrams

### Quick Prototyping Workflow

```
1. Pure Function
     ↓
2. Add jit()
     ↓
3. Add vmap()
     ↓
4. Add grad()
     ↓
5. Compose: jit(vmap(grad(fn)))
     ↓
6. Training Loop
```

### Production Training Workflow

```
1. Flax NNX Model Definition
     ↓
2. Optax Optimizer + LR Schedule
     ↓
3. Orbax Checkpointing Setup
     ↓
4. JIT-compiled Training Step
     ↓
5. Training Loop with Checkpoints
     ↓
6. Evaluation & Validation
```

### Multi-Device Workflow

```
1. Create Device Mesh
     ↓
2. Define Sharding Strategies
     ↓
3. Shard Data & Parameters
     ↓
4. JIT-compiled Distributed Ops
     ↓
5. Automatic Communication
     ↓
6. Gather Results
```

### Bayesian Inference Workflow

```
1. Define Probabilistic Model
     ↓
2. MCMC with NUTS (JAX-accelerated)
     ↓
3. Collect Posterior Samples
     ↓
4. Posterior Predictive (vmap)
     ↓
5. Uncertainty Quantification
```

## Usage

These ASCII diagrams provide visual reference for JAX workflows. For high-resolution
diagrams, consider exporting these to GraphViz, Mermaid, or draw.io formats.

## Generating Visualizations

To generate performance charts from your code:

```python
import matplotlib.pyplot as plt
import numpy as np

# Example: Speedup chart
transformations = ['Baseline', 'jit', 'vmap', 'jit+vmap', 'multi-device']
speedups = [1, 50, 5, 250, 1000]

plt.figure(figsize=(10, 6))
plt.bar(transformations, speedups, log=True)
plt.ylabel('Speedup (log scale)')
plt.title('JAX Transformation Speedups')
plt.savefig('speedup_chart.png')
```

## References

- JAX Documentation: https://jax.readthedocs.io/
- Flax Documentation: https://flax.readthedocs.io/
- Optax Documentation: https://optax.readthedocs.io/
