---
name: deep-learning
description: Core deep-learning implementation skill for standard neural networks, training loops, regularization, loss selection, and framework-neutral PyTorch/JAX/TensorFlow patterns. Use when building baseline models, implementing custom layers or losses, validating data/shape flow, or explaining feedforward/backprop training mechanics. For architecture selection, math derivations, diagnostics, experiments, or distributed training, enter through deep-learning-hub.
---

# Deep Learning Mastery

Comprehensive guide to deep learning theory, architecture, and practice.

## Expert Agents

For deep learning tasks, delegate to the specialized experts:

- **`neural-network-master`**: Theory, architecture design, and training diagnostics.
  - *Location*: `plugins/science-suite/agents/neural-network-master.md`
  - *Capabilities*: Architecture selection, loss landscape analysis, gradient debugging.
- **`ml-expert`**: Distributed training implementation and hardware optimization.
  - *Location*: `plugins/science-suite/agents/ml-expert.md`
  - *Capabilities*: DDP/FSDP setup, multi-node scaling.
- **`julia-ml-hpc`**: Julia DL implementation with Lux.jl/Flux.jl.
  - *Location*: `plugins/science-suite/agents/julia-ml-hpc.md`
  - *Capabilities*: Lux.jl training, GPU acceleration, Julia neural architectures.
  - *Julia skill*: See `julia-neural-networks` for Julia-specific deep learning.

## Core Skills

### [Neural Architecture Patterns](../neural-architecture-patterns/SKILL.md)
Design patterns for Transformers, CNNs, ViTs, and hybrid architectures.

### [Neural Network Mathematics](../neural-network-mathematics/SKILL.md)
Foundational math, calculus, optimization theory, and information theory.

### [Training Diagnostics](../training-diagnostics/SKILL.md)
Debugging vanishing gradients, loss instability, dead ReLUs, and convergence issues.

### [Model Optimization](../model-optimization-deployment/SKILL.md)
Quantization, pruning, knowledge distillation, and efficient inference.

### Research Paper Implementation *(moved to research-suite)*
Translating academic papers into working code in JAX or PyTorch. See `research-paper-implementation` in the `research-suite` plugin (`plugins/research-suite/skills/research-paper-implementation/`).

### [Deep Learning Experimentation](../deep-learning-experimentation/SKILL.md)
Systematic workflows for training, ablation, and hyperparameter tuning.

## Routing Decision Tree

```
What is the deep-learning need?
|
+-- Build a baseline model, layer, loss, or training loop?
|   --> stay in deep-learning
|
+-- Choose CNN/Transformer/GNN/diffusion architecture?
|   --> neural-architecture-patterns
|
+-- Derive backpropagation, Jacobians, Hessians, or convergence math?
|   --> neural-network-mathematics
|
+-- Diagnose NaN loss, gradient explosion/vanishing, dead ReLUs, or plateaus?
|   --> training-diagnostics
|
+-- Plan ablations, hyperparameter search, or reproducible runs?
|   --> deep-learning-experimentation
|
+-- Optimize inference with quantization/pruning/ONNX/TensorRT?
    --> model-optimization-deployment
```

## Frameworks

- **PyTorch**: Standard for research and production.
- **JAX (Flax/Equinox)**: High-performance research and scientific computing.
- **TensorFlow/Keras**: Legacy support and rapid prototyping.

## Checklist

- [ ] Verify architecture choice matches problem type (CNN for spatial, Transformer for sequential, MLP for tabular)
- [ ] Confirm weight initialization scheme is appropriate (He for ReLU, Xavier for tanh/sigmoid)
- [ ] Check that loss function aligns with the task (cross-entropy for classification, MSE for regression)
- [ ] Validate gradient flow through all layers using gradient norm diagnostics
- [ ] Ensure learning rate schedule includes warmup for large-batch training
- [ ] Run single-batch overfit test before full training to verify model capacity
- [ ] Confirm data pipeline produces correctly shaped, normalized, and shuffled batches
- [ ] Check for numerical stability (NaN/Inf) in forward and backward passes
