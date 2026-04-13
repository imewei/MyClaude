---
name: deep-learning-hub
description: Meta-orchestrator for deep learning. Routes to architecture design, mathematical foundations, training diagnostics, experimentation, and advanced systems skills. Use when designing neural architectures, deriving backpropagation, diagnosing training failures (loss divergence, gradient explosion/vanishing), running ablation studies, hyperparameter search, or building large-scale distributed DL systems.
---

# Deep Learning Hub

Orchestrator for deep learning. Routes problems to the appropriate specialized skill.

## Expert Agent

- **`neural-network-master`**: Specialist for deep learning theory, architecture design, and training optimization.
  - *Location*: `plugins/science-suite/agents/neural-network-master.md`
  - *Capabilities*: Architecture design, mathematical foundations, training diagnostics, large-scale systems, and advanced ML research.

## Core Skills

### [Deep Learning](../deep-learning/SKILL.md)
Core deep learning: feedforward networks, backpropagation, regularization, and standard training workflows.

### [Neural Architecture Patterns](../neural-architecture-patterns/SKILL.md)
Architecture design: CNNs, RNNs, attention, Transformers, diffusion models, and normalizing flows.

### [Neural Network Mathematics](../neural-network-mathematics/SKILL.md)
Mathematical foundations: universal approximation, optimization landscapes, generalization theory, and information geometry.

### [Training Diagnostics](../training-diagnostics/SKILL.md)
Diagnosing training failures: loss curves, gradient explosion/vanishing, learning rate tuning, and regularization.

### [Deep Learning Experimentation](../deep-learning-experimentation/SKILL.md)
Experiment design: ablations, hyperparameter search, reproducibility, and benchmark evaluation.

### [Advanced ML Systems](../advanced-ml-systems/SKILL.md)
Large-scale systems: distributed training, mixed precision, gradient checkpointing, and training-time scaling. For post-training inference optimization (quantization, pruning, ONNX), see the `ml-deployment` hub.

## Routing Decision Tree

```
What is the deep learning task?
|
+-- Build a standard neural network / training loop?
|   --> deep-learning
|
+-- Design or choose an architecture?
|   --> neural-architecture-patterns
|
+-- Understand the math behind a method?
|   --> neural-network-mathematics
|
+-- Diagnose training instability / poor convergence?
|   --> training-diagnostics
|
+-- Design experiments / ablations / hyperparameter search?
|   --> deep-learning-experimentation
|
+-- Scale to multi-GPU / distributed training?
    --> advanced-ml-systems
    (for post-training inference optimization, see ml-deployment hub)
```

## Skill Selection Table

| Task | Skill |
|------|-------|
| Standard training workflows | `deep-learning` |
| CNN, Transformer, diffusion | `neural-architecture-patterns` |
| Optimization landscapes, generalization | `neural-network-mathematics` |
| Loss divergence, gradient issues | `training-diagnostics` |
| Ablations, HPO, benchmarks | `deep-learning-experimentation` |
| Distributed training, multi-GPU scaling | `advanced-ml-systems` |

## Checklist

- [ ] Use routing tree to select the most specific sub-skill
- [ ] Verify baseline implementation is correct before designing ablations
- [ ] Check gradient norms at each layer before scaling model size
- [ ] Use mixed precision (bf16/fp16) only after validating numerical stability
- [ ] Ensure experiment seeds are fixed and logged for reproducibility
- [ ] Validate architecture math matches implementation (dimensions, masking)
- [ ] Profile memory usage before enabling gradient checkpointing
- [ ] For Julia-based deep learning (Flux.jl/Lux.jl), prefer the `julia-ml-and-dl` hub
