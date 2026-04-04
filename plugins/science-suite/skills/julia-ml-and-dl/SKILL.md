---
name: julia-ml-and-dl
description: Meta-orchestrator for Julia ML and deep learning. Routes to neural networks, architectures, training diagnostics, AD backends, GPU kernels, GNNs, RL, pipelines, and deployment skills. Use when training neural networks in Julia with Lux.jl, designing architectures, debugging training, writing GPU kernels, or deploying Julia ML models.
---

# Julia ML and Deep Learning

Orchestrator for Julia-based machine learning and deep learning. Routes problems to the appropriate specialized skill.

## Expert Agent

- **`julia-ml-hpc`**: Specialist for Julia ML, Flux.jl, Lux.jl, and GPU-accelerated deep learning.
  - *Location*: `plugins/science-suite/agents/julia-ml-hpc.md`
  - *Capabilities*: Neural network design, AD backends, GPU kernels, GNNs, RL, and model deployment.

## Core Skills

### [Julia Neural Networks](../julia-neural-networks/SKILL.md)
Flux.jl and Lux.jl: model definition, layers, training loops, and callbacks.

### [Julia Neural Architectures](../julia-neural-architectures/SKILL.md)
CNNs, RNNs, Transformers, and custom architectures in Flux.jl / Lux.jl.

### [Julia Training Diagnostics](../julia-training-diagnostics/SKILL.md)
Loss curves, gradient norms, learning rate finders, and convergence monitoring.

### [Julia AD Backends](../julia-ad-backends/SKILL.md)
Zygote.jl, Enzyme.jl, ForwardDiff.jl: selecting and debugging AD backends.

### [Julia GPU Kernels](../julia-gpu-kernels/SKILL.md)
CUDA.jl and KernelAbstractions.jl: writing and profiling custom GPU kernels.

### [Julia Graph Neural Networks](../julia-graph-neural-networks/SKILL.md)
GraphNeuralNetworks.jl: GCN, GAT, message passing, and graph-level tasks.

### [Julia Reinforcement Learning](../julia-reinforcement-learning/SKILL.md)
ReinforcementLearning.jl: environments, policies, value functions, and training loops.

### [Julia ML Pipelines](../julia-ml-pipelines/SKILL.md)
MLJ.jl: data pipelines, model composition, cross-validation, and hyperparameter tuning.

### [Julia Model Deployment](../julia-model-deployment/SKILL.md)
Exporting models: ONNX, TorchScript, and serving via HTTP.jl endpoints.

## Routing Decision Tree

```
What is the Julia ML task?
|
+-- Build / train neural networks?
|   --> julia-neural-networks
|
+-- Design specific architectures (CNN/RNN/Transformer)?
|   --> julia-neural-architectures
|
+-- Diagnose training instability / slow convergence?
|   --> julia-training-diagnostics
|
+-- AD backend errors / gradient issues?
|   --> julia-ad-backends
|
+-- Write custom GPU kernels?
|   --> julia-gpu-kernels
|
+-- Graph-structured data / GNNs?
|   --> julia-graph-neural-networks
|
+-- Reinforcement learning?
|   --> julia-reinforcement-learning
|
+-- Data pipelines / MLJ workflows?
|   --> julia-ml-pipelines
|
+-- Export or serve a trained model?
    --> julia-model-deployment
```

## Checklist

- [ ] Select AD backend before writing training code (Zygote vs Enzyme)
- [ ] Verify CUDA.jl / KA backend compatibility with Julia version
- [ ] Check gradient flow with `Zygote.gradient` on a minimal example first
- [ ] Profile GPU utilization before optimizing kernel code
- [ ] Use MLJ for classical ML; Flux/Lux for deep learning
- [ ] Validate exported ONNX models against Julia predictions before serving
- [ ] Log training metrics with structured logging for reproducibility
