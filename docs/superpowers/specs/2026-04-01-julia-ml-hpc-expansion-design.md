# Julia ML/DL/HPC Expansion — Design Spec

**Date:** 2026-04-01
**Status:** Approved
**Scope:** Add Julia ML, Deep Learning, and HPC coverage to balance the JAX-first Python bias in science-suite

---

## Motivation

The science-suite has 86 skills and 11 agents, but Julia coverage is concentrated on SciML/ODE/Bayesian workflows. The ML/DL/Neural Networks/HPC domains are almost entirely Python/JAX-focused. This expansion adds 10 new skills and 1 new agent to achieve parity.

**Key decisions:**
- **Lux.jl-primary** with Flux.jl interop/migration coverage (not Flux-first)
- **Full cluster scale** HPC (CUDA.jl → multi-GPU → MPI.jl → SLURM) with single-node multi-GPU as foundation
- **MLJ.jl + DrWatson.jl** for ML pipelines (Julia-native, not forcing Airflow/Kubeflow patterns)
- **Dedicated `julia-ml-hpc` agent** (sonnet) — separate from `julia-pro` which keeps SciML/ODE/Bayesian

---

## New Agent: `julia-ml-hpc`

### Frontmatter

```yaml
name: julia-ml-hpc
description: Expert in Julia ML, Deep Learning, and HPC. Use for Lux.jl/Flux.jl neural networks, MLJ.jl pipelines, CUDA.jl GPU acceleration, KernelAbstractions.jl custom kernels, Distributed.jl/MPI.jl cluster computing, and GraphNeuralNetworks.jl. Delegates SciML/ODE work to julia-pro, theory to nonlinear-dynamics-expert, and framework-agnostic DL theory to neural-network-master.
model: sonnet
effort: high
memory: project
maxTurns: 40
tools: Read, Write, Edit, Bash, Grep, Glob
```

### Core Competencies

| Domain | Framework | Key Capabilities |
|--------|-----------|------------------|
| Neural Networks | Lux.jl / Flux.jl | Training loops, architectures, Flux-to-Lux migration |
| ML Pipelines | MLJ.jl / DrWatson.jl | Model selection, tuning, experiment management |
| GPU Computing | CUDA.jl / KernelAbstractions.jl | Custom kernels, multi-GPU, memory management |
| Distributed | Distributed.jl / MPI.jl | Data parallelism, AllReduce, SLURM orchestration |
| AD Backends | Zygote.jl / Enzyme.jl / ForwardDiff.jl | Backend selection, debugging, custom rules |
| Graph Neural Nets | GraphNeuralNetworks.jl | Message passing, node/edge/graph tasks |
| Reinforcement Learning | ReinforcementLearning.jl | Policies, environments, training loops |
| Deployment | ONNX.jl / Genie.jl / PackageCompiler | Export, serving, sysimages |

### Delegation Map

```
julia-ml-hpc (sonnet) — owns Julia ML/DL/HPC
  ├── delegates SciML/ODEs/UDEs → julia-pro
  ├── delegates DL theory (architectures, math) → neural-network-master
  ├── delegates bifurcation/chaos theory → nonlinear-dynamics-expert
  └── delegates Python/JAX equivalents → jax-pro, ml-expert
```

### Agent Structure

The agent markdown follows the same structure as `jax-pro.md` and `julia-pro.md`:
1. Activation rule (Julia ML/DL/HPC context detection)
2. Examples (4 representative use cases)
3. Core Competencies table
4. Pre-Response Validation Framework (5 checks)
5. Domain sections (8 domains matching competencies)
6. Delegation table
7. Cross-Domain Decision Framework
8. Common Failure Modes & Fixes
9. Production Checklist

---

## New Skills (10)

### Group A: Neural Networks & Training

#### 1. `julia-neural-networks`

- **Description:** Master Lux.jl and Flux.jl for deep learning in Julia. Covers explicit-parameter training loops, optimizers (Optimisers.jl), loss functions, data loading (DataLoaders.jl/MLUtils.jl), Flux-to-Lux migration, and supervised/unsupervised patterns. Use when training neural networks in Julia.
- **Packages:** Lux.jl, Flux.jl, Optimisers.jl, MLUtils.jl, OneHotArrays.jl, JLD2.jl
- **Sections:**
  - Expert Agent pointer (`julia-ml-hpc`)
  - Lux.jl vs Flux.jl decision tree
  - Training loop patterns (Lux explicit-parameter style)
  - Optimizer selection (Optimisers.jl: Adam, AdamW, LBFGS)
  - Data loading with MLUtils.jl (DataLoader, batching, shuffling)
  - Loss functions and custom losses
  - Checkpointing and serialization (JLD2)
  - Flux-to-Lux migration guide (step-by-step)
  - Anti-patterns table

#### 2. `julia-neural-architectures`

- **Description:** Design neural network architectures in Julia using Lux.jl. Covers Transformers (multi-head attention, positional encoding), CNNs, RNNs/LSTMs/GRUs (RecurrentLayers.jl), autoencoders, and custom layer design with explicit parameterization. Use when designing or implementing neural architectures in Julia.
- **Packages:** Lux.jl, RecurrentLayers.jl, Boltz.jl, LuxLib.jl, WeightInitializers.jl
- **Sections:**
  - Expert Agent pointer (`julia-ml-hpc`, theory delegation to `neural-network-master`)
  - CNN patterns (Conv, pooling, BatchNorm, residual blocks)
  - Transformer architecture (MultiHeadAttention, positional encoding, encoder/decoder)
  - RNN/LSTM/GRU via RecurrentLayers.jl
  - Autoencoder patterns (VAE, conditional)
  - Custom layer definition (`AbstractLuxLayer`, `initialparameters`, `initialstates`)
  - Composition patterns (Chain, Parallel, SkipConnection, BranchLayer)
  - Weight initialization strategies (WeightInitializers.jl)
  - Architecture selection decision tree

#### 3. `julia-training-diagnostics`

- **Description:** Debug and diagnose neural network training in Julia. Covers gradient analysis, loss landscape visualization, learning rate finding, convergence debugging, NaN/Inf detection, and Lux.jl-specific debugging patterns. Use when training is failing, diverging, or underperforming in Julia.
- **Packages:** Lux.jl, Zygote.jl, Enzyme.jl, Plots.jl/Makie.jl
- **Sections:**
  - Expert Agent pointer (`julia-ml-hpc`)
  - Gradient norm monitoring (per-layer analysis)
  - NaN/Inf detection and tracing
  - Learning rate range test implementation
  - Loss landscape visualization
  - Dead neuron / vanishing gradient detection
  - Zygote-specific error debugging (mutation errors, unsupported operations)
  - Enzyme-specific debugging (activity annotations, type errors)
  - Type instability in training loops (`@code_warntype` on hot paths)
  - Diagnostic checklist

### Group B: ML Pipelines & Experiment Management

#### 4. `julia-ml-pipelines`

- **Description:** Build ML pipelines in Julia with MLJ.jl for model selection, tuning, and evaluation, plus DrWatson.jl for experiment management and reproducibility. Covers learning networks, composable pipelines, hyperparameter tuning (Grid/Random/Latin), cross-validation, and scientific project organization. Use when building end-to-end ML workflows in Julia.
- **Packages:** MLJ.jl, MLJModels.jl, DrWatson.jl, DataFrames.jl, CSV.jl, StatsBase.jl
- **Sections:**
  - Expert Agent pointer (`julia-ml-hpc`)
  - MLJ model interface (@load, machine, fit!, predict)
  - Composable pipelines (Pipeline, learning networks)
  - Tuning strategies (Grid, RandomSearch, LatinHypercube) with TunedModel
  - Evaluation (evaluate!, cross-validation, stratified CV)
  - Model selection and comparison
  - DrWatson.jl project structure (`initialize_project`, `datadir()`, `srcdir()`)
  - Experiment management (`@tagsave`, `produce_or_load`, `dict_list`)
  - DataFrames.jl integration for feature engineering
  - Reproducibility checklist

### Group C: HPC & GPU

#### 5. `julia-hpc-distributed`

- **Description:** Scale Julia computations across clusters with Distributed.jl, MPI.jl, and SLURM job management. Covers multi-node data parallelism, AllReduce for gradient aggregation, pmap/remotecall patterns, Dagger.jl task DAGs, and SLURM batch scripting for HPC facilities. Use when scaling Julia beyond a single node.
- **Packages:** Distributed, MPI.jl, ClusterManagers.jl, Dagger.jl, SlurmClusterManager.jl
- **Sections:**
  - Expert Agent pointer (`julia-ml-hpc`)
  - Distributed.jl fundamentals (addprocs, @everywhere, pmap, remotecall)
  - RemoteChannel and SharedArrays for communication
  - MPI.jl collective operations (Bcast, Reduce, AllReduce, Scatter, Gather)
  - Distributed gradient aggregation pattern (AllReduce for DL training)
  - Dagger.jl DAG-based task scheduling
  - ClusterManagers.jl and SlurmClusterManager.jl setup
  - SLURM batch scripts for Julia (sbatch, srun, module loads)
  - Multi-node Julia startup patterns (--machine-file, --worker)
  - Performance: load balancing, data locality, serialization overhead
  - Decision tree: Distributed.jl vs MPI.jl vs Dagger.jl

#### 6. `julia-gpu-kernels`

- **Description:** Write high-performance GPU code in Julia with CUDA.jl and KernelAbstractions.jl. Covers custom kernel writing, shared memory optimization, multi-GPU data parallelism, memory management (unified/pinned), profiling with NVTX.jl, and portable kernels across CUDA/ROCm/oneAPI/Metal backends. Use when writing custom GPU kernels or optimizing GPU performance in Julia.
- **Packages:** CUDA.jl, KernelAbstractions.jl, AMDGPU.jl, Metal.jl, oneAPI.jl, NCCL.jl, NVTX.jl
- **Sections:**
  - Expert Agent pointer (`julia-ml-hpc`)
  - CUDA.jl CuArray operations (broadcasting, reductions, linear algebra)
  - Custom kernel writing (`@cuda` launch, threadIdx, blockIdx, grid stride)
  - KernelAbstractions.jl portable kernels (@kernel, @index, backend selection)
  - Shared memory and warp-level primitives
  - Memory management (unified memory, pinned memory, async transfers)
  - Multi-GPU with NCCL.jl (AllReduce, Broadcast)
  - Profiling with NVTX.jl and CUDA.@profile
  - Memory coalescing and occupancy optimization
  - Backend portability table (CUDA/ROCm/oneAPI/Metal)
  - Anti-patterns (scalar indexing, unnecessary host transfers)

### Group D: AD Backends & Deployment

#### 7. `julia-ad-backends`

- **Description:** Select and debug automatic differentiation backends in Julia. Covers Zygote.jl (source-to-source reverse-mode), Enzyme.jl (LLVM-level forward/reverse), ForwardDiff.jl (forward-mode dual numbers), and AbstractDifferentiation.jl for backend-agnostic code. Includes custom adjoint rules (ChainRulesCore.jl) and debugging strategies. Use when choosing AD backends or debugging gradient issues in Julia.
- **Packages:** Zygote.jl, Enzyme.jl, ForwardDiff.jl, AbstractDifferentiation.jl, ChainRulesCore.jl, DifferentiationInterface.jl
- **Sections:**
  - Expert Agent pointer (`julia-ml-hpc`)
  - Backend selection decision tree:
    - Few parameters, forward sensitivities → ForwardDiff.jl
    - Neural networks, reverse-mode, SciML → Zygote.jl
    - Maximum performance, mutation-friendly → Enzyme.jl
    - Backend-agnostic library code → DifferentiationInterface.jl
  - Zygote.jl: pullback API, limitations (no mutation, no try/catch), workarounds
  - Enzyme.jl: activity annotations (Const, Active, Duplicated), LLVM-level AD
  - ForwardDiff.jl: Dual numbers, Jacobians, Hessians, chunk size tuning
  - Custom rules with ChainRulesCore.jl (rrule, frule, @non_differentiable)
  - DifferentiationInterface.jl for backend-agnostic code
  - Mixed-mode AD (forward-over-reverse for Hessians)
  - Debugging: gradient correctness testing, finite difference verification
  - Performance comparison table

#### 8. `julia-model-deployment`

- **Description:** Deploy trained Julia models to production. Covers ONNX.jl model export, Genie.jl/Oxygen.jl REST API serving, PackageCompiler.jl system images for startup elimination, Docker containerization, and interop with Python serving stacks via PythonCall.jl. Use when deploying Julia ML models or reducing startup latency.
- **Packages:** ONNX.jl, Genie.jl, Oxygen.jl, PackageCompiler.jl, PythonCall.jl, JLD2.jl
- **Sections:**
  - Expert Agent pointer (`julia-ml-hpc`)
  - Model serialization (JLD2 for Julia-native, BSON legacy)
  - ONNX.jl export from Lux.jl and Flux.jl models
  - REST API serving with Genie.jl (routes, JSON serialization, async)
  - Lightweight serving with Oxygen.jl
  - PackageCompiler.jl system images (create_sysimage, precompile statements)
  - Docker containerization (multi-stage builds, Julia depot caching)
  - Python interop via PythonCall.jl (calling Julia models from Python serving stacks)
  - Latency optimization strategies (precompilation, sysimages, warmup)
  - Production checklist

### Group E: Specialized Domains

#### 9. `julia-graph-neural-networks`

- **Description:** Build graph neural networks in Julia with GraphNeuralNetworks.jl and Lux.jl. Covers GCN, GAT, GraphSAGE, message passing neural networks, node/edge/graph-level tasks, heterogeneous graphs, temporal graphs (TemporalGNNs), and mini-batch training on large graphs. Use when working with graph-structured data in Julia.
- **Packages:** GraphNeuralNetworks.jl, GNNLux.jl, Graphs.jl, MetaGraphs.jl
- **Sections:**
  - Expert Agent pointer (`julia-ml-hpc`)
  - Graph construction (GNNGraph from adjacency, edge list, Graphs.jl)
  - Built-in GNN layers (GCNConv, GATConv, SAGEConv, GINConv)
  - GNNLux.jl integration (explicit-parameter GNN layers)
  - Custom message passing (propagate, aggregate)
  - Task types: node classification, link prediction, graph classification
  - Global pooling (mean, max, attention)
  - Mini-batching with batch/unbatch for large graphs
  - Heterogeneous graphs
  - Temporal GNN patterns
  - Application: molecular property prediction example

#### 10. `julia-reinforcement-learning`

- **Description:** Implement reinforcement learning in Julia with ReinforcementLearning.jl. Covers policy gradient methods (PPO, A2C), value-based methods (DQN, DDPG), custom environments, multi-agent RL, and integration with Lux.jl for policy networks. Use when building RL agents or custom environments in Julia.
- **Packages:** ReinforcementLearning.jl, CommonRLInterface.jl, Lux.jl, StableRNGs.jl
- **Sections:**
  - Expert Agent pointer (`julia-ml-hpc`)
  - Environment interface (AbstractEnv, action_space, state, reward, is_terminated)
  - Custom environment creation
  - Policy definition with Lux.jl neural networks
  - Value-based methods (DQN, Double DQN, Dueling DQN)
  - Policy gradient methods (PPO, A2C, REINFORCE)
  - Actor-critic (DDPG, TD3, SAC)
  - Replay buffers and experience collection
  - CommonRLInterface.jl for interop with other RL packages
  - Training hooks and logging
  - Multi-agent RL patterns
  - Reward shaping strategies

---

## Existing Component Updates

### Agent Delegation Table Additions

| Agent File | Add Row |
|------------|---------|
| `agents/julia-pro.md` | `julia-ml-hpc` — "Julia ML training, GPU kernels, distributed computing, MLJ.jl pipelines" |
| `agents/neural-network-master.md` | `julia-ml-hpc` — "Julia-specific DL implementation (Lux.jl/Flux.jl architectures, training)" |
| `agents/ml-expert.md` | `julia-ml-hpc` — "Julia ML pipelines (MLJ.jl), experiment management (DrWatson.jl)" |
| `agents/simulation-expert.md` | `julia-ml-hpc` — "Julia GPU kernels and HPC distributed computing" |

### `julia-pro` Description Update

**Before:**
> Expert Julia scientific computing agent. Use for Core Julia, SciML (Lux.jl, DifferentialEquations.jl, ModelingToolkit.jl), Turing.jl, nonlinear dynamics (DynamicalSystems.jl, BifurcationKit.jl), and data-driven modeling (DataDrivenDiffEq.jl/SINDy). Handles UDEs, sensitivity analysis, and package development. Delegates theory to nonlinear-dynamics-expert.

**After:**
> Expert Julia scientific computing agent. Use for Core Julia, SciML (DifferentialEquations.jl, ModelingToolkit.jl, Lux.jl for UDEs), Turing.jl, nonlinear dynamics (DynamicalSystems.jl, BifurcationKit.jl), and data-driven modeling (DataDrivenDiffEq.jl/SINDy). Handles UDEs, sensitivity analysis, and package development. Delegates ML/DL/HPC to julia-ml-hpc, theory to nonlinear-dynamics-expert.

### `julia-pro` vs `julia-ml-hpc` Boundary for Lux.jl

| Task | Owner | Rationale |
|------|-------|-----------|
| Lux.jl inside UDEs (neural ODE right-hand side) | `julia-pro` | UDE is a SciML workflow |
| Lux.jl training loop for supervised learning | `julia-ml-hpc` | Pure ML workflow |
| Lux.jl architecture design (Transformer, CNN) | `julia-ml-hpc` | Architecture is DL domain |
| Lux.jl + SciMLSensitivity adjoint training | `julia-pro` | Adjoint sensitivity is SciML |
| Lux.jl + CUDA.jl multi-GPU training | `julia-ml-hpc` | HPC domain |
| Lux.jl custom layer for physics constraint | `julia-pro` | Physics-informed is SciML |

### Existing Skill Cross-References

Add a Julia equivalent line in the Expert Agents section of each:

| Existing Skill | Add Reference |
|---------------|---------------|
| `deep-learning/SKILL.md` | `julia-ml-hpc` agent + `julia-neural-networks` skill |
| `neural-architecture-patterns/SKILL.md` | `julia-neural-architectures` skill |
| `training-diagnostics/SKILL.md` | `julia-training-diagnostics` skill |
| `machine-learning/SKILL.md` | `julia-ml-pipelines` skill |
| `parallel-computing/SKILL.md` | `julia-hpc-distributed` skill |
| `gpu-acceleration/SKILL.md` | `julia-gpu-kernels` skill |
| `model-deployment-serving/SKILL.md` | `julia-model-deployment` skill |

### Plugin Manifest Update

`plugins/science-suite/.claude-plugin/plugin.json` — add file-path references:
- 1 new agent: `agents/julia-ml-hpc.md`
- 10 new skills: `skills/julia-neural-networks/SKILL.md`, `skills/julia-neural-architectures/SKILL.md`, etc.

---

## Context Budget Impact

| Metric | Before | After |
|--------|--------|-------|
| Science suite skills | 86 | 96 |
| Total project skills | 132 | 142 |
| Estimated description tokens | ~5,280 | ~5,680 |
| 2% budget (1M context) | 20,000 | 20,000 |
| Budget utilization | ~26% | ~28% |

Well within limits.

---

## Post-Change Totals

| Metric | Before | After |
|--------|--------|-------|
| Science suite agents | 11 | 12 |
| Science suite skills | 86 | 96 |
| Total project agents | 23 | 24 |
| Total project skills | 132 | 142 |

---

## Implementation Order

Recommended build sequence (dependencies flow downward):

1. **Phase 1 — Agent:** Create `julia-ml-hpc.md` agent definition
2. **Phase 2 — Core skills:** `julia-neural-networks`, `julia-ad-backends` (foundation for all others)
3. **Phase 3 — Architecture & training:** `julia-neural-architectures`, `julia-training-diagnostics`
4. **Phase 4 — ML & HPC:** `julia-ml-pipelines`, `julia-gpu-kernels`, `julia-hpc-distributed`
5. **Phase 5 — Deployment & specialized:** `julia-model-deployment`, `julia-graph-neural-networks`, `julia-reinforcement-learning`
6. **Phase 6 — Integration:** Update existing agents (delegation rows), update existing skills (cross-references), update plugin.json manifest
7. **Phase 7 — Validation:** Run `uv run pytest tools/tests/ -v`, run context budget checker, verify plugin integrity
