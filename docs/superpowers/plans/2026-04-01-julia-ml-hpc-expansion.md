# Julia ML/DL/HPC Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 10 new Julia ML/DL/HPC skills and 1 new agent (`julia-ml-hpc`) to the science-suite plugin, update existing agents/skills with cross-references, and update the plugin manifest.

**Architecture:** New content lives entirely in `plugins/science-suite/`. One new agent markdown file, 10 new skill directories each containing a `SKILL.md`. Existing agents get delegation table rows added. Existing skills get cross-reference lines in their Expert Agents sections. The plugin.json manifest gets updated with file-path references.

**Tech Stack:** Markdown with YAML frontmatter (skills/agents), JSON (plugin.json), Python/pytest (validation)

**Spec:** `docs/superpowers/specs/2026-04-01-julia-ml-hpc-expansion-design.md`

---

## File Structure

### New Files (11)
```
plugins/science-suite/
  agents/
    julia-ml-hpc.md                          # New agent
  skills/
    julia-neural-networks/SKILL.md            # Lux.jl/Flux.jl training
    julia-neural-architectures/SKILL.md       # CNN/Transformer/RNN in Lux
    julia-training-diagnostics/SKILL.md       # Gradient/loss debugging
    julia-ml-pipelines/SKILL.md               # MLJ.jl + DrWatson.jl
    julia-hpc-distributed/SKILL.md            # Distributed.jl/MPI.jl/SLURM
    julia-gpu-kernels/SKILL.md                # CUDA.jl/KernelAbstractions.jl
    julia-ad-backends/SKILL.md                # Zygote/Enzyme/ForwardDiff
    julia-model-deployment/SKILL.md           # ONNX/Genie/PackageCompiler
    julia-graph-neural-networks/SKILL.md      # GraphNeuralNetworks.jl
    julia-reinforcement-learning/SKILL.md     # ReinforcementLearning.jl
```

### Modified Files (12)
```
plugins/science-suite/
  .claude-plugin/plugin.json                    # Add 1 agent + 10 skills
  agents/
    julia-pro.md                              # Update description + delegation
    neural-network-master.md                  # Add delegation row
    ml-expert.md                              # Add delegation row
    simulation-expert.md                      # Add delegation row
  skills/
    deep-learning/SKILL.md                    # Add Julia cross-ref
    neural-architecture-patterns/SKILL.md     # Add Julia cross-ref
    training-diagnostics/SKILL.md             # Add Julia cross-ref
    machine-learning/SKILL.md                 # Add Julia cross-ref
    parallel-computing/SKILL.md               # Add Julia cross-ref
    gpu-acceleration/SKILL.md                 # Add Julia cross-ref
    model-deployment-serving/SKILL.md         # Add Julia cross-ref
```

---

## Task 1: Create the `julia-ml-hpc` agent

**Files:**
- Create: `plugins/science-suite/agents/julia-ml-hpc.md`

- [ ] **Step 1: Create the agent markdown file**

Write `plugins/science-suite/agents/julia-ml-hpc.md` with the complete agent definition. The file must have YAML frontmatter with these exact fields:

```yaml
---
name: julia-ml-hpc
description: Expert in Julia ML, Deep Learning, and HPC. Use for Lux.jl/Flux.jl neural networks, MLJ.jl pipelines, CUDA.jl GPU acceleration, KernelAbstractions.jl custom kernels, Distributed.jl/MPI.jl cluster computing, and GraphNeuralNetworks.jl. Delegates SciML/ODE work to julia-pro, theory to nonlinear-dynamics-expert, and framework-agnostic DL theory to neural-network-master.
model: sonnet
effort: high
memory: project
maxTurns: 40
tools: Read, Write, Edit, Bash, Grep, Glob
---
```

The body must follow the same structure as `plugins/science-suite/agents/jax-pro.md`:
1. Title: `# Julia ML/HPC - Machine Learning & High-Performance Computing Specialist`
2. Activation Rule: Activate ONLY when Julia ML/DL/GPU/HPC context is detected. Delegate SciML/ODE/UDE to julia-pro.
3. Examples: 4 examples covering (a) Lux.jl CNN training with GPU, (b) MPI.jl cluster scaling with SLURM, (c) MLJ.jl pipeline with cross-validation, (d) KernelAbstractions.jl portable kernel
4. Core Competencies table: 8 rows (Neural Networks, ML Pipelines, GPU Computing, Distributed, AD Backends, Graph Neural Nets, RL, Deployment)
5. Pre-Response Validation Framework: 5 checks (Problem Classification, Type Stability, Performance & Memory, Framework Correctness, Production Readiness)
6. Domain sections (8 domains):
   - Domain 1: Neural Networks — Lux.jl training loop template, Lux vs Flux decision table, Flux-to-Lux migration
   - Domain 2: ML Pipelines — MLJ.jl quick reference, DrWatson.jl experiment management
   - Domain 3: GPU Computing — CUDA.jl basics, custom kernel, KernelAbstractions.jl portable kernel, anti-patterns table
   - Domain 4: Distributed Computing — Distributed.jl patterns, MPI.jl AllReduce, SLURM batch script
   - Domain 5: AD Backends — backend selection table, Zygote/Enzyme/ForwardDiff quick reference, ChainRulesCore custom rule
   - Domain 6: Graph Neural Networks — GNNGraph construction, GCN model with Lux, node classification
   - Domain 7: Reinforcement Learning — CartPoleEnv DQN agent
   - Domain 8: Model Deployment — PackageCompiler sysimage, Genie.jl REST API
7. Delegation Table: 6 rows (julia-pro for UDEs, neural-network-master for DL theory, nonlinear-dynamics-expert for chaos, jax-pro for Python, ml-expert for Python ML)
8. Cross-Domain Decision Framework: ASCII decision tree
9. Common Failure Modes & Fixes: 8-row table
10. Production Checklist: 10 items

- [ ] **Step 2: Verify frontmatter parses correctly**

Run: `cd /Users/b80985/Projects/MyClaude && python3 -c "import yaml, re; content=open('plugins/science-suite/agents/julia-ml-hpc.md').read(); match=re.match(r'^---\n(.*?)\n---',content,re.DOTALL); fm=yaml.safe_load(match.group(1)); assert fm['name']=='julia-ml-hpc'; assert fm['model']=='sonnet'; print('OK: frontmatter valid')"`

Expected: `OK: frontmatter valid`

- [ ] **Step 3: Commit**

```bash
git add plugins/science-suite/agents/julia-ml-hpc.md
git commit -m "feat(science-suite): add julia-ml-hpc agent for Julia ML/DL/HPC"
```

---

## Task 2: Create `julia-neural-networks` skill

**Files:**
- Create: `plugins/science-suite/skills/julia-neural-networks/SKILL.md`

- [ ] **Step 1: Create the skill directory and SKILL.md**

Create directory `plugins/science-suite/skills/julia-neural-networks/` and write `SKILL.md` with:

Frontmatter:
```yaml
---
name: julia-neural-networks
description: Master Lux.jl and Flux.jl for deep learning in Julia. Covers explicit-parameter training loops, optimizers (Optimisers.jl), loss functions, data loading (DataLoaders.jl/MLUtils.jl), Flux-to-Lux migration, and supervised/unsupervised patterns. Use when training neural networks in Julia.
---
```

Body sections:
1. Expert Agent pointer to `julia-ml-hpc` and `neural-network-master`
2. Lux.jl vs Flux.jl comparison table (5 rows: Parameters, SciML compat, AD support, State mgmt, New projects)
3. Complete Training Loop — full working example with Lux.jl: model definition (Chain+Dense), `Lux.setup(rng, model)`, `Optimisers.setup(Adam(1e-3), ps)`, train_step function using `Zygote.withgradient`, DataLoader loop, checkpointing
4. Inference section — `Lux.testmode(st)` for BatchNorm/Dropout
5. Optimizer Selection table (Adam, AdamW, SGD, Lion) with ParameterSchedulers.jl examples (CosAnneal, Step)
6. Data Loading with MLUtils.jl — DataLoader, splitobs, augmentation
7. Loss Functions — logitcrossentropy, mse, mae, huber_loss, custom focal_loss example
8. Checkpointing with JLD2 — save/load example
9. Flux-to-Lux Migration Guide — 7-row mapping table
10. Anti-Patterns table (6 rows)
11. Checklist (9 items)

- [ ] **Step 2: Verify skill structure**

Run: `cd /Users/b80985/Projects/MyClaude && python3 -c "import yaml,re; from pathlib import Path; p=Path('plugins/science-suite/skills/julia-neural-networks/SKILL.md'); assert p.exists(); content=p.read_text(); m=re.match(r'^---\n(.*?)\n---',content,re.DOTALL); fm=yaml.safe_load(m.group(1)); assert fm['name']=='julia-neural-networks'; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add plugins/science-suite/skills/julia-neural-networks/
git commit -m "feat(science-suite): add julia-neural-networks skill (Lux.jl/Flux.jl)"
```

---

## Task 3: Create `julia-ad-backends` skill

**Files:**
- Create: `plugins/science-suite/skills/julia-ad-backends/SKILL.md`

- [ ] **Step 1: Create skill directory and SKILL.md**

Frontmatter:
```yaml
---
name: julia-ad-backends
description: Select and debug automatic differentiation backends in Julia. Covers Zygote.jl (source-to-source reverse-mode), Enzyme.jl (LLVM-level forward/reverse), ForwardDiff.jl (forward-mode dual numbers), and AbstractDifferentiation.jl for backend-agnostic code. Includes custom adjoint rules (ChainRulesCore.jl) and debugging strategies. Use when choosing AD backends or debugging gradient issues in Julia.
---
```

Body sections:
1. Expert Agent pointer to `julia-ml-hpc`
2. Backend Selection Decision Tree (ASCII: parameters count -> mode -> mutation support -> library choice)
3. Zygote.jl section — gradient, withgradient, pullback API, Limitations table (4 rows: mutation, try/catch, foreign calls, slow compilation)
4. Enzyme.jl section — autodiff Reverse/Forward, Activity annotations table (Active, Duplicated, Const, DuplicatedNoNeed)
5. ForwardDiff.jl section — gradient, jacobian, hessian, chunk size tuning
6. Custom Rules (ChainRulesCore.jl) — rrule and frule examples, @non_differentiable
7. DifferentiationInterface.jl — unified API with backend swapping
8. Mixed-Mode AD — forward-over-reverse for Hessians
9. Gradient Correctness Testing — FiniteDifferences.jl verification
10. Performance Comparison table (5 columns: Compile Time, Runtime, Memory, Mutation, GPU)
11. Checklist (6 items)

- [ ] **Step 2: Verify and commit**

Run: `cd /Users/b80985/Projects/MyClaude && python3 -c "import yaml,re; content=open('plugins/science-suite/skills/julia-ad-backends/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---',content,re.DOTALL); fm=yaml.safe_load(m.group(1)); assert fm['name']=='julia-ad-backends'; print('OK')"`

```bash
git add plugins/science-suite/skills/julia-ad-backends/
git commit -m "feat(science-suite): add julia-ad-backends skill (Zygote/Enzyme/ForwardDiff)"
```

---

## Task 4: Create `julia-neural-architectures` skill

**Files:**
- Create: `plugins/science-suite/skills/julia-neural-architectures/SKILL.md`

- [ ] **Step 1: Create skill directory and SKILL.md**

Frontmatter:
```yaml
---
name: julia-neural-architectures
description: Design neural network architectures in Julia using Lux.jl. Covers Transformers (multi-head attention, positional encoding), CNNs, RNNs/LSTMs/GRUs (RecurrentLayers.jl), autoencoders, and custom layer design with explicit parameterization. Use when designing or implementing neural architectures in Julia.
---
```

Body sections:
1. Expert Agent pointer to `julia-ml-hpc` and `neural-network-master`
2. CNN Patterns — basic CNN (Conv+MaxPool+GlobalMeanPool+Dense), Residual Block using `Parallel(+, Chain(...), NoOpLayer())`
3. Transformer Architecture — MultiHeadAttention, TransformerBlock function (SkipConnection+LayerNorm+FFN), positional encoding function
4. RNN/LSTM/GRU via RecurrentLayers.jl — Recurrence(LSTMCell), GRUCell, BidirectionalRNN
5. Autoencoder Patterns — VAE with encoder (mu, logvar), decoder, and vae_loss function
6. Custom Layer Design — struct extending `Lux.AbstractLuxLayer`, `initialparameters`, `initialstates`, forward pass
7. Composition Patterns table (Chain, Parallel, SkipConnection, BranchLayer, WrappedFunction)
8. Weight Initialization (WeightInitializers.jl) — kaiming_normal, glorot_uniform, with activation-to-init mapping table
9. Architecture Selection table (task -> architecture -> Lux pattern)
10. Checklist (7 items)

- [ ] **Step 2: Verify and commit**

Run: `cd /Users/b80985/Projects/MyClaude && python3 -c "import yaml,re; content=open('plugins/science-suite/skills/julia-neural-architectures/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---',content,re.DOTALL); fm=yaml.safe_load(m.group(1)); assert fm['name']=='julia-neural-architectures'; print('OK')"`

```bash
git add plugins/science-suite/skills/julia-neural-architectures/
git commit -m "feat(science-suite): add julia-neural-architectures skill (Lux.jl)"
```

---

## Task 5: Create `julia-training-diagnostics` skill

**Files:**
- Create: `plugins/science-suite/skills/julia-training-diagnostics/SKILL.md`

- [ ] **Step 1: Create skill directory and SKILL.md**

Frontmatter:
```yaml
---
name: julia-training-diagnostics
description: Debug and diagnose neural network training in Julia. Covers gradient analysis, loss landscape visualization, learning rate finding, convergence debugging, NaN/Inf detection, and Lux.jl-specific debugging patterns. Use when training is failing, diverging, or underperforming in Julia.
---
```

Body sections:
1. Expert Agent pointer to `julia-ml-hpc`
2. Quick Reference table (7 rows: Loss=NaN, plateau, overfitting, underfitting, spikes, slow convergence, mutation error)
3. Gradient Monitoring — per-layer gradient norm function using ComponentArrays, gradient clipping with `OptimiserChain(ClipNorm(1.0), Adam())`
4. NaN/Inf Detection — `check_for_nans` function, integration into training loop
5. Learning Rate Range Test — full implementation with exponential LR increase and loss tracking
6. Zygote-Specific Debugging — error table (4 rows), mutation fix example (BAD vs GOOD)
7. Enzyme-Specific Debugging — error table (3 rows)
8. Type Instability in Training — `@code_warntype` guidance, common fixes
9. Diagnostic Checklist (8 items)

- [ ] **Step 2: Verify and commit**

Run: `cd /Users/b80985/Projects/MyClaude && python3 -c "import yaml,re; content=open('plugins/science-suite/skills/julia-training-diagnostics/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---',content,re.DOTALL); fm=yaml.safe_load(m.group(1)); assert fm['name']=='julia-training-diagnostics'; print('OK')"`

```bash
git add plugins/science-suite/skills/julia-training-diagnostics/
git commit -m "feat(science-suite): add julia-training-diagnostics skill"
```

---

## Task 6: Create `julia-ml-pipelines` skill

**Files:**
- Create: `plugins/science-suite/skills/julia-ml-pipelines/SKILL.md`

- [ ] **Step 1: Create skill directory and SKILL.md**

Frontmatter:
```yaml
---
name: julia-ml-pipelines
description: Build ML pipelines in Julia with MLJ.jl for model selection, tuning, and evaluation, plus DrWatson.jl for experiment management and reproducibility. Covers learning networks, composable pipelines, hyperparameter tuning (Grid/Random/Latin), cross-validation, and scientific project organization. Use when building end-to-end ML workflows in Julia.
---
```

Body sections:
1. Expert Agent pointer to `julia-ml-hpc`
2. MLJ.jl Model Interface — @load, machine, fit!, predict, predict_mode, models() search
3. Composable Pipelines — pipe with `|>` operator, Learning Networks with @from_network
4. Hyperparameter Tuning — Grid (TunedModel with range), RandomSearch, LatinHypercube
5. Evaluation — evaluate! with CV, StratifiedCV, multiple measures, Holdout
6. Model Comparison — map over models list with evaluate
7. DrWatson.jl Project Structure — initialize_project layout, @quickactivate
8. Experiment Management — dict_list for parameter sweeps, @tagsave, produce_or_load
9. DataFrames.jl Integration — CSV.read, transform!, coerce! for MLJ
10. Reproducibility Checklist (8 items)

- [ ] **Step 2: Verify and commit**

Run: `cd /Users/b80985/Projects/MyClaude && python3 -c "import yaml,re; content=open('plugins/science-suite/skills/julia-ml-pipelines/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---',content,re.DOTALL); fm=yaml.safe_load(m.group(1)); assert fm['name']=='julia-ml-pipelines'; print('OK')"`

```bash
git add plugins/science-suite/skills/julia-ml-pipelines/
git commit -m "feat(science-suite): add julia-ml-pipelines skill (MLJ.jl/DrWatson.jl)"
```

---

## Task 7: Create `julia-gpu-kernels` skill

**Files:**
- Create: `plugins/science-suite/skills/julia-gpu-kernels/SKILL.md`

- [ ] **Step 1: Create skill directory and SKILL.md**

Frontmatter:
```yaml
---
name: julia-gpu-kernels
description: Write high-performance GPU code in Julia with CUDA.jl and KernelAbstractions.jl. Covers custom kernel writing, shared memory optimization, multi-GPU data parallelism, memory management (unified/pinned), profiling with NVTX.jl, and portable kernels across CUDA/ROCm/oneAPI/Metal backends. Use when writing custom GPU kernels or optimizing GPU performance in Julia.
---
```

Body sections:
1. Expert Agent pointer to `julia-ml-hpc`
2. CUDA.jl Basics — CuArray operations (broadcasting, linear algebra), Lux model to GPU with `gpu_device()`
3. Custom CUDA Kernel — `@cuda` launch with threadIdx/blockIdx, grid-stride loop pattern
4. KernelAbstractions.jl — @kernel macro, @index(Global), backend detection, synchronize
5. Backend Portability table (CUDA/ROCm/Metal/oneAPI/CPU)
6. Shared Memory — reduction kernel with @cuStaticSharedMem, sync_threads, tree reduction
7. Memory Management table (Device, Unified, Pinned, Async) with stream-based async transfer example
8. Multi-GPU with NCCL.jl — Communicators, Allreduce
9. Profiling — NVTX.jl @range annotations, CUDA.@profile, nsys/ncu commands
10. Anti-Patterns table (6 rows: scalar indexing, Array in loop, Float64, allocating in kernel, small kernels, ignoring occupancy)
11. Occupancy Optimization — device attribute queries, launch_configuration
12. Checklist (8 items)

- [ ] **Step 2: Verify and commit**

Run: `cd /Users/b80985/Projects/MyClaude && python3 -c "import yaml,re; content=open('plugins/science-suite/skills/julia-gpu-kernels/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---',content,re.DOTALL); fm=yaml.safe_load(m.group(1)); assert fm['name']=='julia-gpu-kernels'; print('OK')"`

```bash
git add plugins/science-suite/skills/julia-gpu-kernels/
git commit -m "feat(science-suite): add julia-gpu-kernels skill (CUDA.jl/KernelAbstractions.jl)"
```

---

## Task 8: Create `julia-hpc-distributed` skill

**Files:**
- Create: `plugins/science-suite/skills/julia-hpc-distributed/SKILL.md`

- [ ] **Step 1: Create skill directory and SKILL.md**

Frontmatter:
```yaml
---
name: julia-hpc-distributed
description: Scale Julia computations across clusters with Distributed.jl, MPI.jl, and SLURM job management. Covers multi-node data parallelism, AllReduce for gradient aggregation, pmap/remotecall patterns, Dagger.jl task DAGs, and SLURM batch scripting for HPC facilities. Use when scaling Julia beyond a single node.
---
```

Body sections:
1. Expert Agent pointer to `julia-ml-hpc`
2. Distributed.jl Fundamentals — addprocs (local + SSH), @everywhere, pmap, @distributed reduction, remotecall/fetch, @spawnat
3. Communication — RemoteChannel producer-consumer pattern, SharedArrays for single-node
4. MPI.jl — Init, COMM_WORLD, rank/size, Collective Operations (bcast, Scatter, Gather, AllReduce)
5. Distributed Gradient Aggregation — full function using Zygote.withgradient + MPI.Allreduce for synchronized training
6. SLURM Integration — basic Julia SLURM script (CPU), GPU-aware SLURM script, module loads
7. ClusterManagers.jl — SlurmManager, PBSManager for auto worker launch
8. Dagger.jl — @spawn task DAG with dependencies, fetch
9. Decision Tree table (6 rows: which framework for which scenario)
10. Performance Tips table (5 rows)
11. Checklist (7 items)

- [ ] **Step 2: Verify and commit**

Run: `cd /Users/b80985/Projects/MyClaude && python3 -c "import yaml,re; content=open('plugins/science-suite/skills/julia-hpc-distributed/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---',content,re.DOTALL); fm=yaml.safe_load(m.group(1)); assert fm['name']=='julia-hpc-distributed'; print('OK')"`

```bash
git add plugins/science-suite/skills/julia-hpc-distributed/
git commit -m "feat(science-suite): add julia-hpc-distributed skill (Distributed.jl/MPI.jl/SLURM)"
```

---

## Task 9: Create `julia-model-deployment` skill

**Files:**
- Create: `plugins/science-suite/skills/julia-model-deployment/SKILL.md`

- [ ] **Step 1: Create skill directory and SKILL.md**

Frontmatter:
```yaml
---
name: julia-model-deployment
description: Deploy trained Julia models to production. Covers ONNX.jl model export, Genie.jl/Oxygen.jl REST API serving, PackageCompiler.jl system images for startup elimination, Docker containerization, and interop with Python serving stacks via PythonCall.jl. Use when deploying Julia ML models or reducing startup latency.
---
```

Body sections:
1. Expert Agent pointer to `julia-ml-hpc`
2. Model Serialization — JLD2 save/load (recommended), ONNX export with ONNXRunTime
3. REST API Serving — Genie.jl (route POST /predict, route GET /health, up(8080)), Oxygen.jl lightweight alternative
4. PackageCompiler.jl — create_sysimage with precompile_execution_file, create_app for standalone executable, launch command
5. Docker Containerization — multi-stage Dockerfile (builder with sysimage, slim runtime), Julia depot caching with --mount=type=cache
6. Python Interop (PythonCall.jl) — calling Julia models from Python
7. Latency Optimization table (4 rows: Default, Sysimage, Standalone, Precompilation with startup/first-request/steady timings)
8. Production Checklist (8 items)

- [ ] **Step 2: Verify and commit**

Run: `cd /Users/b80985/Projects/MyClaude && python3 -c "import yaml,re; content=open('plugins/science-suite/skills/julia-model-deployment/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---',content,re.DOTALL); fm=yaml.safe_load(m.group(1)); assert fm['name']=='julia-model-deployment'; print('OK')"`

```bash
git add plugins/science-suite/skills/julia-model-deployment/
git commit -m "feat(science-suite): add julia-model-deployment skill (ONNX/Genie/PackageCompiler)"
```

---

## Task 10: Create `julia-graph-neural-networks` skill

**Files:**
- Create: `plugins/science-suite/skills/julia-graph-neural-networks/SKILL.md`

- [ ] **Step 1: Create skill directory and SKILL.md**

Frontmatter:
```yaml
---
name: julia-graph-neural-networks
description: Build graph neural networks in Julia with GraphNeuralNetworks.jl and Lux.jl. Covers GCN, GAT, GraphSAGE, message passing neural networks, node/edge/graph-level tasks, heterogeneous graphs, temporal graphs (TemporalGNNs), and mini-batch training on large graphs. Use when working with graph-structured data in Julia.
---
```

Body sections:
1. Expert Agent pointer to `julia-ml-hpc`
2. Graph Construction — GNNGraph from edge list, adjacency matrix, Graphs.jl; adding ndata/edata
3. Built-In GNN Layers table (GCNConv, GATConv, SAGEConv, GINConv, EdgeConv, GatedGraphConv)
4. GCN Example — GNNChain with 3 GCNConv layers, Lux.setup, forward pass
5. GAT — multi-head attention with concat
6. GNNLux.jl Integration — explicit-parameter GNN layers
7. Task Types — node classification (with train_mask), link prediction (dot product), graph classification (GlobalPool)
8. Mini-Batching — batch/unbatch for multiple graphs
9. Custom Message Passing — propagate with custom_message function
10. Application — molecular property prediction with GINConv
11. Checklist (7 items)

- [ ] **Step 2: Verify and commit**

Run: `cd /Users/b80985/Projects/MyClaude && python3 -c "import yaml,re; content=open('plugins/science-suite/skills/julia-graph-neural-networks/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---',content,re.DOTALL); fm=yaml.safe_load(m.group(1)); assert fm['name']=='julia-graph-neural-networks'; print('OK')"`

```bash
git add plugins/science-suite/skills/julia-graph-neural-networks/
git commit -m "feat(science-suite): add julia-graph-neural-networks skill (GraphNeuralNetworks.jl)"
```

---

## Task 11: Create `julia-reinforcement-learning` skill

**Files:**
- Create: `plugins/science-suite/skills/julia-reinforcement-learning/SKILL.md`

- [ ] **Step 1: Create skill directory and SKILL.md**

Frontmatter:
```yaml
---
name: julia-reinforcement-learning
description: Implement reinforcement learning in Julia with ReinforcementLearning.jl. Covers policy gradient methods (PPO, A2C), value-based methods (DQN, DDPG), custom environments, multi-agent RL, and integration with Lux.jl for policy networks. Use when building RL agents or custom environments in Julia.
---
```

Body sections:
1. Expert Agent pointer to `julia-ml-hpc`
2. Environment Interface — built-in envs (CartPoleEnv, PendulumEnv), API (state, action_space, reward, is_terminated)
3. Custom Environment — full GridWorldEnv implementation with RLBase interface methods
4. Value-Based Methods — DQN (full agent setup with NeuralNetworkApproximator, EpsilonGreedyExplorer, CircularArraySARTTrajectory, run), Double DQN flag
5. Policy Gradient Methods — PPO (ActorCritic with GAE), A2C (with entropy bonus)
6. Continuous Action Spaces — DDPG (behavior/target actor-critic, soft update)
7. CommonRLInterface.jl — wrapper struct for cross-package compatibility
8. Training Hooks & Logging — ComposedHook, custom SaveBestModel hook
9. Reward Shaping table (Sparse, Dense, Curiosity, Curriculum)
10. Checklist (8 items)

- [ ] **Step 2: Verify and commit**

Run: `cd /Users/b80985/Projects/MyClaude && python3 -c "import yaml,re; content=open('plugins/science-suite/skills/julia-reinforcement-learning/SKILL.md').read(); m=re.match(r'^---\n(.*?)\n---',content,re.DOTALL); fm=yaml.safe_load(m.group(1)); assert fm['name']=='julia-reinforcement-learning'; print('OK')"`

```bash
git add plugins/science-suite/skills/julia-reinforcement-learning/
git commit -m "feat(science-suite): add julia-reinforcement-learning skill (ReinforcementLearning.jl)"
```

---

## Task 12: Update existing agents with delegation rows

**Files:**
- Modify: `plugins/science-suite/agents/julia-pro.md`
- Modify: `plugins/science-suite/agents/neural-network-master.md`
- Modify: `plugins/science-suite/agents/ml-expert.md`
- Modify: `plugins/science-suite/agents/simulation-expert.md`

- [ ] **Step 1: Update `julia-pro.md` description**

In `plugins/science-suite/agents/julia-pro.md`, change the `description` field in the frontmatter from:

```
description: Expert Julia scientific computing agent. Use for Core Julia, SciML (Lux.jl, DifferentialEquations.jl, ModelingToolkit.jl), Turing.jl, nonlinear dynamics (DynamicalSystems.jl, BifurcationKit.jl), and data-driven modeling (DataDrivenDiffEq.jl/SINDy). Handles UDEs, sensitivity analysis, and package development. Delegates theory to nonlinear-dynamics-expert.
```

to:

```
description: Expert Julia scientific computing agent. Use for Core Julia, SciML (DifferentialEquations.jl, ModelingToolkit.jl, Lux.jl for UDEs), Turing.jl, nonlinear dynamics (DynamicalSystems.jl, BifurcationKit.jl), and data-driven modeling (DataDrivenDiffEq.jl/SINDy). Handles UDEs, sensitivity analysis, and package development. Delegates ML/DL/HPC to julia-ml-hpc, theory to nonlinear-dynamics-expert.
```

- [ ] **Step 2: Add delegation row to `julia-pro.md`**

In the Delegation Table section, add a new row:

```markdown
| **julia-ml-hpc** | Julia ML training (Lux.jl supervised), GPU kernels, distributed computing, MLJ.jl pipelines | "Train a CNN in Julia", "Scale to cluster" |
```

- [ ] **Step 3: Add delegation row to `neural-network-master.md`**

Find the Delegation Strategy/Table section and add:

```markdown
| **julia-ml-hpc** | Julia-specific DL implementation (Lux.jl/Flux.jl architectures, training) | "Implement this transformer in Lux.jl" |
```

- [ ] **Step 4: Add delegation row to `ml-expert.md`**

Find the Delegation Strategy/Table section and add:

```markdown
| **julia-ml-hpc** | Julia ML pipelines (MLJ.jl), experiment management (DrWatson.jl) | "Build ML pipeline in Julia" |
```

- [ ] **Step 5: Add delegation row to `simulation-expert.md`**

Find the Delegation Strategy section and add:

```markdown
| **julia-ml-hpc** | Julia GPU kernels (CUDA.jl) and HPC distributed computing (MPI.jl) | "Write CUDA kernel in Julia", "Scale to SLURM cluster" |
```

- [ ] **Step 6: Verify frontmatter still parses**

Run: `cd /Users/b80985/Projects/MyClaude && python3 -c "
import yaml, re
for agent in ['julia-pro', 'neural-network-master', 'ml-expert', 'simulation-expert']:
    content = open(f'plugins/science-suite/agents/{agent}.md').read()
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    fm = yaml.safe_load(match.group(1))
    assert fm is not None, f'{agent} frontmatter broken'
    print(f'OK: {agent}')
"`

Expected: All 4 agents print OK.

- [ ] **Step 7: Commit**

```bash
git add plugins/science-suite/agents/julia-pro.md plugins/science-suite/agents/neural-network-master.md plugins/science-suite/agents/ml-expert.md plugins/science-suite/agents/simulation-expert.md
git commit -m "feat(science-suite): update 4 agents with julia-ml-hpc delegation rows"
```

---

## Task 13: Update existing skills with Julia cross-references

**Files:**
- Modify: `plugins/science-suite/skills/deep-learning/SKILL.md`
- Modify: `plugins/science-suite/skills/neural-architecture-patterns/SKILL.md`
- Modify: `plugins/science-suite/skills/training-diagnostics/SKILL.md`
- Modify: `plugins/science-suite/skills/machine-learning/SKILL.md`
- Modify: `plugins/science-suite/skills/parallel-computing/SKILL.md`
- Modify: `plugins/science-suite/skills/gpu-acceleration/SKILL.md`
- Modify: `plugins/science-suite/skills/model-deployment-serving/SKILL.md`

- [ ] **Step 1: Update `deep-learning/SKILL.md`**

After the existing Expert Agents section entries (after the `ml-expert` entry), add:

```markdown
- **`julia-ml-hpc`**: Julia DL implementation with Lux.jl/Flux.jl.
  - *Location*: `plugins/science-suite/agents/julia-ml-hpc.md`
  - *Capabilities*: Lux.jl training, GPU acceleration, Julia neural architectures.
  - *Julia skill*: See `julia-neural-networks` for Julia-specific deep learning.
```

- [ ] **Step 2: Update `neural-architecture-patterns/SKILL.md`**

Add after the title section (before `## Core Patterns`):

```markdown
## Julia Equivalent

For neural architecture implementation in Julia using Lux.jl, see the `julia-neural-architectures` skill.
```

- [ ] **Step 3: Update `training-diagnostics/SKILL.md`**

Add after the title (before `## Quick Reference`):

```markdown
## Julia Equivalent

For training diagnostics in Julia (Lux.jl, Zygote/Enzyme debugging), see the `julia-training-diagnostics` skill.
```

- [ ] **Step 4: Update `machine-learning/SKILL.md`**

In the Expert Agent section, after the existing `ml-expert` entry, add:

```markdown
- **`julia-ml-hpc`**: Julia ML pipelines with MLJ.jl and DrWatson.jl.
  - *Location*: `plugins/science-suite/agents/julia-ml-hpc.md`
  - *Julia skill*: See `julia-ml-pipelines` for Julia-specific ML workflows.
```

- [ ] **Step 5: Update `parallel-computing/SKILL.md`**

In the Expert Agent section, after the existing `simulation-expert` entry, add:

```markdown
- **`julia-ml-hpc`** (for Julia ML/HPC):
  - *Location*: `plugins/science-suite/agents/julia-ml-hpc.md`
  - *Capabilities*: Julia GPU kernels (CUDA.jl/KernelAbstractions.jl), MPI.jl, SLURM.
  - *Julia skills*: See `julia-hpc-distributed` and `julia-gpu-kernels`.
```

- [ ] **Step 6: Update `gpu-acceleration/SKILL.md`**

In the Expert Agent section, after the existing entries, add:

```markdown
- **`julia-ml-hpc`**: For advanced Julia GPU kernels, KernelAbstractions.jl, and multi-GPU with NCCL.jl.
  - *Julia skill*: See `julia-gpu-kernels` for detailed Julia GPU programming.
```

- [ ] **Step 7: Update `model-deployment-serving/SKILL.md`**

In the Expert Agent section, after the existing `ml-expert` entry, add:

```markdown
- **`julia-ml-hpc`**: Julia model deployment with Genie.jl, PackageCompiler.jl, and ONNX.jl.
  - *Julia skill*: See `julia-model-deployment` for Julia-specific deployment patterns.
```

- [ ] **Step 8: Verify all modified skills still parse**

Run: `cd /Users/b80985/Projects/MyClaude && python3 -c "
import yaml, re
skills = ['deep-learning', 'neural-architecture-patterns', 'training-diagnostics', 'machine-learning', 'parallel-computing', 'gpu-acceleration', 'model-deployment-serving']
for s in skills:
    content = open(f'plugins/science-suite/skills/{s}/SKILL.md').read()
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    fm = yaml.safe_load(match.group(1))
    assert fm is not None and 'name' in fm, f'{s} frontmatter broken'
    print(f'OK: {s}')
"`

Expected: All 7 skills print OK.

- [ ] **Step 9: Commit**

```bash
git add plugins/science-suite/skills/deep-learning/SKILL.md plugins/science-suite/skills/neural-architecture-patterns/SKILL.md plugins/science-suite/skills/training-diagnostics/SKILL.md plugins/science-suite/skills/machine-learning/SKILL.md plugins/science-suite/skills/parallel-computing/SKILL.md plugins/science-suite/skills/gpu-acceleration/SKILL.md plugins/science-suite/skills/model-deployment-serving/SKILL.md
git commit -m "feat(science-suite): add Julia cross-references to 7 existing skills"
```

---

## Task 14: Update plugin.json manifest

**Files:**
- Modify: `plugins/science-suite/.claude-plugin/plugin.json`

- [ ] **Step 1: Add new agent to agents array**

In the `"agents"` array, add after `"./agents/julia-pro.md"`:

```json
"./agents/julia-ml-hpc.md",
```

Keep the array alphabetically sorted. The final agents array should have 12 entries.

- [ ] **Step 2: Add 10 new skills to skills array**

In the `"skills"` array, add the following entries in alphabetical order. They should appear between `"./skills/jax-physics-applications"` and `"./skills/julia-mastery"`:

```json
"./skills/julia-ad-backends",
"./skills/julia-gpu-kernels",
"./skills/julia-graph-neural-networks",
"./skills/julia-hpc-distributed",
"./skills/julia-ml-pipelines",
"./skills/julia-model-deployment",
"./skills/julia-neural-architectures",
"./skills/julia-neural-networks",
"./skills/julia-reinforcement-learning",
"./skills/julia-training-diagnostics",
```

The final skills array should have 96 entries.

- [ ] **Step 3: Verify JSON is valid**

Run: `cd /Users/b80985/Projects/MyClaude && python3 -c "
import json
data = json.load(open('plugins/science-suite/.claude-plugin/plugin.json'))
agents = data['agents']
skills = data['skills']
assert './agents/julia-ml-hpc.md' in agents, 'Agent missing'
new_skills = [
    './skills/julia-ad-backends', './skills/julia-gpu-kernels',
    './skills/julia-graph-neural-networks', './skills/julia-hpc-distributed',
    './skills/julia-ml-pipelines', './skills/julia-model-deployment',
    './skills/julia-neural-architectures', './skills/julia-neural-networks',
    './skills/julia-reinforcement-learning', './skills/julia-training-diagnostics'
]
for s in new_skills:
    assert s in skills, f'Skill missing: {s}'
assert len(agents) == 12, f'Expected 12 agents, got {len(agents)}'
assert len(skills) == 96, f'Expected 96 skills, got {len(skills)}'
print(f'OK: {len(agents)} agents, {len(skills)} skills')
"`

Expected: `OK: 12 agents, 96 skills`

- [ ] **Step 4: Commit**

```bash
git add plugins/science-suite/.claude-plugin/plugin.json
git commit -m "feat(science-suite): update plugin.json with julia-ml-hpc agent and 10 new skills"
```

---

## Task 15: Run validation suite

**Files:** (no files modified -- validation only)

- [ ] **Step 1: Run science-suite integrity tests**

Run: `cd /Users/b80985/Projects/MyClaude && uv run pytest tools/tests/test_science_suite_integrity.py -v`

Expected: All tests pass:
- `test_plugin_exists` -- plugin directory found
- `test_plugin_json_structure` -- plugin.json valid
- `test_agents_exist` -- all 12 agent files exist
- `test_agent_frontmatter_validity` -- all YAML frontmatter parses

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/b80985/Projects/MyClaude && uv run pytest tools/tests/ -v`

Expected: All tests pass (60+ tests).

- [ ] **Step 3: Run context budget checker**

Run: `cd /Users/b80985/Projects/MyClaude && python3 tools/validation/context_budget_checker.py`

Expected: All 142 skills within 2% budget.

- [ ] **Step 4: Run metadata validator**

Run: `cd /Users/b80985/Projects/MyClaude && python3 tools/validation/metadata_validator.py plugins/science-suite`

Expected: No validation errors.

- [ ] **Step 5: If any tests fail, fix the issues**

Read error output, identify failing assertion, fix the relevant file, re-run.

- [ ] **Step 6: Final commit if any fixes were needed**

```bash
git add -u
git commit -m "fix(science-suite): resolve validation issues from julia-ml-hpc expansion"
```

---

## Task 16: Update documentation counts

**Files:**
- Modify: `CLAUDE.md` (project root)

- [ ] **Step 1: Update CLAUDE.md counts**

Update the Project Overview line from:

```
MyClaude is a Claude Code plugin marketplace: 3 plugin suites containing 23 agents, 33 commands, and 132 skills.
```

to:

```
MyClaude is a Claude Code plugin marketplace: 3 plugin suites containing 24 agents, 33 commands, and 142 skills.
```

Update the Suite Breakdown table row for science-suite from:

```
| science-suite | 11 | 0 | 86 | 0 | JAX, Julia, physics/chemistry simulations, and data science workflows |
```

to:

```
| science-suite | 12 | 0 | 96 | 0 | JAX, Julia, physics/chemistry simulations, ML/DL/HPC, and data science workflows |
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update counts to 24 agents, 142 skills after julia-ml-hpc expansion"
```
