---
name: science-hub
description: >-
  Top-level router for all scientific computing topics. Use for: JAX/NumPyro/Diffrax/JIT/vmap/pmap/custom-VJP/optax; Julia language/packages/SciML/DiffEq.jl/Lux/ModelingToolkit; nonlinear dynamics/bifurcation/chaos/Lyapunov/SINDy/attractors/coupled oscillators/pattern formation; correlation functions/DLS/SAXS/XPCS/FFT correlators/MSD/VACF/g(r); statistical physics/non-equilibrium/stochastic dynamics/active matter/multiscale/rare-event sampling/extreme-value statistics; deep learning architecture/transformers/CNNs/GNNs/training diagnostics/gradient explosion; classical ML/scikit-learn/XGBoost/data analysis/wrangling/scientific visualization; LLM applications/RAG/LangChain/NLP/evaluation; ML deployment/FastAPI/TorchServe/MLOps/quantization/federated learning; MD simulation/HPC/GPU kernels/GROMACS/OpenMM/numerical methods/trajectory analysis/ML force fields; scientific Python packaging/Rust extensions/type-driven design/quantum computing/bioinformatics/RL/symbolic math; Bayesian inference/MCMC/NumPyro/Turing.jl/Pigeons/consensus MCMC/Bayesian UDE/PINNs/variational inference/MCMC diagnostics.
---

# Science Suite (science-hub)

Top-level entry point for the science-suite plugin. Identifies the correct hub skill and delegates.

## Expert Agents

Located in `plugins/science-suite/agents/`:

- **`jax-pro`** — JAX/XLA expert: JIT, vmap, pmap, custom VJPs, NumPyro
- **`julia-pro`** — Core Julia: type stability, packaging, dispatch, interop
- **`julia-ml-hpc`** — Julia ML + HPC: Lux, GNNLux, CUDA.jl, distributed
- **`ml-expert`** — Classical ML: scikit-learn, XGBoost, MLflow pipelines
- **`neural-network-master`** — Deep learning: architectures, training diagnostics
- **`nonlinear-dynamics-expert`** — Dynamical systems, chaos, SINDy, bifurcations
- **`pinn-engineer`** — PINNs, NeuralPDE, BPINNs, MethodOfLines
- **`python-pro`** — Python toolchain, packaging, async, performance
- **`sci-workflow-engineer`** — End-to-end scientific pipelines, CI, reproducibility
- **`simulation-expert`** — MD, Monte Carlo, GPU-accelerated physics
- **`statistical-physicist`** — Langevin, fluctuation theorems, soft matter, phase transitions

## Hub Skills

- [Nonlinear Dynamics](../nonlinear-dynamics/SKILL.md) — Bifurcations, chaos, Lyapunov, SINDy, chimera states
- [JAX Computing](../jax-computing/SKILL.md) — JAX/JIT/vmap/pmap, custom VJPs, NumPyro, JAX-MD
- [Julia Language](../julia-language/SKILL.md) — Core Julia, packaging, testing, type-stable patterns
- [Julia ML & DL](../julia-ml-and-dl/SKILL.md) — MLJ, Flux/Lux, CUDA.jl, GNN in Julia
- [SciML & DiffEq](../sciml-and-diffeq/SKILL.md) — DiffEq.jl, ModelingToolkit, UDEs, Turing.jl
- [Correlation Analysis](../correlation-analysis/SKILL.md) — Correlation functions: math, physics, computational, experimental
- [Statistical Physics](../statistical-physics-hub/SKILL.md) — Phase transitions, Langevin, fluctuation theorems, soft matter
- [Deep Learning](../deep-learning-hub/SKILL.md) — Neural architectures, Transformers/CNNs/GNNs, training diagnostics
- [ML & Data Science](../ml-and-data-science/SKILL.md) — scikit-learn, XGBoost, feature engineering, MLflow
- [LLM & AI](../llm-and-ai/SKILL.md) — LLM integration into scientific workflows, RAG, codegen
- [ML Deployment](../ml-deployment/SKILL.md) — Model serving, optimization, deployment pipelines
- [Simulation & HPC](../simulation-and-hpc/SKILL.md) — MD (GROMACS/OpenMM/JAX-MD), Monte Carlo, GPU physics
- [Research & Domains](../research-and-domains/SKILL.md) — Bioinformatics, quantum, control theory, signal processing
- [Bayesian Inference](../bayesian-inference/SKILL.md) — NumPyro NUTS, Turing, MCMC diagnostics, variational inference

## Routing Decision Tree

```
What is the primary task?
|
+-- Dynamical systems / chaos / SINDy / bifurcation / coupled oscillators / pattern formation / synchronization?
|   --> science-suite:nonlinear-dynamics
|
+-- JAX code, JIT, vmap, pmap, custom VJP, JAX-MD?
|   --> science-suite:jax-computing
|
+-- Julia language, packaging, dispatch, type stability?
|   --> science-suite:julia-language
|
+-- Julia ML/DL: Lux, MLJ, CUDA.jl, GNN?
|   --> science-suite:julia-ml-and-dl
|
+-- DiffEq.jl, ModelingToolkit, UDEs, Turing, SciML?
|   --> science-suite:sciml-and-diffeq
|
+-- Correlation functions (MSD, VACFs, g(r), DLS, DDM, SAXS, XPCS, scattering, spectroscopy)?
|   --> science-suite:correlation-analysis
|
+-- Phase transitions / Langevin / Fokker-Planck / stochastic dynamics / non-equilibrium / active matter / rare events / extreme value statistics / soft matter?
|   --> science-suite:statistical-physics-hub
|
+-- Neural architecture, CNNs/Transformers/GNNs, training?
|   --> science-suite:deep-learning-hub
|
+-- scikit-learn, XGBoost, feature engineering, MLflow?
|   --> science-suite:ml-and-data-science
|
+-- LLM integration, RAG, AI-assisted science, codegen?
|   --> science-suite:llm-and-ai
|
+-- Serving, ONNX, TensorRT, deployment pipeline?
|   --> science-suite:ml-deployment
|
+-- MD / Monte Carlo / GROMACS/OpenMM / GPU physics / HPC / signal processing / FFT / time series / control theory / numerical methods?
|   --> science-suite:simulation-and-hpc
|
+-- Bioinformatics / quantum computing / computer vision / RL / symbolic math / DSPy / RLAIF / self-improving AI / scientific Python / Rust extensions?
|   --> science-suite:research-and-domains
|
+-- MCMC, NUTS, variational inference, posterior diagnostics?
|   --> science-suite:bayesian-inference
|
+-- None of the above / concern is ambiguous?
    --> Identify whether the task is Python/JAX or Julia, then re-enter the
        routing tree. If still unclear, delegate to `jax-pro` or `julia-pro`.
```

## Routing Table

| Trigger | Hub skill |
|---|---|
| bifurcation, chaos, Lyapunov, SINDy, attractor, chimera, coupled oscillators, synchronization, Turing patterns, pattern formation, network dynamics | science-suite:nonlinear-dynamics |
| JAX, JIT, vmap, pmap, custom_vjp, JAX-MD, Diffrax, optax, NumPyro JAX, JAX physics, XLA | science-suite:jax-computing |
| Julia basics, dispatch, type stability, packaging, Pkg, Makie, Plots.jl, visualization, Julia CI/CD, Julia testing, Julia HPC, MPI.jl, Distributed.jl, PyCall, interop, Genie.jl | science-suite:julia-language |
| Lux, Flux.jl, MLJ, CUDA.jl, GNNLux, Julia neural network, Enzyme, Zygote, KernelAbstractions, Julia RL, Julia model export, ONNX Julia | science-suite:julia-ml-and-dl |
| DiffEq.jl, MTK, ModelingToolkit, UDE, Turing, SciML, NeuralPDE, PINN, Catalyst.jl, JuMP, DataDrivenDiffEq, SINDy Julia, bifurcation Julia | science-suite:sciml-and-diffeq |
| correlation function, MSD, VACF, g(r), DLS, DDM, SAXS, XPCS, scattering, structure factor, spectroscopy, rheology, Green's function, FDT | science-suite:correlation-analysis |
| phase transition, Langevin, stat mech, free energy, soft matter, Fokker-Planck, stochastic dynamics, non-equilibrium, entropy production, active matter, MIPS, flocking, coarse-graining, rare events, FFS, TIS, extreme value, GEV, GPD, SOC, avalanche | science-suite:statistical-physics-hub |
| Transformer, CNN, GNN, neural architecture, training diagnostics, backpropagation, loss divergence, gradient explosion, gradient vanishing, ablation, hyperparameter search, distributed training, multi-GPU | science-suite:deep-learning-hub |
| scikit-learn, XGBoost, feature engineering, MLflow, pandas, curve fitting, NLSQ, EDA, statistical tests, hypothesis testing, scientific visualization, matplotlib, seaborn, W&B | science-suite:ml-and-data-science |
| LLM, RAG, Claude API, AI workflow, codegen, embeddings, LangChain, LangGraph, NLP, tokenization, NER, LLM evaluation, LLM-as-judge | science-suite:llm-and-ai |
| ONNX, TensorRT, model serving, deployment, inference server, FastAPI serving, TorchServe, Triton, BentoML, quantization, pruning, Airflow ML, Prefect, K8s GPU, federated learning, differential privacy | science-suite:ml-deployment |
| GROMACS, OpenMM, LAMMPS, molecular dynamics, Monte Carlo, HPC, MPI, trajectory analysis, MDAnalysis, ML force fields, NequIP, MACE, signal processing, FFT, filtering, time series, ARIMA, changepoint detection, control theory, PID, LQR, MPC, numerical methods, FEM | science-suite:simulation-and-hpc |
| bioinformatics, genomics, quantum computing, VQE, QAOA, computer vision, reinforcement learning, symbolic math, SymPy, DSPy, RLAIF, self-improving AI, Rust extensions, type-driven design, scientific Python packaging | science-suite:research-and-domains |
| NUTS, MCMC, NumPyro, Turing, ArviZ, variational inference, Pigeons, parallel tempering, multimodal posterior, Hawkes process, point processes, Bayesian SINDy, credible intervals, BlackJAX, R-hat, ESS | science-suite:bayesian-inference |

## Checklist

- [ ] Identify the primary domain using the routing decision tree before invoking any hub skill
- [ ] If the task spans two hubs (e.g., Bayesian + SciML), start with the inference hub and delegate computation to the SciML hub
- [ ] Confirm the target ecosystem (JAX/Python vs. Julia) before selecting a sub-skill within the hub
- [ ] Delegate to the appropriate expert agent for problems requiring deep domain judgment
- [ ] Verify that hardware constraints (CPU/GPU/memory) are compatible with the chosen hub's default tools
- [ ] For cross-language workflows, check the jax-julia-interop skill under jax-computing or julia-language
- [ ] After routing, read the hub skill's own routing tree before touching any code
- [ ] Do not invoke multiple hub skills simultaneously — resolve one hub's output before chaining to the next
