---
name: science-suite
description: Meta-router for all scientific computing tasks — JAX, Julia, SciML, nonlinear dynamics, Bayesian inference, statistical physics, deep learning, ML, simulation, HPC, LLM integration, and research domains. Use when any scientific, computational, or ML task is requested.
---

# Science Suite

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
+-- Dynamical systems / chaos / SINDy / bifurcation?
|   --> nonlinear-dynamics
|
+-- JAX code, JIT, vmap, pmap, custom VJP, JAX-MD?
|   --> jax-computing
|
+-- Julia language, packaging, dispatch, type stability?
|   --> julia-language
|
+-- Julia ML/DL: Lux, MLJ, CUDA.jl, GNN?
|   --> julia-ml-and-dl
|
+-- DiffEq.jl, ModelingToolkit, UDEs, Turing, SciML?
|   --> sciml-and-diffeq
|
+-- Correlation functions (MSD, VACFs, g(r), DLS, DDM)?
|   --> correlation-analysis
|
+-- Phase transitions, Langevin, stat mech, soft matter?
|   --> statistical-physics-hub
|
+-- Neural architecture, CNNs/Transformers/GNNs, training?
|   --> deep-learning-hub
|
+-- scikit-learn, XGBoost, feature engineering, MLflow?
|   --> ml-and-data-science
|
+-- LLM integration, RAG, AI-assisted science, codegen?
|   --> llm-and-ai
|
+-- Serving, ONNX, TensorRT, deployment pipeline?
|   --> ml-deployment
|
+-- MD, Monte Carlo, GROMACS/OpenMM, GPU physics, HPC?
|   --> simulation-and-hpc
|
+-- Bioinformatics, quantum, control theory, signal proc?
|   --> research-and-domains
|
+-- MCMC, NUTS, variational inference, posterior diagnostics?
    --> bayesian-inference
```

## Routing Table

| Trigger | Hub skill |
|---|---|
| bifurcation, chaos, Lyapunov, SINDy, attractor, chimera | nonlinear-dynamics |
| JAX, JIT, vmap, pmap, custom_vjp, JAX-MD, Diffrax | jax-computing |
| Julia basics, dispatch, type stability, packaging, Pkg | julia-language |
| Lux, MLJ, CUDA.jl, GNNLux, Julia neural network | julia-ml-and-dl |
| DiffEq.jl, MTK, ModelingToolkit, UDE, Turing, SciML | sciml-and-diffeq |
| correlation function, MSD, VACF, g(r), DLS, DDM, rheology | correlation-analysis |
| phase transition, Langevin, stat mech, free energy, soft matter | statistical-physics-hub |
| Transformer, CNN, GNN, neural architecture, training diagnostics | deep-learning-hub |
| scikit-learn, XGBoost, feature engineering, MLflow, pandas | ml-and-data-science |
| LLM, RAG, Claude API, AI workflow, codegen, embeddings | llm-and-ai |
| ONNX, TensorRT, model serving, deployment, inference server | ml-deployment |
| GROMACS, OpenMM, molecular dynamics, Monte Carlo, HPC, MPI | simulation-and-hpc |
| bioinformatics, quantum, control theory, signal processing, EEG | research-and-domains |
| NUTS, MCMC, NumPyro, Turing, ArviZ, variational inference | bayesian-inference |

## Checklist

- [ ] Identify the primary domain using the routing decision tree before invoking any hub skill
- [ ] If the task spans two hubs (e.g., Bayesian + SciML), start with the inference hub and delegate computation to the SciML hub
- [ ] Confirm the target ecosystem (JAX/Python vs. Julia) before selecting a sub-skill within the hub
- [ ] Delegate to the appropriate expert agent for problems requiring deep domain judgment
- [ ] Verify that hardware constraints (CPU/GPU/memory) are compatible with the chosen hub's default tools
- [ ] For cross-language workflows, check the jax-julia-interop skill under jax-computing or julia-language
- [ ] After routing, read the hub skill's own routing tree before touching any code
- [ ] Do not invoke multiple hub skills simultaneously — resolve one hub's output before chaining to the next
