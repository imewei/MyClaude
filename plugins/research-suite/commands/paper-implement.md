---
name: paper-implement
description: Implement a paper's core method in JAX or Julia — parse equations, scaffold the algorithm, wire up a minimal experiment, and run smoke validation. For full claim extraction and tolerance-based replication against reported numbers, use /replicate.
argument-hint: "[--paper path/to/pdf|arxiv-id] [--framework jax|julia] [--section methods|experiments|all]"
allowed-tools: ["Read", "Write", "Edit", "Bash", "WebFetch"]
---

# /paper-implement — Paper Method Reproduction

Routes to `research-expert` for methodology parsing, then cross-delegates to the specialist matching the paper's method: agent `jax-pro` (general JAX numerics), agent `julia-pro` (general Julia numerics), agent `continuum-mechanics-engineer` (FEM/FEA, constitutive modeling, rheology/DMA, transient networks, nanocomposites), agent `statistical-physicist` (phase transitions, correlations, glass/collective phenomena, physical learning), agent `pinn-engineer` (physics-informed neural networks, NeuralPDE), or agent `simulation-expert` (MD/HPC particle simulation).

## Usage

```
/paper-implement --paper 2301.04567 --framework jax --section methods
/paper-implement --paper /path/to/diffusion_model.pdf --framework julia --section all
```

## What This Does

1. Fetches or reads the paper (`--paper`)
2. Extracts core equations and algorithmic steps from `--section`
3. Scaffolds implementation in `--framework` with proper structure
4. Wires up a minimal experiment reproducing the paper's key result
5. Notes any discrepancies with reported numbers

## Section Routing

| `--section` | Loads |
|---|---|
| `methods` | Equation extraction + implementation only |
| `experiments` | Experiment setup + validation against reported numbers |
| `all` | Both phases sequentially |

## Framework Delegation

`research-expert`'s methodology-parsing step determines the paper's domain and picks the specialist (see the routing list above); `--framework` (where applicable) only disambiguates JAX vs. Julia within a numerics-based specialist, not which specialist handles the paper.
