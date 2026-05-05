---
name: paper-implement
description: Reproduce a paper's core methods in JAX or Julia — parse equations, scaffold implementation, wire up experiment, validate outputs.
argument-hint: "[--paper path/to/pdf|arxiv-id] [--framework jax|julia] [--section methods|experiments|all]"
allowed-tools: ["Read", "Write", "Edit", "Bash", "WebFetch"]
---

# /paper-implement — Paper Method Reproduction

Routes to `research-expert` for methodology parsing, then cross-delegates to `jax-pro` (JAX) or `julia-pro` (Julia) for implementation.

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

`research-expert` owns methodology parsing. Implementation delegated to `jax-pro` (JAX) or `julia-pro` (Julia) via cross-suite call.
