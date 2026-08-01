---
name: replicate
description: End-to-end replication pipeline — fetch paper, extract claims, implement in JAX or Julia, validate outputs against reported numbers within tolerance.
argument-hint: "[--paper arxiv-id|doi] [--tolerance 0.01] [--framework jax|julia]"
allowed-tools: ["Read", "Write", "Edit", "Bash", "WebFetch", "WebSearch"]
---

# /replicate — End-to-End Paper Replication

Routes to `research-expert` (claim extraction and replication design) → the specialist matching the paper's method (agent `jax-pro`, agent `julia-pro`, agent `continuum-mechanics-engineer`, agent `statistical-physicist`, agent `pinn-engineer`, or agent `simulation-expert` — implementation) → `quality-specialist` (numerical validation gates).

## Usage

```bash
/replicate --paper 2301.04567 --framework jax --tolerance 0.01
/replicate --paper 10.1038/s41586-021-03819-2 --framework julia --tolerance 0.05
```

## What This Does

1. Fetches paper via arXiv ID or DOI
2. `research-expert` extracts falsifiable claims, key numerical results, and designs the replication plan
3. The specialist matching the paper's method implements the core method (agent `jax-pro`/agent `julia-pro` for general numerics, agent `continuum-mechanics-engineer` for FEM/rheology/materials, agent `statistical-physicist` for stat-mech/glass/physical-learning, agent `pinn-engineer` for physics-informed neural PDEs, agent `simulation-expert` for MD/HPC particle simulation)
4. `quality-specialist` validates numerical outputs against reported numbers within `--tolerance`
5. Produces a replication report noting exact match, within-tolerance match, or deviation

## Tolerance

`--tolerance` is the relative L2 error threshold (default `0.01` = 1%). Results within tolerance are marked ✓; deviations are flagged with the actual vs reported values.

## Turn Strategy

Claim extraction and implementation run as separate turns to avoid context overflow on large papers. Only the active turn's context loads into the window.
