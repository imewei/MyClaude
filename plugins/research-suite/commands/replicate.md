---
name: replicate
description: End-to-end replication pipeline — fetch paper, extract claims, implement in JAX or Julia, validate outputs against reported numbers within tolerance.
argument-hint: "[--paper arxiv-id|doi] [--tolerance 0.01] [--framework jax|julia]"
allowed-tools: ["Read", "Write", "Edit", "Bash", "WebFetch", "WebSearch"]
---

# /replicate — End-to-End Paper Replication

Routes to `research-expert` (claim extraction) → `research-spark-orchestrator` (pipeline) → `jax-pro` or `julia-pro` (implementation).

## Usage

```
/replicate --paper 2301.04567 --framework jax --tolerance 0.01
/replicate --paper 10.1038/s41586-021-03819-2 --framework julia --tolerance 0.05
```

## What This Does

1. Fetches paper via arXiv ID or DOI
2. `research-expert` extracts falsifiable claims and key numerical results
3. `research-spark-orchestrator` structures the replication as a staged pipeline
4. `jax-pro` / `julia-pro` implement the core method
5. Runs experiment and compares outputs to reported numbers within `--tolerance`
6. Produces a replication report noting exact match, within-tolerance match, or deviation

## Tolerance

`--tolerance` is the relative L2 error threshold (default `0.01` = 1%). Results within tolerance are marked ✓; deviations are flagged with the actual vs reported values.

## Turn Strategy

Claim extraction and implementation run as separate turns to avoid context overflow on large papers. Only the active turn's context loads into the window.
