---
name: graph-theory
description: Graph theory foundations — spectral graph theory, graph algorithms, and network topology metrics. Use when the task requires reasoning about a graph/network structure itself (spectral properties, centrality, connectivity, community structure) rather than a specific application domain (GNNs, coupled oscillators, FEM meshes) that happens to use a graph representation.
---

# Graph Theory

Cross-cutting graph-theoretic foundations used across GNN, network-dynamics, and mesh-representation work.

## Expert Agents

Consulted by multiple agents depending on the application: `neural-network-master` (GNN architecture), `nonlinear-dynamics-expert` (coupled-oscillator network topology), `continuum-mechanics-engineer` (FEM mesh connectivity).

## Core Concepts

| Concept | Definition | Application |
|---------|------------|-------------|
| Adjacency/Laplacian matrix | `A` (connections), `L = D - A` (degree-adjacency) | Spectral clustering, GNN message passing, diffusion on networks |
| Spectral graph theory | Eigenvalues/eigenvectors of `L` | Graph partitioning (Fiedler vector), synchronization thresholds in coupled-oscillator networks |
| Centrality measures | Degree, betweenness, eigenvector centrality | Identifying influential nodes in a network |
| Connectivity | Vertex/edge connectivity, connected components | Percolation threshold (structural side — see `science-suite:glass-and-collective-dynamics` for the statistical-mechanics side), network robustness |
| Community structure | Modularity, spectral/hierarchical clustering | Detecting mesoscale structure in large networks |

## Spectral Graph Theory Quick Reference

- The Laplacian `L`'s smallest eigenvalue is always 0 (with eigenvector = all-ones); the number of zero eigenvalues equals the number of connected components.
- The second-smallest eigenvalue (algebraic connectivity, "Fiedler value") governs how well-connected a graph is — small values indicate near-disconnection or bottlenecks; its eigenvector (Fiedler vector) gives a natural graph bipartition for spectral clustering.
- For coupled-oscillator networks, the Laplacian eigenvalue spectrum directly determines synchronization stability (via the master stability function) — this is the graph-theory input `nonlinear-dynamics-expert` needs for synchronization analysis.

## Application-Specific Delegation

| Application | Delegate To | This skill provides |
|-------------|-------------|------------------------|
| GNN architecture (message passing, pooling) | `neural-network-master` | Adjacency/Laplacian formalism underlying message-passing operators |
| Coupled-oscillator synchronization | `nonlinear-dynamics-expert` | Laplacian spectrum for master-stability-function analysis |
| FEM mesh connectivity/quality | `continuum-mechanics-engineer` | Graph representation of mesh topology, connectivity checks |
| Julia-native GNN implementation | `julia-ml-hpc` | GNNGraphs.jl's graph representation conventions |

This skill is deliberately thin — it holds the shared mathematical vocabulary so the 4 application-specific agents above don't each re-explain spectral graph theory independently. Route here first when a question is about the graph structure itself, then to the application-specific agent for what to do with it.
