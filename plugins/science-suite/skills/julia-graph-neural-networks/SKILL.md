---
name: julia-graph-neural-networks
description: Build graph neural networks in Julia with GraphNeuralNetworks.jl and Lux.jl. Covers GCN, GAT, GraphSAGE, message passing neural networks, node/edge/graph-level tasks, heterogeneous graphs, temporal graphs (TemporalGNNs), and mini-batch training on large graphs. Use when working with graph-structured data in Julia.
---

# Julia Graph Neural Networks

## Mode Flag

- `--mode quick`: routing table + agent delegation only
- `--mode standard` (default): task types, architecture overview, framework comparison
- `--mode deep`: canonical Lux training loop and custom message passing code

## Expert Agent

For graph neural network architecture and training in Julia, delegate to:

- **`julia-ml-hpc`**: Julia ML/HPC specialist for GraphNeuralNetworks.jl ecosystem (GNNGraphs / GNNlib / GNNLux), message passing, and GPU-portable graph training.
  - *Location*: `plugins/science-suite/agents/julia-ml-hpc.md`

## Graph Construction

Create graphs with `GNNGraph`:

```julia
using GraphNeuralNetworks

# From edge list
src = [1, 1, 2, 3, 3, 4]
dst = [2, 3, 4, 4, 5, 5]
g = GNNGraph(src, dst)

# From adjacency matrix
A = [0 1 1 0; 1 0 0 1; 1 0 0 1; 0 1 1 0]
g = GNNGraph(A)

# From Graphs.jl
using Graphs
lg = barabasi_albert(100, 3)
g = GNNGraph(lg)

# Add node and edge features
g = GNNGraph(src, dst;
    ndata=(; x=randn(Float32, 16, 5)),    # 16-dim features, 5 nodes
    edata=(; w=randn(Float32, 1, 6)))      # 1-dim edge weights, 6 edges
```

## Built-in GNN Layers

| Layer | Description | Key Parameters |
|-------|-------------|----------------|
| `GCNConv` | Graph Convolutional Network | `in => out` |
| `GATConv` | Graph Attention Network | `in => out`, `heads` |
| `SAGEConv` | GraphSAGE (sampling + aggregation) | `in => out`, `aggr` |
| `GINConv` | Graph Isomorphism Network | `nn` (MLP) |
| `EdgeConv` | Dynamic Edge Convolution | `nn` (MLP) |
| `GatedGraphConv` | Gated Graph Neural Network | `out`, `num_layers` |

## GCN Example

```julia
using GraphNeuralNetworks, Lux, Random

# Build GCN model with GNNChain
model = GNNChain(
    GCNConv(16 => 64, relu),
    GCNConv(64 => 32, relu),
    Dense(32, 7)               # Output layer for 7 classes
)

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

# Forward pass
g = GNNGraph(src, dst; ndata=(; x=randn(Float32, 16, num_nodes)))
y, st = model(g, g.x, ps, st)   # y: (7, num_nodes)
```

## GAT with Multi-Head Attention

```julia
model = GNNChain(
    GATConv(16 => 8, heads=4, concat=true),   # Output: 32 (8 * 4 heads)
    GATConv(32 => 8, heads=1, concat=false),   # Output: 8 (single head)
    Dense(8, 7)
)
```

## GNNLux.jl Explicit-Parameter Integration

For explicit-parameter style consistent with Lux.jl:

```julia
using GNNLux

model = GNNChain(
    GCNConv(16 => 64, relu),
    Dropout(0.5),
    GCNConv(64 => 7)
)

ps, st = Lux.setup(rng, model)
y, st = model(g, g.x, ps, st)
```

## Task Types

### Node Classification

```julia
function train_node_classification!(model, g, ps, st, opt_state, train_mask)
    (loss, st), grads = Zygote.withgradient(ps) do p
        y_hat, st_ = model(g, g.x, p, st)
        logitcrossentropy(y_hat[:, train_mask], g.ndata.y[:, train_mask]), st_
    end
    opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
    return ps, st, opt_state, loss
end
```

### Link Prediction

```julia
function link_score(model, g, ps, st, src, dst)
    h, st = model(g, g.x, ps, st)
    # Dot product between node embeddings
    scores = sum(h[:, src] .* h[:, dst]; dims=1)
    return sigmoid.(scores)
end
```

### Graph Classification with GlobalPool

```julia
model = GNNChain(
    GCNConv(16 => 64, relu),
    GCNConv(64 => 32, relu),
    GlobalPool(mean),             # Aggregate node features to graph-level
    Dense(32, 2)                  # Binary graph classification
)
```

## Mini-Batching

Batch multiple graphs for efficient training. The canonical batching function is `MLUtils.batch` (`GraphNeuralNetworks` re-exports it, but the namespace makes the dependency explicit and works on heterogeneous graph lists too):

```julia
using MLUtils

graphs = [rand_graph(n, 2n; ndata=(; x=randn(Float32, 16, n))) for n in [10, 20, 15]]
batched_g = MLUtils.batch(graphs)

# Forward pass on batch (nodes are concatenated, edges re-indexed)
y, st = model(batched_g, batched_g.x, ps, st)

# Unbatch results
individual_graphs = unbatch(batched_g)
```

> **--mode deep required** for training loop and message passing code below.

## Canonical Lux training loop

The cleanest GNNLux training pattern uses `Lux.Training.TrainState` plus `single_train_step!` — much shorter than rolling your own `Zygote.withgradient` and it threads `(ps, st)` through optimiser updates automatically:

```julia
using Lux, Optimisers, MLUtils

train_state = Lux.Training.TrainState(model, ps, st, Adam(1e-2))

custom_loss(model, ps, st, (g, x, y)) = let
    ŷ, st_new = model(g, x, ps, st)
    loss = logitcrossentropy(ŷ, y)
    loss, st_new, (;)            # (loss, new_state, stats)
end

for epoch in 1:200
    for (g, y) in train_loader
        g = MLUtils.batch(g)
        _, loss, _, train_state = Lux.Training.single_train_step!(
            AutoZygote(), custom_loss, (g, g.ndata.x, y), train_state,
        )
    end

    # Switch Dropout / BatchNorm into eval mode for validation
    st_eval = Lux.testmode(train_state.states)
    val_acc = evaluate(model, train_state.parameters, st_eval, val_loader)
    train_state = @set train_state.states = Lux.trainmode(st_eval)
end
```

`Lux.testmode(st)` and `Lux.trainmode(st)` flip stochastic regularizers (Dropout, BatchNorm running stats) on and off — required whenever a `GNNChain` contains `Dropout` or normalisation layers, otherwise validation metrics will be biased by training-mode noise.

## Custom Message Passing

Implement custom message functions with `propagate`:

```julia
using GraphNeuralNetworks: propagate

function custom_conv(g::GNNGraph, x)
    # Message function: applied to each edge
    message(xi, xj, e) = xj .* e  # Weight neighbor features by edge attr

    # Aggregate messages at each node
    m = propagate(message, g, +, xj=x, e=g.edata.w)

    return relu.(m)
end
```

## Molecular Property Prediction

End-to-end example for molecular graphs:

```julia
using GraphNeuralNetworks, Lux

# Molecular GNN: atoms as nodes, bonds as edges
mol_model = GNNChain(
    # Atom embedding
    Embedding(118, 64),           # 118 elements
    # Message passing
    GCNConv(64 => 128, relu),
    GCNConv(128 => 128, relu),
    GCNConv(128 => 64, relu),
    # Readout
    GlobalPool(mean),             # Graph-level representation
    Dense(64, 32, relu),
    Dense(32, 1)                  # Scalar property prediction
)

# Build molecular graph
mol_graph = GNNGraph(
    bond_src, bond_dst;
    ndata=(; z=atomic_numbers),    # Atomic numbers as node features
    edata=(; bond_type=bond_types) # Bond types as edge features
)

y_pred, st = mol_model(mol_graph, mol_graph.ndata.z, ps, st)
```

## Package layering: GNNGraphs, GNNlib, GNNLux

The Julia GNN ecosystem splits cleanly into three layers — pick the layer that matches your need:

| Package | Layer | Use when |
|---------|-------|----------|
| `GNNGraphs` | Data structure | Building, batching, querying graphs. Provides `GNNGraph`, heterogeneous graphs (`GNNHeteroGraph`), temporal graphs (`TemporalSnapshotsGNNGraph`). No NN code. |
| `GNNlib` | Low-level kernels | Writing custom message-passing layers. Provides `propagate`, aggregation primitives, sparse-matrix message kernels. Framework-agnostic. |
| `GNNLux` | Lux-native layer collection | High-level GCN/GAT/SAGE/GIN/GatedGraph layers exposing Lux's explicit `(ps, st)` interface. Composable inside `GNNChain`. |

Most application code lives in `GNNLux` + `GNNGraphs`. Drop into `GNNlib` only when an existing layer doesn't fit and you need custom message semantics.

## Benchmark datasets

Use `MLDatasets` for canonical graph benchmarks (Cora, CiteSeer, PubMed, OGB-arXiv, ZINC, QM9). Combine with `OneHotArrays` for label encoding:

```julia
using MLDatasets, OneHotArrays

data = Cora()
g = data[:]                                # GNNGraph with x, y, train_mask, val_mask, test_mask
y_onehot = onehotbatch(g.y, 1:7)          # 7 classes for Cora

# After model forward pass:
preds = onecold(y_hat, 1:7)
accuracy = mean(preds .== g.y)
```

`MLDatasets` returns ready-to-use `GNNGraph`s with feature matrices and split masks already attached.

## GPU backend portability

GNN training is GPU-portable through `CUDA.jl` (NVIDIA) and `KernelAbstractions.jl` (vendor-neutral). Move both the graph and parameters to the device:

```julia
using CUDA

g_gpu  = g |> gpu
ps_gpu = ps |> gpu
```

`GNNLux` layers dispatch through `KernelAbstractions.jl` for the message-passing kernels, so the same code runs on CUDA, ROCm, or oneAPI backends. CPU is the fallback when no GPU is available — useful for development and CI without changing the model code.

## Extension sketches

These compose `julia-graph-neural-networks` with neighboring skills for advanced workflows. Each is a pointer, not a recipe.

- **Bayesian GNN** — wrap a `GNNLux` model in a `Turing.@model`, sample posterior over node embeddings or weights. For multimodal posteriors (expected with deep GNNs), use Pigeons. See `consensus-mcmc-pigeons`.
- **Graph Neural ODE** — replace stacked GNN layers with a continuous-depth `ODEProblem` whose RHS is a single GNN layer; integrate with `OrdinaryDiffEq` and train through `SciMLSensitivity` adjoints. Useful for irregular time-step temporal graphs.
- **Equivariant primitives** — `EquivariantTensors.jl` provides E(3)-equivariant tensor operations for molecular GNNs. For SchNet, DimeNet, or SE(3)-Transformer architectures (no native Julia implementation), bridge to `torch_geometric` + `e3nn` (PyTorch) via `PythonCall`. The **JAX-native counterpart** is `e3nn-jax` (Mario Geiger): `IrrepsArray`, `tensor_product`, `spherical_harmonics`, `gate`, Wigner-D / Clebsch-Gordan — fully `jit` / `vmap` / `grad` compatible. `e3nn-jax` is currently the only maintained JAX-native equivariant NN primitives library and the right target when the surrounding pipeline is already JAX. MACE / NequIP / Allegro remain PyTorch-first in both ecosystems.
- **Plain message-passing GNNs in JAX** — `jraph` (DeepMind's original JAX GNN library) has been archived. The community successor is the unofficial `JraphX` (Flax NNX). For production plain GCN/GAT/SAGE work, `GNNLux` + `GNNGraphs` on the Julia side is arguably better-maintained today than any JAX option; reserve the JAX stack for equivariant / MLIP work where `e3nn-jax` shines.

## Composition with neighboring skills

- **Julia neural networks** — Lux fundamentals: `setup`, `(ps, st)` plumbing, training loops. See `julia-neural-networks`.
- **Julia AD backends** — Zygote (default), Enzyme (faster on large GNNs). See `julia-ad-backends`.
- **Julia GPU kernels** — custom CUDA / KernelAbstractions kernels for unsupported message-passing patterns. See `julia-gpu-kernels`.
- **Bayesian UDE workflow** — pattern for Bayesian neural-physics models, transferable to Bayesian GNNs. See `bayesian-ude-workflow`.

## Checklist

- [ ] Picked the layer (`GNNGraphs` data, `GNNLux` high-level, `GNNlib` custom kernels) that matches the task
- [ ] Used `GNNGraph` for graph construction with node/edge features
- [ ] Started with `GCNConv` before trying more complex layers
- [ ] Used `GNNChain` for sequential GNN architectures and `GlobalPool` for graph-level tasks
- [ ] Used `batch`/`unbatch` for mini-batch training
- [ ] Loaded benchmark datasets via `MLDatasets`; encoded labels with `OneHotArrays`
- [ ] Added dropout and skip connections for deeper GNNs
- [ ] Profiled with `@benchmark` to identify message-passing bottlenecks
- [ ] Verified GPU code path on a small graph before scaling up
