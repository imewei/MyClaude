---
name: julia-graph-neural-networks
description: Build graph neural networks in Julia with GraphNeuralNetworks.jl and Lux.jl. Covers GCN, GAT, GraphSAGE, message passing neural networks, node/edge/graph-level tasks, heterogeneous graphs, temporal graphs (TemporalGNNs), and mini-batch training on large graphs. Use when working with graph-structured data in Julia.
---

# Julia Graph Neural Networks

## Expert Agent

For graph neural network architecture and training in Julia, delegate to:

- **`julia-ml-hpc`** at `plugins/science-suite/agents/julia-ml-hpc.md`

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

Batch multiple graphs for efficient training:

```julia
# Create a batch of graphs
graphs = [rand_graph(n, 2n; ndata=(; x=randn(Float32, 16, n))) for n in [10, 20, 15]]
batched_g = batch(graphs)

# Forward pass on batch (nodes are concatenated, edges re-indexed)
y, st = model(batched_g, batched_g.x, ps, st)

# Unbatch results
individual_graphs = unbatch(batched_g)
```

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

## Checklist

- [ ] Use `GNNGraph` for graph construction with node/edge features
- [ ] Start with `GCNConv` before trying more complex layers
- [ ] Use `GNNChain` for sequential GNN architectures
- [ ] Apply `GlobalPool` for graph-level tasks
- [ ] Use `batch`/`unbatch` for mini-batch training
- [ ] Add dropout and skip connections for deeper GNNs
- [ ] Profile with `@benchmark` to identify message-passing bottlenecks
