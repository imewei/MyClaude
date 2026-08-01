# Lux Training Loop, Custom Message Passing, and Molecular GNN — Full Reference

`--mode deep` content for **julia-graph-neural-networks**.

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
