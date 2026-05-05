using GraphNeuralNetworks
using GNNLux
using Lux
using MLDatasets
using OneHotArrays
using Optimisers
using Zygote
using Random

# Load Cora node-classification benchmark
data = Cora()
g = data[:]                                  # GNNGraph with x, y, train/val/test masks
n_features = size(g.x, 1)
n_classes = 7
y_onehot = onehotbatch(g.y, 1:n_classes)

# Build a 2-layer GCN with GNNLux explicit-parameter style
model = GNNChain(
    GCNConv(n_features => 64, relu),
    Dropout(0.5),
    GCNConv(64 => n_classes),
)

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)
opt_state = Optimisers.setup(Optimisers.Adam(0.01), ps)

function loss_fn(ps, st, g, x, y, mask)
    logits, st_new = model(g, x, ps, st)
    loss = -mean(sum(y[:, mask] .* logsoftmax(logits[:, mask]); dims = 1))
    return loss, st_new
end

# Training loop
for epoch in 1:200
    (loss, st), grads = Zygote.withgradient(p -> loss_fn(p, st, g, g.x, y_onehot, g.train_mask)[1], ps)
    opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
    if epoch % 20 == 0
        preds = onecold(model(g, g.x, ps, st)[1], 1:n_classes)
        acc = mean(preds[g.test_mask] .== g.y[g.test_mask])
        @info "epoch $epoch loss=$loss test_acc=$acc"
    end
end
