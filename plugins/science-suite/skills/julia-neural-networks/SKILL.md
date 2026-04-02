---
name: julia-neural-networks
description: Master Lux.jl and Flux.jl for deep learning in Julia. Covers explicit-parameter training loops, optimizers (Optimisers.jl), loss functions, data loading (DataLoaders.jl/MLUtils.jl), Flux-to-Lux migration, and supervised/unsupervised patterns. Use when training neural networks in Julia.
---

# Julia Neural Networks

## Expert Agent

For neural network training in Julia, delegate to:

- **`julia-ml-hpc`**: Julia ML/HPC specialist for Lux.jl, Flux.jl, and GPU training.
  - *Location*: `plugins/science-suite/agents/julia-ml-hpc.md`
  - *Capabilities*: Training loops, optimizer selection, data pipelines, model checkpointing.

## Lux.jl vs Flux.jl

| Feature | Lux.jl | Flux.jl |
|---------|--------|---------|
| Parameter style | Explicit (functional) | Implicit (stateful) |
| State management | Separate `ps`, `st` | Embedded in model |
| AD compatibility | All backends | Zygote only |
| Composability | High (pure functions) | Medium |
| SciML integration | Native | Limited |
| GPU transfer | Explicit `gpu_device()` | `gpu()` on model |
| Recommended for | New projects, SciML | Legacy, quick prototyping |

## Lux.jl Training Loop

```julia
using Lux, Random, Optimisers, Zygote, MLUtils

# Model definition
model = Chain(
    Dense(784, 256, relu),
    Dense(256, 128, relu),
    Dense(128, 10)
)

# Initialize parameters and state
rng = Random.MersenneTwister(42)
ps, st = Lux.setup(rng, model)

# Move to GPU
using LuxCUDA
gdev = gpu_device()
ps = ps |> gdev
st = st |> gdev

# Optimizer setup
opt_state = Optimisers.setup(Adam(1e-3), ps)

# Training step
function train_step(model, ps, st, opt_state, x, y)
    (loss, st_new), grads = Zygote.withgradient(ps) do p
        y_pred, st_ = model(x, p, st)
        Lux.Training.compute_loss(MSELoss(), y_pred, y), st_
    end
    opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
    return ps, st_new, opt_state, loss
end

# DataLoader loop
train_loader = DataLoader((x_train, y_train); batchsize=64, shuffle=true)
for epoch in 1:100
    for (x_batch, y_batch) in train_loader
        x_batch = x_batch |> gdev
        y_batch = y_batch |> gdev
        ps, st, opt_state, loss = train_step(model, ps, st, opt_state, x_batch, y_batch)
    end
end
```

## Inference with Test Mode

```julia
# Switch to test mode (disables dropout, uses running stats for BatchNorm)
st_test = Lux.testmode(st)
y_pred, _ = model(x_test, ps, st_test)
```

## Optimizer Selection

| Optimizer | Learning Rate | Use Case | Notes |
|-----------|--------------|----------|-------|
| `Adam(lr)` | 1e-3 | Default choice | Good for most tasks |
| `AdamW(lr; weight_decay)` | 1e-3 to 1e-4 | Regularized training | Decoupled weight decay |
| `SGD(lr; momentum)` | 1e-2 to 1e-1 | Vision, fine-tuning | Better generalization |
| `Lion(lr)` | 1e-4 | Memory-efficient | Sign-based updates |

### Learning Rate Schedules with ParameterSchedulers.jl

```julia
using ParameterSchedulers

schedule = Sequence(
    SinExp(lr0=1e-3, lr1=1e-5, period=20) => 80,
    Constant(1e-5) => 20
)

# Or cosine annealing
schedule = CosAnneal(lr0=1e-3, lr1=1e-6, period=100)
```

## Data Loading with MLUtils.jl

```julia
using MLUtils

# Splitting
train_data, val_data = splitobs((X, Y); at=0.8, shuffle=true)

# DataLoader with parallel workers
train_loader = DataLoader(train_data; batchsize=32, shuffle=true, parallel=true)

# Lazy transforms
transformed = mapobs(x -> x ./ 255f0, X)
```

## Loss Functions

```julia
using Lux, Statistics

# Built-in losses
loss_ce = CrossEntropyLoss()          # Classification
loss_bce = BinaryCrossEntropyLoss()   # Binary classification
loss_mse = MSELoss()                  # Regression
loss_mae = MAELoss()                  # Robust regression

# Huber loss
loss_huber = HuberLoss(; delta=1.0f0)

# Custom focal loss for imbalanced classification
function focal_loss(y_pred, y_true; gamma=2.0f0, alpha=0.25f0)
    p = softmax(y_pred)
    ce = -y_true .* log.(p .+ 1f-8)
    weight = alpha .* (1 .- p) .^ gamma
    return mean(sum(weight .* ce; dims=1))
end
```

## Checkpointing with JLD2

```julia
using JLD2

# Save checkpoint
function save_checkpoint(path, ps, st, opt_state, epoch)
    jldsave(path; ps=cpu_device()(ps), st=cpu_device()(st),
            opt_state=cpu_device()(opt_state), epoch=epoch)
end

# Load checkpoint
function load_checkpoint(path)
    data = load(path)
    return data["ps"], data["st"], data["opt_state"], data["epoch"]
end
```

## Flux-to-Lux Migration

| Flux Pattern | Lux Equivalent | Notes |
|-------------|----------------|-------|
| `Dense(in, out)` | `Dense(in, out)` | Same API |
| `Chain(layers...)` | `Chain(layers...)` | Same API |
| `model(x)` | `model(x, ps, st)` | Explicit params |
| `params(model)` | `Lux.setup(rng, model)` | Returns `(ps, st)` |
| `Flux.train!(loss, ps, data, opt)` | Manual loop with `Zygote.withgradient` | More control |
| `gpu(model)` | `ps \|> gpu_device()` | Params only |
| `@epochs N train!(...)` | `for epoch in 1:N` loop | Explicit loop |

## Anti-Patterns

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| Mutating parameters in-place | Breaks AD | Use functional updates |
| Forgetting `st` return | Stale BatchNorm stats | Always capture updated state |
| `Float64` on GPU | 2x slower, 2x memory | Use `Float32` / `f32()` |
| Not using `testmode` | Dropout active at inference | Call `Lux.testmode(st)` |
| Loading full dataset to GPU | OOM | Use `DataLoader` with batches |
| Ignoring gradient clipping | Exploding gradients | Use `OptimiserChain(ClipNorm(1.0), Adam())` |

## Checklist

- [ ] Choose Lux.jl for new projects (explicit parameters, multi-AD support)
- [ ] Initialize with explicit RNG seed for reproducibility
- [ ] Use `Float32` for GPU training (`f32()` or `Float32.()`)
- [ ] Set up `DataLoader` with appropriate batch size and shuffling
- [ ] Select optimizer based on task (Adam default, AdamW for regularization)
- [ ] Implement learning rate schedule with ParameterSchedulers.jl
- [ ] Use `Lux.testmode(st)` for inference
- [ ] Save checkpoints with JLD2 (move to CPU before saving)
- [ ] Monitor training loss and validation metrics each epoch
