---
name: julia-training-diagnostics
description: Debug and diagnose neural network training in Julia. Covers gradient analysis, loss landscape visualization, learning rate finding, convergence debugging, NaN/Inf detection, and Lux.jl-specific debugging patterns. Use when training is failing, diverging, or underperforming in Julia.
---

# Julia Training Diagnostics

## Expert Agent

For diagnosing and fixing training failures in Julia, delegate to:

- **`julia-ml-hpc`**: Julia ML/HPC specialist for training pipeline debugging.
  - *Location*: `plugins/science-suite/agents/julia-ml-hpc.md`
  - *Capabilities*: Gradient analysis, convergence debugging, performance profiling.

## Quick Reference

| Symptom | Likely Cause | First Action |
|---------|-------------|-------------|
| NaN loss | Exploding gradients, bad LR | Check gradient norms, reduce LR |
| Loss plateau | LR too low, dead neurons | Increase LR, check activations |
| Overfitting | Insufficient regularization | Add Dropout, weight decay, data augmentation |
| Underfitting | Model too small, LR too high | Increase capacity, reduce LR |
| Loss spikes | LR too high, bad batch | Reduce LR, check data for outliers |
| Slow convergence | Poor initialization, wrong optimizer | Use Kaiming init, switch to Adam |
| Mutation error | In-place ops in AD path | Rewrite as functional (no `x[i] = v`) |

## Gradient Monitoring

```julia
using ComponentArrays, Statistics

function monitor_gradients(grads, step)
    grad_flat = ComponentArrays.getdata(ComponentArray(grads))
    grad_norm = sqrt(sum(grad_flat .^ 2))
    grad_max = maximum(abs.(grad_flat))
    grad_min = minimum(abs.(grad_flat))

    @info "Step $step" grad_norm grad_max grad_min

    # Warnings
    if grad_norm > 100.0f0
        @warn "Exploding gradients!" grad_norm
    elseif grad_norm < 1e-7f0
        @warn "Vanishing gradients!" grad_norm
    end

    if any(isnan, grad_flat)
        @error "NaN detected in gradients at step $step"
    end

    return grad_norm
end
```

## Gradient Clipping

```julia
using Optimisers

# Clip by global norm (recommended)
opt = OptimiserChain(
    ClipNorm(1.0f0),
    Adam(1e-3)
)

# Clip by value
opt = OptimiserChain(
    ClipGrad(1.0f0),
    Adam(1e-3)
)
```

## NaN/Inf Detection

```julia
function check_numerical_health(ps, st, grads, loss)
    issues = String[]

    # Check loss
    if isnan(loss) || isinf(loss)
        push!(issues, "Loss is $(loss)")
    end

    # Check parameters
    ps_flat = ComponentArrays.getdata(ComponentArray(ps))
    if any(isnan, ps_flat)
        push!(issues, "NaN in parameters ($(count(isnan, ps_flat)) values)")
    end

    # Check gradients
    grad_flat = ComponentArrays.getdata(ComponentArray(grads))
    if any(isnan, grad_flat)
        push!(issues, "NaN in gradients ($(count(isnan, grad_flat)) values)")
    end
    if any(isinf, grad_flat)
        push!(issues, "Inf in gradients ($(count(isinf, grad_flat)) values)")
    end

    if !isempty(issues)
        @error "Numerical issues detected" issues
        return false
    end
    return true
end
```

## Learning Rate Range Test

```julia
function lr_range_test(model, ps, st, train_loader, gdev;
                       lr_min=1e-7, lr_max=10.0, num_steps=100)
    ps_test = deepcopy(ps)
    lr_mult = (lr_max / lr_min)^(1 / num_steps)
    lr = lr_min
    losses = Float32[]
    lrs = Float32[]
    best_loss = Inf32

    opt_state = Optimisers.setup(SGD(lr), ps_test)

    for (i, (x, y)) in enumerate(train_loader)
        i > num_steps && break
        x, y = x |> gdev, y |> gdev

        (loss, st), grads = Zygote.withgradient(ps_test) do p
            y_pred, st_ = model(x, p, st)
            Lux.Training.compute_loss(MSELoss(), y_pred, y), st_
        end

        push!(losses, loss)
        push!(lrs, lr)

        # Stop if loss explodes
        if loss > 4 * best_loss
            break
        end
        best_loss = min(best_loss, loss)

        # Update with current LR
        Optimisers.adjust!(opt_state, lr)
        opt_state, ps_test = Optimisers.update(opt_state, ps_test, grads[1])
        lr *= lr_mult
    end

    return lrs, losses
end
```

## Zygote Debugging

| Error Message | Cause | Fix |
|--------------|-------|-----|
| `Mutating arrays is not supported` | In-place array modification | Use functional alternatives |
| `Can't differentiate foreigncall` | C/Fortran function call | Write `rrule` or use Enzyme |
| `StackOverflowError` | Recursive differentiation | Simplify computation graph |
| `Need an adjoint for ...` | Missing AD rule | Add `ChainRulesCore.rrule` |

### Mutation Fix Example

```julia
# BAD: Mutating array
function bad_layer(x)
    out = similar(x)
    out[1] = x[1]^2     # Mutation!
    return out
end

# GOOD: Functional style
function good_layer(x)
    return vcat(x[1:1] .^ 2, x[2:end])
end
```

## Enzyme Debugging

| Error Message | Cause | Fix |
|--------------|-------|-----|
| `Enzyme cannot yet handle ...` | Unsupported operation | Simplify or use Zygote |
| `Type unstable` | Dynamic dispatch in hot path | Add type annotations |
| `Activity mismatch` | Wrong annotation | Check `Active`/`Duplicated`/`Const` |
| `Illegal instruction` | LLVM compilation issue | Update Enzyme, simplify code |

## Type Instability Guidance

```julia
using Lux

# Check for type instability in model forward pass
function check_type_stability(model, ps, st, x)
    @code_warntype model(x, ps, st)
end

# Common causes of type instability:
# 1. Mixing Float32 and Float64
# 2. Abstract container types (Vector{Any})
# 3. Global variables without const
# 4. Untyped struct fields

# Fix: Ensure consistent types
x = Float32.(x)         # Not: x (which might be Float64)
ps = Lux.f32(ps)        # Convert all params to Float32
```

## Diagnostic Checklist

- [ ] Monitor gradient norms each step (flag > 100 or < 1e-7)
- [ ] Apply gradient clipping with `OptimiserChain(ClipNorm(1.0), optimizer)`
- [ ] Run NaN/Inf checks on loss, parameters, and gradients
- [ ] Use LR range test before training to find optimal learning rate
- [ ] Check for Zygote mutation errors and rewrite as functional code
- [ ] Verify type stability with `@code_warntype` on model forward pass
- [ ] Ensure all tensors use consistent `Float32` dtype
- [ ] Log training and validation loss curves for convergence analysis
