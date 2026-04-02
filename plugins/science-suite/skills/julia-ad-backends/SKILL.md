---
name: julia-ad-backends
description: Select and debug automatic differentiation backends in Julia. Covers Zygote.jl (source-to-source reverse-mode), Enzyme.jl (LLVM-level forward/reverse), ForwardDiff.jl (forward-mode dual numbers), and AbstractDifferentiation.jl for backend-agnostic code. Includes custom adjoint rules (ChainRulesCore.jl) and debugging strategies. Use when choosing AD backends or debugging gradient issues in Julia.
---

# Julia AD Backends

## Expert Agent

For AD backend selection, custom rules, and gradient debugging, delegate to:

- **`julia-ml-hpc`**: Julia ML/HPC specialist for AD pipelines and performance.
  - *Location*: `plugins/science-suite/agents/julia-ml-hpc.md`
  - *Capabilities*: Backend selection, custom adjoints, mixed-mode AD, gradient correctness.

## Backend Selection Decision Tree

```
Need gradients?
├── Yes: How many parameters?
│   ├── Few (<10) ──────────── ForwardDiff.jl (forward-mode)
│   ├── Many (>10) ─────────── Reverse-mode:
│   │   ├── Pure Julia code? ── Zygote.jl
│   │   ├── Need mutation? ──── Enzyme.jl
│   │   └── FFI / C calls? ──── Enzyme.jl
│   └── Need Hessian? ──────── Mixed: ForwardDiff-over-Zygote
└── No: Just evaluation ────── No AD needed
```

## Zygote.jl (Source-to-Source Reverse-Mode)

```julia
using Zygote

# Basic gradient
f(x) = sum(x .^ 2)
grad = Zygote.gradient(f, [1.0, 2.0, 3.0])  # ([2.0, 4.0, 6.0],)

# With auxiliary outputs
(val, grad) = Zygote.withgradient(f, [1.0, 2.0, 3.0])

# Pullback (manual VJP)
y, pb = Zygote.pullback(f, [1.0, 2.0, 3.0])
grad = pb(1.0)  # ([2.0, 4.0, 6.0],)
```

### Zygote Limitations

| Limitation | Description | Workaround |
|-----------|-------------|------------|
| No mutation | Cannot differentiate `x[i] = v` | Use `Zygote.Buffer` or functional style |
| No `try/catch` | Exception handling unsupported | Remove from hot path |
| No `Dict` mutation | Hash map writes fail | Use `NamedTuple` or struct |
| Foreign calls | C/Fortran opaque to Zygote | Write `rrule` or use Enzyme |
| Dynamic control flow | Limited support | Use `ifelse()` instead of `if` |

## Enzyme.jl (LLVM-Level Forward/Reverse)

```julia
using Enzyme

# Reverse mode
function f!(y, x)
    y[1] = x[1]^2 + x[2]^2
    return nothing
end

dy = [1.0]
dx = [0.0, 0.0]
Enzyme.autodiff(Reverse, f!, Duplicated(dy, dy), Duplicated([3.0, 4.0], dx))
# dx = [6.0, 8.0]
```

### Enzyme Activity Annotations

| Annotation | Meaning | Use When |
|-----------|---------|----------|
| `Active` | Scalar return, want derivative | Scalar-valued functions |
| `Duplicated(x, dx)` | Primal + shadow (mutable) | Arrays, in-place mutation |
| `Const(x)` | No derivative needed | Fixed parameters, constants |
| `DuplicatedNoNeed(x, dx)` | Shadow only, skip primal | Save memory when primal unneeded |

## ForwardDiff.jl (Forward-Mode Dual Numbers)

```julia
using ForwardDiff

f(x) = sin(x[1]) * cos(x[2])

# Gradient (forward-mode, best for few parameters)
g = ForwardDiff.gradient(f, [1.0, 2.0])

# Jacobian
J = ForwardDiff.jacobian(x -> [x[1]^2, x[1]*x[2]], [3.0, 4.0])

# Hessian
H = ForwardDiff.hessian(f, [1.0, 2.0])

# Chunk size for performance (controls SIMD width)
cfg = ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{4}())
ForwardDiff.gradient(f, x, cfg)
```

## Custom Rules with ChainRulesCore.jl

```julia
using ChainRulesCore

# Reverse-mode rule (rrule)
function ChainRulesCore.rrule(::typeof(my_func), x)
    y = my_func(x)
    function my_func_pullback(dy)
        dx = dy * custom_derivative(x)
        return NoTangent(), dx  # NoTangent for function itself
    end
    return y, my_func_pullback
end

# Forward-mode rule (frule)
function ChainRulesCore.frule((_, dx), ::typeof(my_func), x)
    y = my_func(x)
    dy = custom_derivative(x) * dx
    return y, dy
end

# Mark non-differentiable functions
@non_differentiable println(::Any)
@non_differentiable rand(::Any)
```

## DifferentiationInterface.jl

```julia
using DifferentiationInterface
import Zygote, ForwardDiff

# Backend-agnostic gradient
backend = AutoZygote()
g = DifferentiationInterface.gradient(f, backend, x)

# Switch backend without changing code
backend = AutoForwardDiff()
g = DifferentiationInterface.gradient(f, backend, x)

# Available backends
# AutoZygote(), AutoEnzyme(), AutoForwardDiff(),
# AutoReverseDiff(), AutoTapir()
```

## Mixed-Mode AD for Hessians

```julia
using ForwardDiff, Zygote

# Hessian via ForwardDiff-over-Zygote (efficient for many parameters)
function hessian_fwd_rev(f, x)
    return ForwardDiff.jacobian(x -> Zygote.gradient(f, x)[1], x)
end

H = hessian_fwd_rev(f, [1.0, 2.0, 3.0])
```

## Gradient Correctness Testing

```julia
using FiniteDifferences

# Test against finite differences
f(x) = sum(x .^ 3)
x = randn(5)

ad_grad = Zygote.gradient(f, x)[1]
fd_grad = FiniteDifferences.grad(central_fdm(5, 1), f, x)[1]

@assert maximum(abs.(ad_grad .- fd_grad)) < 1e-6 "Gradient mismatch!"
```

## Performance Comparison

| Backend | Mode | Mutation | GPU | Compile Time | Runtime | Best For |
|---------|------|----------|-----|-------------|---------|----------|
| Zygote | Reverse | No | Yes | Medium | Fast | ML, neural nets |
| Enzyme | Both | Yes | Yes | Slow | Fastest | HPC, mutating code |
| ForwardDiff | Forward | Yes | Limited | Fast | Fast (few params) | Small param count |
| ReverseDiff | Reverse | Yes | No | Fast | Medium | CPU, mutation needed |
| Tapir | Reverse | Yes | No | Medium | Medium | Experimental |

## Checklist

- [ ] Choose AD backend based on parameter count and mutation needs
- [ ] Use Zygote for standard ML (Lux.jl default)
- [ ] Use Enzyme for mutating code or C/Fortran interop
- [ ] Use ForwardDiff for small parameter counts or Hessians
- [ ] Write `rrule`/`frule` for custom or non-differentiable operations
- [ ] Validate gradients against `FiniteDifferences.jl` during development
