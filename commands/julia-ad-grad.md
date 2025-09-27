---
description: Generate Julia code for automatic differentiation using Zygote.jl with JAX-like grad functionality
category: julia-scientific
argument-hint: "[--higher-order] [--vectorize] [--performance]"
allowed-tools: "*"
---

# /julia-ad-grad

Generate Julia code for automatic differentiation using Zygote.jl (JAX-like grad). Vectorize with Broadcasting or Loops.jl for performance.

## Description

Implements automatic differentiation in Julia using Zygote.jl to mirror JAX's gradient functionality. Includes vectorization strategies, higher-order derivatives, and performance optimization techniques for scientific computing workflows.

## Usage

```
/julia-ad-grad [--higher-order] [--vectorize] [--performance]
```

## What it does

1. Set up Zygote.jl for automatic differentiation (JAX-like grad)
2. Implement gradient computation patterns similar to JAX
3. Vectorize operations using Broadcasting and Loops.jl
4. Handle higher-order derivatives and value_and_grad equivalents
5. Optimize performance for scientific computing

## Example output

```julia
using Zygote
using LinearAlgebra
using Statistics
using LoopVectorization  # For high-performance loops
using Tullio            # For tensor operations
using ChainRulesCore    # For custom AD rules

# Basic gradient computation (JAX-like)
function basic_loss(x::Vector{Float64})
    return sum(x .^ 2) + 2 * sum(x)
end

# Compute gradient (equivalent to jax.grad)
∇basic_loss = gradient(basic_loss, [1.0, 2.0, 3.0])
println("Gradient: ", ∇basic_loss[1])

# Value and gradient computation (equivalent to jax.value_and_grad)
function value_and_gradient(f, x)
    val, back = Zygote.pullback(f, x)
    grad = back(1.0)[1]
    return val, grad
end

# Example usage
val, grad = value_and_gradient(basic_loss, [1.0, 2.0, 3.0])
println("Value: $val, Gradient: $grad")

# Machine learning loss function with gradient
function mse_loss(params::NamedTuple, X::Matrix{Float64}, y::Vector{Float64})
    # Simple linear model: y_pred = X * w + b
    w, b = params.w, params.b
    y_pred = X * w .+ b
    return mean((y_pred .- y) .^ 2)
end

# Parameters
params = (w = randn(3), b = 0.0)
X = randn(100, 3)
y = randn(100)

# Compute gradient w.r.t. parameters
∇params = gradient(p -> mse_loss(p, X, y), params)[1]
println("Parameter gradients: ", ∇params)

# Higher-order derivatives (like jax.grad(jax.grad(...)))
function second_derivative_example(x::Float64)
    return x^4 + 2*x^3 + x^2
end

# First derivative
first_deriv = gradient(second_derivative_example, 2.0)[1]
println("First derivative at x=2: $first_deriv")

# Second derivative
second_deriv = gradient(x -> gradient(second_derivative_example, x)[1], 2.0)[1]
println("Second derivative at x=2: $second_deriv")

# Alternatively, use Zygote's hessian for second derivatives
using ForwardDiff  # For efficient higher-order derivatives
hess = ForwardDiff.derivative(x -> gradient(second_derivative_example, x)[1], 2.0)
println("Hessian (second derivative): $hess")

# Vectorized gradient computation (batch processing)
function vectorized_gradients(f, batch_x::Vector{Vector{Float64}})
    """Compute gradients for a batch of inputs (similar to jax.vmap(jax.grad(f)))"""
    return [gradient(f, x)[1] for x in batch_x]
end

# Example with batch processing
batch_inputs = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
batch_grads = vectorized_gradients(basic_loss, batch_inputs)
println("Batch gradients: ", batch_grads)

# High-performance vectorized gradients using broadcasting
function broadcast_gradients(f, X::Matrix{Float64})
    """Efficient batch gradient computation using broadcasting"""
    # Note: This requires f to be broadcasting-compatible
    return gradient(X -> sum(f.(eachcol(X))), X)[1]
end

# Neural network example with Zygote
struct SimpleNN
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
end

function (nn::SimpleNN)(x::Vector{Float64})
    h = tanh.(nn.W1 * x .+ nn.b1)
    return nn.W2 * h .+ nn.b2
end

function nn_loss(nn::SimpleNN, X::Matrix{Float64}, y::Vector{Float64})
    predictions = [nn(x) for x in eachcol(X)]
    return mean((vcat(predictions...) .- y) .^ 2)
end

# Create network and compute gradients
nn = SimpleNN(randn(10, 3), randn(10), randn(1, 10), randn(1))
X_train = randn(3, 100)
y_train = randn(100)

# Compute gradients w.r.t. network parameters
∇nn = gradient(nn -> nn_loss(nn, X_train, y_train), nn)[1]
println("Network gradients computed successfully")

# Performance optimization with LoopVectorization
function fast_gradient_computation!(grad::Vector{Float64}, x::Vector{Float64})
    """High-performance gradient computation using @turbo"""
    @turbo for i in eachindex(x)
        grad[i] = 2 * x[i] + 2  # Gradient of x^2 + 2x
    end
end

# Tensor operations with Tullio (Einstein notation)
function tensor_gradient_example(A::Matrix{Float64}, B::Matrix{Float64})
    # Einstein summation with automatic differentiation
    @tullio C[i,j] := A[i,k] * B[k,j]
    return sum(C)
end

# Compute gradient w.r.t. tensor operations
A = randn(5, 3)
B = randn(3, 4)
∇A, ∇B = gradient(tensor_gradient_example, A, B)
println("Tensor gradients: shapes $(size(∇A)), $(size(∇B))")

# Custom AD rules for performance (similar to JAX custom gradients)
function custom_operation(x::Vector{Float64})
    return sum(exp.(x))
end

# Define custom backward pass
function ChainRulesCore.rrule(::typeof(custom_operation), x::Vector{Float64})
    y = custom_operation(x)
    function custom_operation_pullback(ȳ)
        return NoTangent(), ȳ .* exp.(x)
    end
    return y, custom_operation_pullback
end

# Test custom rule
custom_grad = gradient(custom_operation, [1.0, 2.0, 3.0])[1]
println("Custom gradient: $custom_grad")

# Optimization workflow (similar to JAX training loop)
mutable struct OptimizationState
    params::NamedTuple
    optimizer_state::Any
end

function sgd_update!(state::OptimizationState, loss_fn, learning_rate::Float64)
    """SGD update step with gradient computation"""
    loss_val, grad = value_and_gradient(loss_fn, state.params)

    # Update parameters
    new_params = map(state.params, grad) do param, g
        param - learning_rate * g
    end

    state.params = new_params
    return loss_val
end

# Training loop example
function train_model(initial_params, loss_fn, num_steps::Int=1000, lr::Float64=0.01)
    state = OptimizationState(initial_params, nothing)

    for step in 1:num_steps
        loss = sgd_update!(state, loss_fn, lr)

        if step % 100 == 0
            println("Step $step, Loss: $loss")
        end
    end

    return state.params
end

# Advanced: Jacobian and Hessian computation
function compute_jacobian(f, x::Vector{Float64})
    """Compute Jacobian matrix (similar to jax.jacobian)"""
    return Zygote.jacobian(f, x)[1]
end

function compute_hessian(f, x::Vector{Float64})
    """Compute Hessian matrix for scalar-valued functions"""
    return Zygote.hessian(f, x)
end

# Multi-dimensional gradient example
function multidim_function(x::Vector{Float64})
    return [sum(x .^ 2), sum(x .^ 3), prod(x)]
end

jac = compute_jacobian(multidim_function, [1.0, 2.0, 3.0])
println("Jacobian shape: $(size(jac))")

# Gradient checking utilities
function gradient_check(f, x::Vector{Float64}, h::Float64=1e-5)
    """Numerical gradient checking for validation"""
    analytical_grad = gradient(f, x)[1]
    numerical_grad = similar(x)

    for i in eachindex(x)
        x_plus = copy(x)
        x_minus = copy(x)
        x_plus[i] += h
        x_minus[i] -= h

        numerical_grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    end

    error = norm(analytical_grad - numerical_grad)
    println("Gradient check error: $error")

    return error < 1e-6
end

# Test gradient checking
test_func(x) = sum(x .^ 3) + sum(x .^ 2)
is_correct = gradient_check(test_func, [1.0, 2.0, 3.0])
println("Gradient check passed: $is_correct")

# Integration with Julia ML ecosystem
using Flux  # For neural networks

function flux_integration_example()
    """Example integrating Zygote with Flux.jl"""
    # Define a simple neural network
    model = Chain(
        Dense(10, 32, relu),
        Dense(32, 32, relu),
        Dense(32, 1)
    )

    # Dummy data
    X = randn(Float32, 10, 100)
    y = randn(Float32, 1, 100)

    # Loss function
    loss(m, x, y) = Flux.mse(m(x), y)

    # Compute gradients (Flux uses Zygote internally)
    gradients = gradient(m -> loss(m, X, y), model)

    println("Flux gradients computed successfully")
    return gradients
end

# Performance comparison with JAX equivalents
function performance_comparison()
    """Compare performance with JAX-equivalent operations"""
    println("\nPerformance Notes:")
    println("1. Zygote.jl provides reverse-mode AD similar to JAX")
    println("2. Use @turbo for high-performance loops")
    println("3. Broadcasting (.) is highly optimized in Julia")
    println("4. ChainRulesCore for custom gradients")
    println("5. ForwardDiff.jl for forward-mode AD (good for low-dimensional inputs)")

    # Timing example
    large_x = randn(10000)
    println("\nTiming large gradient computation...")

    @time begin
        large_grad = gradient(x -> sum(x .^ 2), large_x)[1]
    end

    println("Gradient computation completed")
end

# Main demonstration
function demonstrate_julia_ad()
    println("=== Julia Automatic Differentiation with Zygote.jl ===")

    # Basic examples
    println("\n1. Basic gradient computation:")
    val, grad = value_and_gradient(x -> sum(x .^ 2), [1.0, 2.0])
    println("f([1,2]) = $val, ∇f = $grad")

    # Batch processing
    println("\n2. Batch gradient computation:")
    batch_x = [[1.0, 2.0], [3.0, 4.0]]
    batch_grads = vectorized_gradients(x -> sum(x .^ 2), batch_x)
    println("Batch gradients: $batch_grads")

    # Higher-order derivatives
    println("\n3. Higher-order derivatives:")
    second_deriv = gradient(x -> gradient(t -> t^4, x)[1], 2.0)[1]
    println("Second derivative of x^4 at x=2: $second_deriv")

    # Performance comparison
    performance_comparison()

    return "Julia AD demonstration completed!"
end

# Run demonstration
demonstrate_julia_ad()

# JAX vs Julia AD comparison table
println("""
=== JAX vs Julia AD Comparison ===

JAX                          | Julia (Zygote.jl)
---------------------------|---------------------------
jax.grad(f)                 | gradient(f, x)[1]
jax.value_and_grad(f)       | Zygote.pullback(f, x)
jax.jacobian(f)             | Zygote.jacobian(f, x)[1]
jax.hessian(f)              | Zygote.hessian(f, x)
jax.vmap(jax.grad(f))       | [gradient(f, x)[1] for x in batch]
@jax.jit                    | @code_warntype, precompile
jax.custom_gradient         | ChainRulesCore.rrule

Performance Tips:
- Use broadcasting (.) for vectorization
- @turbo for high-performance loops
- ForwardDiff for low-dimensional forward-mode AD
- ChainRulesCore for custom AD rules
- Flux.jl integrates seamlessly with Zygote
""")
```

## Related Commands

- `/julia-jit-like` - Optimize Julia code for performance
- `/julia-prob-model` - Probabilistic modeling with AD
- `/jax-grad` - Compare with JAX gradient computation
- `/python-debug-prof` - Cross-language performance analysis