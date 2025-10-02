---
description: Generate Julia code for automatic differentiation using Zygote.jl with JAX-like grad functionality and intelligent 23-agent optimization
category: julia-scientific
argument-hint: "[--higher-order] [--vectorize] [--performance] [--agents=auto|julia|scientific|ai|optimization|all] [--orchestrate] [--intelligent] [--breakthrough] [--optimize] [--distributed]"
allowed-tools: "*"
model: inherit
---

# Julia Automatic Differentiation with Zygote.jl

Generate Julia code for automatic differentiation using Zygote.jl with JAX-like gradient functionality. Includes vectorization, higher-order derivatives, and performance optimization.

## Usage

```bash
/julia-ad-grad [--higher-order] [--vectorize] [--performance] [--agents=auto|julia|scientific|ai|optimization|all] [--orchestrate] [--intelligent] [--breakthrough] [--optimize] [--distributed]
```

Generates Julia automatic differentiation code with intelligent 23-agent optimization for Zygote.jl and Julia AD ecosystem.

## Arguments

- `--higher-order`: Include higher-order derivatives and Hessian computation
- `--vectorize`: Add vectorization with Broadcasting and LoopVectorization.jl
- `--performance`: Include performance optimization techniques
- `--agents=<agents>`: Agent selection (auto, julia, scientific, ai, optimization, all)
- `--orchestrate`: Enable advanced 23-agent orchestration with AD intelligence
- `--intelligent`: Enable intelligent agent selection based on Julia AD analysis
- `--breakthrough`: Enable breakthrough Julia AD optimization
- `--optimize`: Apply performance optimization to AD computations
- `--distributed`: Enable distributed Julia AD computation with agent coordination

## What it does

1. Set up Zygote.jl for automatic differentiation (JAX-like grad)
2. Implement gradient computation patterns similar to JAX
3. Vectorize operations using Broadcasting and LoopVectorization.jl
4. Handle higher-order derivatives and value_and_grad equivalents
5. Optimize performance for scientific computing applications
6. **23-Agent Julia AD Intelligence**: Multi-agent collaboration for optimal automatic differentiation
7. **Advanced Julia AD Optimization**: Agent-driven Julia AD performance and accuracy optimization
8. **Intelligent Julia Integration**: Agent-coordinated Julia ecosystem integration and optimization

## 23-Agent Intelligent Julia AD System

### Intelligent Agent Selection (`--intelligent`)
**Auto-Selection Algorithm**: Analyzes Julia AD requirements, computation patterns, and performance goals to automatically choose optimal agent combinations from the 23-agent library.

```bash
# Julia AD Pattern Detection → Agent Selection
- Research Computing → research-intelligence-master + scientific-computing-master + julia-pro
- Production ML → ai-systems-architect + julia-pro + neural-networks-master
- Scientific Simulation → scientific-computing-master + julia-pro + optimization-master
- Educational Projects → documentation-architect + julia-pro + neural-networks-master
- High-Performance Computing → optimization-master + julia-pro + systems-architect
```

### Core Julia AD Agents

#### **`julia-pro`** - Julia Ecosystem AD Expert
- **Julia AD Optimization**: Deep expertise in Zygote.jl, ForwardDiff.jl, and Julia AD ecosystem
- **Performance Engineering**: Julia AD performance optimization and memory management
- **Type System Integration**: Julia type system optimization for automatic differentiation
- **Package Integration**: Julia AD package ecosystem coordination and optimization
- **Compilation Optimization**: Julia compilation optimization for AD performance

#### **`scientific-computing-master`** - Scientific AD Applications
- **Mathematical Modeling**: Scientific computing applications of automatic differentiation
- **Numerical Analysis**: Advanced numerical methods with automatic differentiation
- **Research Applications**: Academic and research-grade AD implementations
- **Algorithm Design**: Scientific algorithm development with AD integration
- **Domain Integration**: Cross-domain scientific computing with Julia AD

#### **`optimization-master`** - AD Optimization & Performance
- **AD Performance**: Advanced automatic differentiation performance optimization
- **Memory Management**: Memory-efficient gradient computation strategies
- **Computational Efficiency**: High-performance Julia AD algorithm development
- **Scaling Strategies**: Large-scale automatic differentiation optimization
- **Numerical Stability**: Numerical stability analysis and optimization for AD

### Advanced Agent Selection Strategies

#### **`auto`** - Intelligent Agent Selection for Julia AD
Automatically analyzes Julia AD requirements and selects optimal agent combinations:
- **Pattern Analysis**: Detects Julia AD usage patterns and optimization opportunities
- **Performance Assessment**: Evaluates computational requirements and constraints
- **Agent Matching**: Maps Julia AD needs to relevant agent expertise
- **Optimization Balance**: Balances AD accuracy with computational efficiency

#### **`julia`** - Julia-Specialized AD Team
- `julia-pro` (Julia AD ecosystem lead)
- `optimization-master` (performance optimization)
- `scientific-computing-master` (scientific applications)
- `neural-networks-master` (ML integration)

#### **`scientific`** - Scientific Computing AD Team
- `scientific-computing-master` (lead)
- `julia-pro` (Julia implementation)
- `optimization-master` (performance optimization)
- `research-intelligence-master` (research methodology)

#### **`ai`** - AI/ML Julia AD Team
- `neural-networks-master` (lead)
- `julia-pro` (Julia optimization)
- `ai-systems-architect` (production systems)
- `optimization-master` (performance)

#### **`optimization`** - Performance-Focused AD Team
- `optimization-master` (lead)
- `julia-pro` (Julia performance)
- `scientific-computing-master` (numerical methods)
- `systems-architect` (system optimization)

#### **`all`** - Complete 23-Agent Julia AD Ecosystem
Activates all relevant agents with intelligent orchestration for breakthrough Julia AD development.

### Advanced 23-Agent Julia AD Examples

```bash
# Intelligent auto-selection for Julia AD
/julia-ad-grad --agents=auto --intelligent --optimize

# Scientific computing AD with specialized agents
/julia-ad-grad --agents=scientific --breakthrough --orchestrate --higher-order

# High-performance Julia AD optimization
/julia-ad-grad --agents=optimization --optimize --performance --distributed

# Research-grade AD development
/julia-ad-grad --agents=all --breakthrough --orchestrate --vectorize

# AI/ML Julia AD integration
/julia-ad-grad --agents=ai --optimize --intelligent --higher-order

# Complete 23-agent Julia AD ecosystem
/julia-ad-grad --agents=all --orchestrate --breakthrough --intelligent --distributed
```

## Julia AD Setup

```julia
using Zygote
using LinearAlgebra
using Statistics

# Basic gradient computation (JAX-like)
grad_fn = gradient(loss_function, x)

# Value and gradient (like JAX's value_and_grad)
function value_and_grad(f, x)
    return f(x), gradient(f, x)[1]
end

# Example usage
function loss(x)
    return sum(x.^2)
end

# Compute gradient
∇loss = gradient(loss, [1.0, 2.0, 3.0])[1]
# Result: [2.0, 4.0, 6.0]

# Value and gradient together
val, grad = value_and_grad(loss, [1.0, 2.0, 3.0])
# val = 14.0, grad = [2.0, 4.0, 6.0]
```

## Higher-Order Derivatives

```julia
using ForwardDiff

# Second derivatives (Hessian)
function hessian_zygote(f, x)
    return jacobian(x -> gradient(f, x)[1], x)
end

# Example: f(x) = x₁² + x₁x₂ + x₂²
f(x) = x[1]^2 + x[1]*x[2] + x[2]^2

x = [1.0, 2.0]
H = hessian_zygote(f, x)
# H = [2.0 1.0; 1.0 2.0]

# Higher-order using ForwardDiff for efficiency
H_forward = ForwardDiff.hessian(f, x)

# Mixed approach: Zygote for first derivative, ForwardDiff for second
function mixed_hessian(f, x)
    return ForwardDiff.jacobian(x -> Zygote.gradient(f, x)[1], x)
end
```

## Vectorization and Performance

```julia
using LoopVectorization
using Tullio

# Vectorized gradient computation
function vectorized_grad_example(W, X, y)
    # Batch gradient computation
    function batch_loss(W)
        predictions = X * W
        return sum((predictions .- y).^2) / length(y)
    end

    # Efficient gradient
    return gradient(batch_loss, W)[1]
end

# Using Tullio for tensor operations
@tullio grad[i] := 2 * (X[j,i] * (X[j,k] * W[k] - y[j])) / length(y)

# LoopVectorization for performance-critical inner loops
function fast_gradient_computation!(grad, X, W, y)
    @turbo for i in axes(grad, 1)
        grad[i] = 0.0
        for j in axes(X, 1)
            pred = 0.0
            for k in axes(X, 2)
                pred += X[j,k] * W[k]
            end
            error = pred - y[j]
            grad[i] += 2 * X[j,i] * error / length(y)
        end
    end
end
```

## JAX-like Functional Patterns

```julia
# JAX-style grad function
function grad(f)
    return x -> gradient(f, x)[1]
end

# JAX-style jacobian
function jacfwd(f)
    return x -> jacobian(f, x)
end

function jacrev(f)
    return x -> jacobian(f, x)  # Zygote uses reverse-mode by default
end

# Compose transformations
f(x) = sum(x.^3)
grad_f = grad(f)
hess_f = grad(grad_f)

x = [1.0, 2.0]
∇f = grad_f(x)      # [3.0, 12.0]
∇²f = hess_f(x)     # [6.0, 12.0] (diagonal of Hessian for this function)
```

## Custom Gradient Rules

```julia
using ChainRulesCore

# Define custom gradient rule for performance
function my_special_function(x)
    return sum(sin.(x) .+ cos.(x))
end

# Custom rule (if needed for performance)
function ChainRulesCore.rrule(::typeof(my_special_function), x)
    y = my_special_function(x)
    function pullback(ȳ)
        return NoTangent(), ȳ .* (cos.(x) .- sin.(x))
    end
    return y, pullback
end

# Verify custom rule
x = randn(100)
manual_grad = cos.(x) .- sin.(x)
zygote_grad = gradient(my_special_function, x)[1]
# Should be approximately equal
```

## Neural Network Example

```julia
using Flux  # For neural network utilities

# Simple neural network with custom training
struct SimpleNN
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
end

function (nn::SimpleNN)(x)
    h = tanh.(nn.W1 * x .+ nn.b1)
    return nn.W2 * h .+ nn.b2
end

# Initialize network
function init_network(input_dim, hidden_dim, output_dim)
    return SimpleNN(
        randn(hidden_dim, input_dim) * 0.1,
        zeros(hidden_dim),
        randn(output_dim, hidden_dim) * 0.1,
        zeros(output_dim)
    )
end

# Loss function
function mse_loss(nn, X, y)
    predictions = [nn(X[:, i]) for i in 1:size(X, 2)]
    pred_matrix = hcat(predictions...)
    return sum((pred_matrix .- y).^2) / length(y)
end

# Training step with gradients
function train_step!(nn, X, y, lr=0.01)
    # Get gradients
    grads = gradient(nn -> mse_loss(nn, X, y), nn)[1]

    # Update parameters
    nn.W1 .-= lr .* grads.W1
    nn.b1 .-= lr .* grads.b1
    nn.W2 .-= lr .* grads.W2
    nn.b2 .-= lr .* grads.b2

    return mse_loss(nn, X, y)
end

# Example usage
nn = init_network(2, 10, 1)
X = randn(2, 100)
y = sum(X.^2, dims=1)  # Target function

# Training loop
for epoch in 1:1000
    loss = train_step!(nn, X, y, 0.01)
    if epoch % 100 == 0
        println("Epoch $epoch, Loss: $loss")
    end
end
```

## Performance Optimization

```julia
using BenchmarkTools

# Gradient computation with different approaches
function benchmark_gradients()
    f(x) = sum(x.^4 .- 2*x.^2 .+ x)
    x = randn(1000)

    # Zygote (reverse-mode)
    @btime gradient($f, $x)

    # ForwardDiff (forward-mode, good for high-dimensional output)
    @btime ForwardDiff.gradient($f, $x)

    # Manual gradient (if simple enough)
    manual_grad(x) = 4*x.^3 .- 4*x .+ 1
    @btime manual_grad($x)
end

# Memory-efficient gradient accumulation
function efficient_batch_gradient(loss_fn, params, data_batches)
    total_grad = gradient(loss_fn, params, data_batches[1])[1]

    for batch in data_batches[2:end]
        batch_grad = gradient(loss_fn, params, batch)[1]
        # In-place addition to save memory
        total_grad .= total_grad .+ batch_grad
    end

    # Average over batches
    total_grad .= total_grad ./ length(data_batches)
    return total_grad
end
```

## Debugging and Validation

```julia
# Check gradient correctness with finite differences
using FiniteDifferences

function check_gradient(f, x; rtol=1e-6)
    analytical = gradient(f, x)[1]
    numerical = FiniteDifferences.grad(central_fdm(5, 1), f, x)[1]

    relative_error = abs.(analytical .- numerical) ./ (abs.(numerical) .+ 1e-10)
    max_error = maximum(relative_error)

    if max_error < rtol
        println("✓ Gradient check passed (max relative error: $max_error)")
        return true
    else
        println("✗ Gradient check failed (max relative error: $max_error)")
        return false
    end
end

# Example usage
f(x) = sum(x.^3 .+ sin.(x))
x = randn(10)
check_gradient(f, x)
```

## Agent-Enhanced Julia AD Integration Patterns

### Complete Julia AD Development Workflow
```bash
# Intelligent Julia AD development pipeline
/julia-ad-grad --agents=auto --intelligent --optimize --higher-order
/julia-jit-like --agents=auto --intelligent --performance
/julia-prob-model --agents=scientific --breakthrough --orchestrate
```

### Scientific Computing Julia AD Pipeline
```bash
# High-performance scientific Julia AD workflow
/julia-ad-grad --agents=scientific --breakthrough --orchestrate --vectorize
/jax-essentials --agents=julia --intelligent --operation=grad
/optimize --agents=julia --category=algorithm --implement
```

### Production ML Julia AD Infrastructure
```bash
# Large-scale production Julia AD optimization
/julia-ad-grad --agents=ai --optimize --distributed --higher-order
/generate-tests --agents=julia --type=scientific --coverage=85
/run-all-tests --agents=julia --scientific --performance
```

## Related Commands

**Julia Ecosystem Development**: Enhanced Julia development with agent intelligence
- `/julia-jit-like --agents=auto` - Optimize Julia code for performance with agent intelligence
- `/julia-prob-model --agents=scientific` - Probabilistic modeling with AD and scientific agents
- `/optimize --agents=julia` - Julia optimization with specialized agents

**Cross-Language Scientific Computing**: Multi-language integration
- `/jax-essentials --agents=auto` - Compare with JAX gradient computation
- `/python-debug-prof --agents=auto` - Cross-language performance analysis with agents
- `/jax-performance --agents=julia` - JAX performance comparison with Julia agents

**Quality Assurance**: Julia AD validation and optimization
- `/generate-tests --agents=julia --type=scientific` - Generate Julia AD tests with agent intelligence
- `/run-all-tests --agents=julia --scientific` - Comprehensive Julia testing with specialized agents
- `/check-code-quality --agents=auto --language=julia` - Julia code quality with agent optimization