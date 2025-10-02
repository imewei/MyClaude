---
description: Nonlinear least-squares optimization with NLSQ library - handles both standard and massive datasets with memory management and algorithm selection
category: jax-optimization
argument-hint: "[--dataset-size=small|large|massive] [--algorithm=TRR|LM] [--gpu-accel] [--chunking] [--agents=auto|jax|scientific|ai|optimization|all] [--orchestrate] [--intelligent] [--breakthrough] [--optimize] [--robust]"
allowed-tools: "*"
model: inherit
---

# JAX NLSQ

Nonlinear least-squares optimization with NLSQ library - handles both standard and massive datasets with memory management and algorithm selection.

```bash
/jax-nlsq [--dataset-size=small|large|massive] [--algorithm=TRR|LM] [--gpu-accel] [--chunking] [--agents=auto|jax|scientific|ai|optimization|all] [--orchestrate] [--intelligent] [--breakthrough] [--optimize] [--robust]
```

## Options

- `--dataset-size=<size>`: Dataset size category (small, large, massive)
- `--algorithm=<algo>`: Algorithm choice (TRR, LM)
- `--gpu-accel`: Enable GPU acceleration
- `--chunking`: Enable chunking for large datasets
- `--agents=<agents>`: Agent selection (auto, jax, scientific, ai, optimization, all)
- `--orchestrate`: Enable advanced 23-agent orchestration with optimization intelligence
- `--intelligent`: Enable intelligent agent selection based on optimization analysis
- `--breakthrough`: Enable breakthrough optimization techniques and algorithms
- `--optimize`: Apply performance optimization to optimization workflows
- `--robust`: Advanced robust optimization with research-grade numerical methods

## What it does

1. **Nonlinear Curve Fitting**: Trust Region Reflective and Levenberg-Marquardt algorithms
2. **Memory Management**: Efficient handling of datasets from small to massive scales
3. **Algorithm Selection**: Automatic choice between TRR and LM based on problem characteristics
4. **Performance Optimization**: GPU acceleration and intelligent chunking strategies
5. **Robust Optimization**: Error handling, convergence monitoring, and numerical stability
6. **23-Agent Optimization Intelligence**: Multi-agent collaboration for optimal optimization strategies
7. **Advanced Algorithms**: Agent-driven breakthrough optimization techniques and methodologies
8. **Research-Grade Methods**: Agent-coordinated robust optimization for scientific computing

## 23-Agent Intelligent Optimization System

### Intelligent Agent Selection (`--intelligent`)
**Auto-Selection Algorithm**: Analyzes optimization requirements, problem characteristics, and numerical challenges to automatically choose optimal agent combinations from the 23-agent library.

```bash
# Optimization Problem Type Detection → Agent Selection
- Scientific Parameter Estimation → scientific-computing-master + research-intelligence-master + correlation-function-expert
- Large-Scale Optimization → ai-systems-architect + systems-architect + jax-pro
- Robust Curve Fitting → scientific-computing-master + jax-pro + correlation-function-expert
- High-Performance Optimization → systems-architect + jax-pro + ai-systems-architect
- Research Optimization → research-intelligence-master + scientific-computing-master + multi-agent-orchestrator
```

### Core Optimization Agents

#### **`scientific-computing-master`** - Scientific Optimization Expert
- **Numerical Optimization**: Advanced numerical optimization methods and algorithms
- **Scientific Applications**: Optimization for computational science and engineering problems
- **Algorithm Development**: Cutting-edge optimization algorithms for scientific computing
- **Robust Methods**: Numerically stable optimization for challenging scientific problems
- **Performance Engineering**: High-performance optimization for scientific workflows

#### **`jax-pro`** - JAX Optimization Specialist
- **JAX Optimization**: Deep expertise in JAX-based optimization and automatic differentiation
- **Performance Optimization**: JAX-specific optimization performance and memory efficiency
- **GPU Acceleration**: Hardware-accelerated optimization with JAX transformations
- **Memory Management**: Efficient memory usage for large-scale optimization problems
- **Advanced Transformations**: Optimal use of JIT, grad, vmap for optimization workflows

#### **`ai-systems-architect`** - AI Optimization Systems & Scalability
- **ML Optimization**: Machine learning optimization systems and scalable algorithms
- **Distributed Optimization**: Multi-device and multi-node optimization architecture
- **Production Optimization**: Optimization systems for production AI applications
- **Resource Management**: Computational resource optimization for large-scale problems
- **System Integration**: Optimization system integration with AI infrastructure

#### **`correlation-function-expert`** - Statistical Optimization & Analysis
- **Statistical Optimization**: Statistical methods for parameter estimation and optimization
- **Curve Fitting**: Advanced curve fitting techniques and statistical analysis
- **Uncertainty Quantification**: Statistical uncertainty analysis in optimization results
- **Experimental Design**: Statistical experimental design for optimization studies
- **Model Validation**: Statistical validation of optimization results and models

### Specialized Optimization Agents

#### **`research-intelligence-master`** - Optimization Research & Innovation
- **Optimization Research**: Cutting-edge optimization research and methodology development
- **Algorithm Innovation**: Novel optimization algorithms and breakthrough techniques
- **Academic Standards**: Research-grade optimization for academic publication
- **Cross-Domain Innovation**: Optimization innovation across multiple scientific domains
- **Methodology Development**: Advanced optimization frameworks and methodologies

#### **`systems-architect`** - Optimization Infrastructure & Performance
- **Computational Infrastructure**: System architecture for high-performance optimization
- **Memory Management**: Large-scale optimization memory architecture and efficiency
- **Performance Engineering**: System-level optimization performance and scalability
- **Resource Allocation**: Optimal resource distribution for optimization workloads
- **Fault Tolerance**: Robust optimization systems with failure recovery

#### **`neural-networks-master`** - ML Optimization Expert
- **Neural Network Optimization**: Optimization techniques for neural network training
- **Gradient-Based Methods**: Advanced gradient-based optimization for deep learning
- **Large Model Optimization**: Optimization strategies for large-scale neural networks
- **Training Efficiency**: Optimization methods for efficient neural network training
- **Hyperparameter Optimization**: Advanced hyperparameter optimization techniques

### Advanced Agent Selection Strategies

#### **`auto`** - Intelligent Agent Selection for Optimization
Automatically analyzes optimization requirements and selects optimal agent combinations:
- **Problem Analysis**: Detects optimization type, scale, numerical challenges, performance needs
- **Algorithm Assessment**: Evaluates optimization algorithm requirements and constraints
- **Agent Matching**: Maps optimization challenges to relevant agent expertise
- **Method Optimization**: Balances comprehensive optimization with computational efficiency

#### **`jax`** - JAX-Specialized Optimization Team
- `jax-pro` (JAX ecosystem lead)
- `scientific-computing-master` (numerical methods)
- `ai-systems-architect` (system integration)
- `systems-architect` (performance)

#### **`scientific`** - Scientific Computing Optimization Team
- `scientific-computing-master` (lead)
- `research-intelligence-master` (research methodology)
- `correlation-function-expert` (statistical methods)
- `jax-pro` (JAX implementation)
- Domain-specific experts based on scientific application

#### **`ai`** - AI/ML Optimization Team
- `ai-systems-architect` (lead)
- `neural-networks-master` (ML optimization)
- `scientific-computing-master` (numerical methods)
- `jax-pro` (JAX optimization)
- `systems-architect` (infrastructure)

#### **`optimization`** - Dedicated Optimization Team
- `scientific-computing-master` (lead)
- `ai-systems-architect` (optimization systems)
- `research-intelligence-master` (advanced methods)
- `correlation-function-expert` (statistical optimization)
- `jax-pro` (JAX implementation)

#### **`all`** - Complete 23-Agent Optimization Ecosystem
Activates all relevant agents with intelligent orchestration for breakthrough optimization.

### 23-Agent Optimization Orchestration (`--orchestrate`)

#### **Multi-Agent Optimization Pipeline**
1. **Problem Analysis Phase**: Multiple agents analyze optimization requirements simultaneously
2. **Algorithm Selection**: Collaborative selection of optimal optimization methods
3. **Implementation Optimization**: Agent-coordinated optimization algorithm implementation
4. **Performance Monitoring**: Multi-agent optimization performance tracking and tuning
5. **Convergence Analysis**: Comprehensive optimization convergence and validation

#### **Breakthrough Optimization Discovery (`--breakthrough`)**
- **Cross-Domain Innovation**: Optimization techniques from multiple mathematical domains
- **Emergent Algorithms**: Novel optimization methods through agent collaboration
- **Research-Grade Performance**: Academic and industry-leading optimization standards
- **Adaptive Methods**: Dynamic optimization strategy adaptation based on problem evolution

### Advanced 23-Agent Optimization Examples

```bash
# Intelligent auto-selection for optimization
/jax-nlsq --agents=auto --intelligent --algorithm=TRR --optimize

# Scientific computing optimization with specialized agents
/jax-nlsq --agents=scientific --breakthrough --orchestrate --robust

# AI/ML optimization with performance focus
/jax-nlsq --agents=ai --gpu-accel --optimize --breakthrough

# Research-grade optimization development
/jax-nlsq --agents=all --breakthrough --orchestrate --robust

# JAX-specialized optimization
/jax-nlsq --agents=jax --chunking --optimize --intelligent

# Complete 23-agent optimization ecosystem
/jax-nlsq --agents=all --orchestrate --breakthrough --intelligent

# Large-scale parameter estimation
/jax-nlsq parameter_estimation.py --agents=scientific --intelligent --robust

# High-performance curve fitting
/jax-nlsq curve_fitting.py --agents=optimization --breakthrough --orchestrate

# Production optimization systems
/jax-nlsq production_opt.py --agents=ai --optimize --gpu-accel

# Research optimization methodology
/jax-nlsq research_opt.py --agents=all --breakthrough --orchestrate

# Statistical optimization analysis
/jax-nlsq statistical_fit.py --agents=scientific --intelligent --robust

# Massive dataset optimization
/jax-nlsq massive_data.py --agents=ai --chunking --optimize
```

### Intelligent Agent Selection Examples

```bash
# Optimization Problem Type Detection → Intelligent Agent Selection

# Scientific parameter estimation
/jax-nlsq parameter_study.py --agents=auto --intelligent
# → Selects: scientific-computing-master + correlation-function-expert + research-intelligence-master

# Large-scale curve fitting
/jax-nlsq large_scale_fit.py --agents=auto --intelligent
# → Selects: ai-systems-architect + systems-architect + jax-pro

# Robust statistical optimization
/jax-nlsq robust_optimization.py --agents=auto --intelligent
# → Selects: scientific-computing-master + correlation-function-expert + jax-pro

# High-performance optimization system
/jax-nlsq hpc_optimization.py --agents=auto --intelligent
# → Selects: systems-architect + jax-pro + ai-systems-architect

# Complex research optimization
/jax-nlsq complex_research.py --agents=auto --intelligent
# → Selects: research-intelligence-master + multi-agent-orchestrator + scientific-computing-master

# Comprehensive optimization platform
/jax-nlsq optimization_platform.py --agents=all --orchestrate --breakthrough
# → Activates: All 23 agents with intelligent coordination for maximum optimization effectiveness
```

## Example output

```python
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, grad, value_and_grad, jacfwd, jacrev
import jax.scipy.optimize as jsp_opt
import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any
import time

# Initialize PRNG key
key = random.PRNGKey(0)

# ============================================================================
# 1. NLSQ LIBRARY SETUP AND IMPORTS
# ============================================================================

try:
    import nlsq
    NLSQ_AVAILABLE = True
except ImportError:
    NLSQ_AVAILABLE = False
    print("NLSQ library not available. Using JAX scipy.optimize instead.")

# ============================================================================
# 2. BASIC NONLINEAR LEAST SQUARES
# ============================================================================

def create_test_problem(n_data: int = 100, noise_level: float = 0.1):
    """Create a nonlinear test problem"""

    # Generate synthetic data for exponential decay
    key_data, key_noise = random.split(key)

    # True parameters: amplitude, decay rate, offset
    true_params = jnp.array([2.5, 0.8, 0.1])

    # Generate x data
    x_data = jnp.linspace(0, 5, n_data)

    # Generate clean y data
    def exponential_model(x, params):
        amplitude, decay_rate, offset = params
        return amplitude * jnp.exp(-decay_rate * x) + offset

    y_clean = exponential_model(x_data, true_params)

    # Add noise
    noise = noise_level * random.normal(key_noise, y_clean.shape)
    y_data = y_clean + noise

    return x_data, y_data, true_params, exponential_model

def residual_function(params, x_data, y_data, model_func):
    """Compute residuals for least squares fitting"""
    y_pred = model_func(x_data, params)
    return y_pred - y_data

def jacobian_function(params, x_data, y_data, model_func):
    """Compute Jacobian matrix for residuals"""
    residual_func = lambda p: residual_function(p, x_data, y_data, model_func)
    return jacfwd(residual_func)(params)

# ============================================================================
# 3. TRUST REGION REFLECTIVE (TRR) ALGORITHM
# ============================================================================

@jit
def trust_region_step(params, residuals, jacobian, trust_radius):
    """Single trust region step for nonlinear least squares"""

    # Gauss-Newton direction
    JtJ = jnp.dot(jacobian.T, jacobian)
    Jtr = jnp.dot(jacobian.T, residuals)

    # Regularization for numerical stability
    regularization = 1e-8 * jnp.trace(JtJ) / len(params)
    JtJ_reg = JtJ + regularization * jnp.eye(len(params))

    # Solve normal equations
    try:
        step = jnp.linalg.solve(JtJ_reg, -Jtr)
    except:
        # Fallback to pseudoinverse
        step = -jnp.dot(jnp.linalg.pinv(JtJ_reg), Jtr)

    # Trust region constraint
    step_norm = jnp.linalg.norm(step)
    if step_norm > trust_radius:
        step = step * (trust_radius / step_norm)

    return step

def trust_region_reflective(params_init, x_data, y_data, model_func,
                          max_iterations=100, tolerance=1e-6):
    """Trust Region Reflective algorithm for NLSQ"""

    params = params_init.copy()
    trust_radius = 1.0

    history = {
        'params': [params.copy()],
        'residuals': [],
        'costs': [],
        'trust_radius': [trust_radius]
    }

    for iteration in range(max_iterations):
        # Compute residuals and Jacobian
        residuals = residual_function(params, x_data, y_data, model_func)
        jacobian = jacobian_function(params, x_data, y_data, model_func)

        current_cost = 0.5 * jnp.sum(residuals**2)

        # Store history
        history['residuals'].append(residuals)
        history['costs'].append(current_cost)

        # Check convergence
        gradient = jnp.dot(jacobian.T, residuals)
        if jnp.linalg.norm(gradient) < tolerance:
            print(f"Converged after {iteration} iterations")
            break

        # Compute trust region step
        step = trust_region_step(params, residuals, jacobian, trust_radius)

        # Evaluate new parameters
        params_new = params + step
        residuals_new = residual_function(params_new, x_data, y_data, model_func)
        cost_new = 0.5 * jnp.sum(residuals_new**2)

        # Compute actual vs predicted reduction
        predicted_reduction = -jnp.dot(gradient, step) - 0.5 * jnp.dot(step, jnp.dot(jnp.dot(jacobian.T, jacobian), step))
        actual_reduction = current_cost - cost_new

        # Update trust radius
        if predicted_reduction > 0:
            ratio = actual_reduction / predicted_reduction
            if ratio > 0.75:
                trust_radius = min(2 * trust_radius, 10.0)
            elif ratio < 0.25:
                trust_radius = 0.5 * trust_radius
        else:
            trust_radius = 0.5 * trust_radius

        # Accept or reject step
        if actual_reduction > 0:
            params = params_new

        history['params'].append(params.copy())
        history['trust_radius'].append(trust_radius)

        # Minimum trust radius check
        if trust_radius < 1e-12:
            print(f"Trust radius too small, stopping at iteration {iteration}")
            break

    return params, history

# ============================================================================
# 4. LEVENBERG-MARQUARDT ALGORITHM
# ============================================================================

def levenberg_marquardt(params_init, x_data, y_data, model_func,
                       max_iterations=100, tolerance=1e-6, lambda_init=1e-3):
    """Levenberg-Marquardt algorithm for NLSQ"""

    params = params_init.copy()
    lambda_param = lambda_init

    history = {
        'params': [params.copy()],
        'residuals': [],
        'costs': [],
        'lambda': [lambda_param]
    }

    for iteration in range(max_iterations):
        # Compute residuals and Jacobian
        residuals = residual_function(params, x_data, y_data, model_func)
        jacobian = jacobian_function(params, x_data, y_data, model_func)

        current_cost = 0.5 * jnp.sum(residuals**2)

        # Store history
        history['residuals'].append(residuals)
        history['costs'].append(current_cost)

        # Check convergence
        gradient = jnp.dot(jacobian.T, residuals)
        if jnp.linalg.norm(gradient) < tolerance:
            print(f"Converged after {iteration} iterations")
            break

        # Levenberg-Marquardt step
        JtJ = jnp.dot(jacobian.T, jacobian)
        Jtr = jnp.dot(jacobian.T, residuals)

        # Add damping
        damping_matrix = lambda_param * jnp.diag(jnp.diag(JtJ))
        system_matrix = JtJ + damping_matrix

        # Solve for step
        try:
            step = jnp.linalg.solve(system_matrix, -Jtr)
        except:
            step = -jnp.dot(jnp.linalg.pinv(system_matrix), Jtr)

        # Evaluate new parameters
        params_new = params + step
        residuals_new = residual_function(params_new, x_data, y_data, model_func)
        cost_new = 0.5 * jnp.sum(residuals_new**2)

        # Update damping parameter
        if cost_new < current_cost:
            # Accept step
            params = params_new
            lambda_param = max(lambda_param / 10, 1e-12)
        else:
            # Reject step
            lambda_param = min(lambda_param * 10, 1e12)

        history['params'].append(params.copy())
        history['lambda'].append(lambda_param)

        # Check for excessive damping
        if lambda_param > 1e10:
            print(f"Damping parameter too large, stopping at iteration {iteration}")
            break

    return params, history

# ============================================================================
# 5. MEMORY-EFFICIENT CHUNKED OPTIMIZATION
# ============================================================================

def chunked_residuals(params, x_chunks, y_chunks, model_func):
    """Compute residuals in chunks to save memory"""

    total_residuals = []

    for x_chunk, y_chunk in zip(x_chunks, y_chunks):
        chunk_residuals = residual_function(params, x_chunk, y_chunk, model_func)
        total_residuals.append(chunk_residuals)

    return jnp.concatenate(total_residuals)

def chunked_jacobian(params, x_chunks, y_chunks, model_func):
    """Compute Jacobian in chunks to save memory"""

    total_jacobian = []

    for x_chunk, y_chunk in zip(x_chunks, y_chunks):
        chunk_jacobian = jacobian_function(params, x_chunk, y_chunk, model_func)
        total_jacobian.append(chunk_jacobian)

    return jnp.concatenate(total_jacobian, axis=0)

def create_data_chunks(x_data, y_data, chunk_size=1000):
    """Split data into chunks for memory efficiency"""

    n_data = len(x_data)
    n_chunks = (n_data + chunk_size - 1) // chunk_size

    x_chunks = []
    y_chunks = []

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_data)

        x_chunks.append(x_data[start_idx:end_idx])
        y_chunks.append(y_data[start_idx:end_idx])

    return x_chunks, y_chunks

def chunked_optimization(params_init, x_data, y_data, model_func,
                        algorithm='TRR', chunk_size=1000, **kwargs):
    """Memory-efficient optimization using chunking"""

    # Create data chunks
    x_chunks, y_chunks = create_data_chunks(x_data, y_data, chunk_size)

    print(f"Using {len(x_chunks)} chunks for optimization")

    # Modify residual and jacobian functions for chunking
    def chunked_residual_func(params, x_data, y_data, model_func):
        return chunked_residuals(params, x_chunks, y_chunks, model_func)

    def chunked_jacobian_func(params, x_data, y_data, model_func):
        return chunked_jacobian(params, x_chunks, y_chunks, model_func)

    # Run optimization with chunked functions
    if algorithm == 'TRR':
        return trust_region_reflective(params_init, x_data, y_data, model_func, **kwargs)
    elif algorithm == 'LM':
        return levenberg_marquardt(params_init, x_data, y_data, model_func, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

# ============================================================================
# 6. PERFORMANCE MONITORING AND ANALYSIS
# ============================================================================

def benchmark_optimization(optimization_func, params_init, x_data, y_data,
                         model_func, n_runs=5):
    """Benchmark optimization performance"""

    times = []
    final_costs = []
    iterations = []

    for run in range(n_runs):
        start_time = time.time()
        params_opt, history = optimization_func(params_init, x_data, y_data, model_func)
        end_time = time.time()

        times.append(end_time - start_time)
        final_costs.append(history['costs'][-1])
        iterations.append(len(history['costs']))

    results = {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'avg_cost': np.mean(final_costs),
        'std_cost': np.std(final_costs),
        'avg_iterations': np.mean(iterations),
        'std_iterations': np.std(iterations)
    }

    return results

def memory_usage_analysis(data_size, chunk_size=None):
    """Analyze memory usage for different dataset sizes"""

    if chunk_size is None:
        # Full dataset in memory
        memory_estimate = data_size * 8 * 2  # x and y data, float64
        jacobian_memory = data_size * 8 * 3  # Jacobian matrix (n_data x n_params)
        total_memory = memory_estimate + jacobian_memory
    else:
        # Chunked processing
        memory_estimate = chunk_size * 8 * 2
        jacobian_memory = chunk_size * 8 * 3
        total_memory = memory_estimate + jacobian_memory

    memory_mb = total_memory / (1024 * 1024)

    print(f"Estimated memory usage: {memory_mb:.2f} MB")

    return memory_mb

# ============================================================================
# 7. ADVANCED FEATURES AND UTILITIES
# ============================================================================

def parameter_confidence_intervals(params_opt, x_data, y_data, model_func, alpha=0.05):
    """Compute parameter confidence intervals"""

    # Compute residuals and Jacobian at optimum
    residuals = residual_function(params_opt, x_data, y_data, model_func)
    jacobian = jacobian_function(params_opt, x_data, y_data, model_func)

    # Estimate parameter covariance matrix
    JtJ = jnp.dot(jacobian.T, jacobian)

    try:
        covariance = jnp.linalg.inv(JtJ)
    except:
        covariance = jnp.linalg.pinv(JtJ)

    # Residual variance estimate
    dof = len(residuals) - len(params_opt)  # degrees of freedom
    residual_variance = jnp.sum(residuals**2) / dof

    # Parameter standard errors
    param_std = jnp.sqrt(jnp.diag(covariance) * residual_variance)

    # Confidence intervals (assuming t-distribution)
    from scipy.stats import t
    t_value = t.ppf(1 - alpha/2, dof)

    confidence_intervals = []
    for i, (param, std) in enumerate(zip(params_opt, param_std)):
        lower = param - t_value * std
        upper = param + t_value * std
        confidence_intervals.append((lower, upper))

    return confidence_intervals, param_std

def goodness_of_fit_analysis(params_opt, x_data, y_data, model_func):
    """Analyze goodness of fit"""

    # Compute predictions and residuals
    y_pred = model_func(x_data, params_opt)
    residuals = y_pred - y_data

    # R-squared
    ss_res = jnp.sum(residuals**2)
    ss_tot = jnp.sum((y_data - jnp.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Root mean square error
    rmse = jnp.sqrt(jnp.mean(residuals**2))

    # Mean absolute error
    mae = jnp.mean(jnp.abs(residuals))

    # Adjusted R-squared
    n = len(y_data)
    p = len(params_opt)
    r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

    return {
        'r_squared': float(r_squared),
        'r_squared_adj': float(r_squared_adj),
        'rmse': float(rmse),
        'mae': float(mae),
        'residual_std': float(jnp.std(residuals))
    }

# ============================================================================
# 8. COMPREHENSIVE EXAMPLES
# ============================================================================

def run_nlsq_examples():
    """Run comprehensive NLSQ optimization examples"""

    print("=== JAX NLSQ Optimization Examples ===")

    # Example 1: Basic exponential decay fitting
    print("\n1. Basic exponential decay fitting:")
    x_data, y_data, true_params, model_func = create_test_problem(n_data=100)

    # Initial parameter guess
    params_init = jnp.array([1.0, 1.0, 0.0])

    # Trust Region Reflective
    print("  TRR Algorithm:")
    params_trr, history_trr = trust_region_reflective(
        params_init, x_data, y_data, model_func
    )
    print(f"    True params: {true_params}")
    print(f"    TRR result:  {params_trr}")
    print(f"    Final cost:  {history_trr['costs'][-1]:.6f}")

    # Levenberg-Marquardt
    print("  LM Algorithm:")
    params_lm, history_lm = levenberg_marquardt(
        params_init, x_data, y_data, model_func
    )
    print(f"    LM result:   {params_lm}")
    print(f"    Final cost:  {history_lm['costs'][-1]:.6f}")

    # Example 2: Large dataset with chunking
    print("\n2. Large dataset optimization:")
    x_large, y_large, true_params_large, model_func = create_test_problem(n_data=10000)

    memory_usage_analysis(10000)
    memory_usage_analysis(10000, chunk_size=1000)

    # Chunked optimization
    params_chunked, history_chunked = chunked_optimization(
        params_init, x_large, y_large, model_func,
        algorithm='TRR', chunk_size=1000
    )
    print(f"    Chunked result: {params_chunked}")

    # Example 3: Performance comparison
    print("\n3. Performance comparison:")

    # Benchmark TRR
    trr_results = benchmark_optimization(
        trust_region_reflective, params_init, x_data, y_data, model_func
    )
    print(f"    TRR: {trr_results['avg_time']:.4f}s ± {trr_results['std_time']:.4f}s")

    # Benchmark LM
    lm_results = benchmark_optimization(
        levenberg_marquardt, params_init, x_data, y_data, model_func
    )
    print(f"    LM:  {lm_results['avg_time']:.4f}s ± {lm_results['std_time']:.4f}s")

    # Example 4: Statistical analysis
    print("\n4. Statistical analysis:")

    # Confidence intervals
    conf_intervals, param_std = parameter_confidence_intervals(
        params_trr, x_data, y_data, model_func
    )

    for i, (param, std, (lower, upper)) in enumerate(zip(params_trr, param_std, conf_intervals)):
        print(f"    Parameter {i}: {param:.4f} ± {std:.4f}, CI: [{lower:.4f}, {upper:.4f}]")

    # Goodness of fit
    fit_stats = goodness_of_fit_analysis(params_trr, x_data, y_data, model_func)
    print(f"    R²: {fit_stats['r_squared']:.4f}")
    print(f"    RMSE: {fit_stats['rmse']:.4f}")

# Run examples
run_nlsq_examples()
```

## Algorithm Selection Guide

### Trust Region Reflective (TRR)
- **Best for**: Well-conditioned problems, moderate noise levels
- **Advantages**: Robust convergence, handles ill-conditioned Jacobians well
- **Memory usage**: Moderate (stores trust region information)
- **Recommended for**: General-purpose nonlinear least squares

### Levenberg-Marquardt (LM)
- **Best for**: Problems with good initial guesses, low noise
- **Advantages**: Fast convergence near solution, simple implementation
- **Memory usage**: Low (only stores damping parameter)
- **Recommended for**: Well-behaved problems with good initial estimates

## Dataset Size Guidelines

### Small Datasets (< 1,000 points)
- Use standard algorithms without chunking
- Both TRR and LM perform well
- Memory usage negligible

### Large Datasets (1,000 - 100,000 points)
- Consider chunking for memory efficiency
- TRR generally more robust
- Monitor memory usage

### Massive Datasets (> 100,000 points)
- Always use chunking
- Consider stochastic methods
- Profile memory and performance carefully

## Performance Optimization

### Memory Management
- Use appropriate chunk sizes (1,000-10,000 points per chunk)
- Monitor device memory usage
- Consider gradient checkpointing for very large problems

### Numerical Stability
- Use proper regularization in normal equations
- Handle singular Jacobian matrices with pseudoinverse
- Monitor condition numbers

### Convergence Monitoring
- Track residual norms and parameter changes
- Set appropriate tolerance levels
- Implement maximum iteration limits

## Agent-Enhanced Optimization Integration Patterns

### Complete Optimization Workflow
```bash
# Intelligent optimization analysis and implementation pipeline
/jax-nlsq --agents=auto --intelligent --algorithm=TRR --optimize
/jax-performance --agents=jax --technique=memory --optimization
/jax-sparse-ops --agents=scientific --operation=jacobian --optimize
```

### Scientific Computing Optimization Pipeline
```bash
# High-performance scientific optimization
/jax-nlsq --agents=scientific --breakthrough --orchestrate
/jax-essentials --agents=scientific --operation=grad --optimization
/run-all-tests --agents=scientific --reproducible --optimization
```

### Production Optimization System Infrastructure
```bash
# Large-scale production optimization systems
/jax-nlsq --agents=ai --gpu-accel --optimize --breakthrough
/jax-performance --agents=ai --optimization --gpu-accel
/ci-setup --agents=ai --optimization-monitoring --performance
```

## Related Commands

**Prerequisites**: Commands to run before optimization setup
- `/jax-essentials --agents=auto` - Core JAX operations with optimization considerations
- `/jax-init --agents=scientific` - JAX project setup with optimization configuration

**Core Workflow**: Optimization development with agent intelligence
- `/jax-performance --agents=jax` - Optimization performance tuning
- `/jax-sparse-ops --agents=scientific` - Sparse optimization methods
- `/jax-debug --agents=auto` - Debug optimization convergence issues

**Advanced Integration**: Specialized optimization development
- `/jax-training --agents=ai` - ML training optimization integration
- `/jax-numpyro-prob --agents=scientific` - Probabilistic optimization methods
- `/jax-models --agents=scientific` - Model optimization and parameter estimation

**Quality Assurance**: Optimization validation and testing
- `/generate-tests --agents=auto --type=optimization` - Generate optimization tests
- `/run-all-tests --agents=scientific --reproducible` - Comprehensive optimization testing
- `/check-code-quality --agents=auto --optimization` - Optimization code quality

**Research & Documentation**: Advanced optimization workflows
- `/update-docs --agents=research --type=optimization` - Research-grade optimization documentation
- `/reflection --agents=research --type=optimization` - Optimization methodology analysis
- `/multi-agent-optimize --agents=all --focus=optimization` - Comprehensive optimization enhancement

ARGUMENTS: [--dataset-size=small|large|massive] [--algorithm=TRR|LM] [--gpu-accel] [--chunking] [--agents=auto|jax|scientific|ai|optimization|all] [--orchestrate] [--intelligent] [--breakthrough] [--optimize] [--robust]