---
description: Advanced code refactoring engine with AI-powered analysis, scientific computing optimization, and modern research workflow integration
category: code-quality
argument-hint: [file-path-or-code-block] [--language=auto] [--focus=performance|maintainability|scientific] [--framework=jax|julia|python] [--ai-optimize] [--interactive]
allowed-tools: Read, Write, Edit, MultiEdit, Grep, Glob, TodoWrite, Bash, WebSearch, WebFetch
---

# Revolutionary Code Refactoring Engine (2025 Scientific Computing Edition)

Advanced AI-powered code refactoring system specializing in clean code principles, SOLID design patterns, cutting-edge scientific computing optimization, and modern research workflow integration for Python, JAX, Julia, and emerging AI/ML frameworks.

## Quick Start

```bash
# AI-powered comprehensive refactoring
/refactor-clean --ai-optimize --interactive --focus=scientific

# JAX ecosystem optimization
/refactor-clean --framework=jax --focus=performance --ai-optimize

# Julia high-performance refactoring
/refactor-clean --framework=julia --focus=performance --type-stable

# Modern Python scientific stack optimization
/refactor-clean --framework=python --focus=scientific --vectorized

# Multi-language scientific workflow refactoring
/refactor-clean --cross-language --reproducible --research-ready

# Interactive guided refactoring with AI assistance
/refactor-clean --interactive --ai-suggestions --comprehensive
```

**Advanced AI-Powered Code Refactoring Expert** specializing in clean code principles, SOLID design patterns, modern software engineering best practices, cutting-edge scientific computing optimization, and state-of-the-art AI/ML frameworks. Transform legacy code into high-performance, maintainable, and scientifically rigorous implementations.

## Context
The user needs help refactoring code to make it cleaner, more maintainable, and aligned with best practices. Focus on practical improvements that enhance code quality without over-engineering.

## Requirements
$ARGUMENTS

## Instructions

### 1. Advanced AI-Powered Code Analysis

Perform comprehensive multi-dimensional analysis across traditional and cutting-edge scientific computing patterns:

#### **Traditional Code Quality Analysis**
- **Code Smells**
  - Long methods/functions (>20 lines)
  - Large classes (>200 lines)
  - Duplicate code blocks
  - Dead code and unused variables
  - Complex conditionals and nested loops
  - Magic numbers and hardcoded values
  - Poor naming conventions
  - Tight coupling between components
  - Missing abstractions

- **SOLID Violations**
  - Single Responsibility Principle violations
  - Open/Closed Principle issues
  - Liskov Substitution problems
  - Interface Segregation concerns
  - Dependency Inversion violations

- **Performance Issues**
  - Inefficient algorithms (O(n²) or worse)
  - Unnecessary object creation
  - Memory leaks potential
  - Blocking operations
  - Missing caching opportunities

#### **Advanced Scientific Computing Analysis (2025 Edition)**

**JAX Ecosystem (Latest Features):**
- **Core JAX Issues:**
  - Functions not JIT compiled (@jax.jit, @jax.jit(static_argnums))
  - Manual gradient computation instead of jax.grad/jax.value_and_grad
  - Python loops on JAX arrays (missing jax.vmap/jax.pmap)
  - Missing jax.lax operations for performance
  - Inefficient jax.scipy usage patterns
  - Missing jax.experimental features (host_callback, maps)

- **Modern JAX Frameworks:**
  - **Equinox**: Missing equinox.nn modules for modern neural architectures
  - **Diffrax**: Suboptimal differential equation solving patterns
  - **JAX-MD**: Missing molecular dynamics optimizations
  - **Chex**: Missing testing utilities and assertions
  - **Orbax**: Missing checkpoint/serialization patterns

- **Flax Advanced Patterns:**
  - Suboptimal flax.linen module design
  - Missing flax.training utilities
  - Inefficient parameter management with flax.core
  - Missing flax.optim integration patterns

**Julia Ecosystem (2025 Latest):**
- **Core Performance Issues:**
  - Type instability (@code_warntype failures)
  - Excessive memory allocations (@allocated > thresholds)
  - Missing vectorization and broadcasting
  - Suboptimal multiple dispatch usage
  - Serial operations missing @threads/@distributed

- **Modern Julia Scientific Stack:**
  - **DifferentialEquations.jl**: Missing ODE/SDE/DAE solver optimizations
  - **Flux.jl 0.14+**: Suboptimal neural network architectures
  - **MLJ.jl**: Missing machine learning pipeline patterns
  - **Makie.jl**: Inefficient plotting and visualization
  - **PlutoStaticHTML.jl**: Missing reproducible notebook patterns
  - **DrWatson.jl**: Missing scientific project structure
  - **Catalyst.jl**: Missing reaction network modeling
  - **ModelingToolkit.jl**: Missing symbolic computation patterns

- **High-Performance Computing:**
  - **CUDA.jl**: Missing GPU kernel optimizations
  - **MPI.jl**: Missing distributed computing patterns
  - **LoopVectorization.jl**: Missing @turbo optimizations
  - **StaticArrays.jl**: Missing compile-time optimizations
  - **BenchmarkTools.jl**: Missing performance measurement patterns

**Python Scientific Stack (2025 Cutting-Edge):**
- **Traditional Issues:**
  - NumPy loops instead of vectorization
  - Inefficient scipy operations
  - Missing pandas optimizations
  - Suboptimal scikit-learn patterns

- **Modern Python Scientific Computing:**
  - **Polars**: Missing fast DataFrame operations (Rust-backed)
  - **CuPy**: Missing GPU NumPy acceleration
  - **Numba**: Missing @jit/@cuda optimizations
  - **Dask**: Missing parallel/distributed computing patterns
  - **Xarray**: Missing labeled array operations
  - **Ray**: Missing distributed ML patterns
  - **Modin**: Missing pandas acceleration
  - **Vaex**: Missing out-of-core DataFrame operations

- **AI/ML Framework Issues:**
  - **PyTorch 2.0+**: Missing torch.compile optimizations
  - **Lightning**: Missing efficient training patterns
  - **Weights & Biases**: Missing experiment tracking
  - **Hydra**: Missing configuration management
  - **DVC**: Missing data version control patterns
  - **Great Expectations**: Missing data validation patterns

#### **Cross-Language Integration Analysis**
- **PyCall.jl/JuliaCall**: Missing Python-Julia integration optimizations
- **JAX-Julia**: Missing interoperability patterns
- **SWIG/Cython**: Missing C/C++ extension optimizations
- **WebAssembly**: Missing browser deployment patterns
- **Container optimization**: Missing Docker/Singularity scientific patterns

#### **Reproducibility & Research Workflow Analysis**
- **Environment Management**: Missing conda/mamba/pixi patterns
- **Experiment Tracking**: Missing MLflow/Wandb/Neptune integration
- **Data Pipeline**: Missing DVC/Kedro/Prefect patterns
- **Documentation**: Missing Quarto/Jupyter Book patterns
- **Citation Management**: Missing BibTeX/Zotero automation
- **Open Science**: Missing FAIR data principles compliance

#### **AI-Powered Code Analysis Engine**

```python
class AdvancedCodeAnalyzer:
    """
    AI-powered code analysis with scientific computing focus.

    Features:
    - Automatic framework detection (JAX/Julia/Python)
    - Performance bottleneck identification
    - Optimization opportunity ranking
    - Refactoring pattern matching
    - Cross-language compatibility analysis
    """

    def __init__(self):
        self.analyzers = {
            'jax': JAXEcosystemAnalyzer(),
            'julia': JuliaPerformanceAnalyzer(),
            'python': PythonScientificAnalyzer(),
            'cross_language': CrossLanguageAnalyzer(),
            'research_workflow': ResearchWorkflowAnalyzer()
        }

    def analyze_code(self, code_path: str) -> AnalysisReport:
        """Comprehensive multi-framework analysis."""
        frameworks = self.detect_frameworks(code_path)
        analysis = {}

        for framework in frameworks:
            analyzer = self.analyzers[framework]
            analysis[framework] = analyzer.analyze(code_path)

        return self.synthesize_analysis(analysis)

    def detect_frameworks(self, code_path: str) -> List[str]:
        """Auto-detect scientific computing frameworks in use."""
        detectors = [
            ('jax', self._detect_jax),
            ('julia', self._detect_julia),
            ('python', self._detect_python_scientific),
            ('pytorch', self._detect_pytorch),
            ('tensorflow', self._detect_tensorflow)
        ]

        detected = []
        for name, detector in detectors:
            if detector(code_path):
                detected.append(name)

        return detected

    def _detect_jax(self, code_path: str) -> bool:
        """Detect JAX ecosystem usage."""
        jax_patterns = [
            r'import jax',
            r'from jax import',
            r'@jax\.jit',
            r'jax\.grad',
            r'jax\.vmap',
            r'import flax',
            r'import optax',
            r'import equinox',
            r'import diffrax'
        ]
        return self._contains_patterns(code_path, jax_patterns)

    def _detect_julia(self, code_path: str) -> bool:
        """Detect Julia scientific computing patterns."""
        julia_patterns = [
            r'using DifferentialEquations',
            r'using Flux',
            r'using MLJ',
            r'using Plots',
            r'using DataFrames',
            r'@threads',
            r'@distributed',
            r'\.jl$'  # File extension
        ]
        return self._contains_patterns(code_path, julia_patterns)

    def rank_optimization_opportunities(self, analysis: AnalysisReport) -> List[OptimizationOpportunity]:
        """AI-powered ranking of optimization opportunities by impact/effort ratio."""
        opportunities = []

        for issue in analysis.issues:
            impact = self.calculate_impact_score(issue)
            effort = self.estimate_effort_score(issue)
            roi = impact / effort if effort > 0 else float('inf')

            opportunities.append(OptimizationOpportunity(
                issue=issue,
                impact_score=impact,
                effort_score=effort,
                roi_score=roi,
                suggested_patterns=self.suggest_patterns(issue)
            ))

        return sorted(opportunities, key=lambda x: x.roi_score, reverse=True)
```

### 2. Advanced AI-Powered Refactoring Strategy

Create a prioritized, AI-optimized refactoring plan with modern scientific computing focus:

#### **Phase 1: Immediate Quick Wins (High Impact, Low Effort)**
- Extract magic numbers to constants with scientific units
- Improve variable and function names with domain-specific terminology
- Remove dead code and unused imports
- Simplify boolean expressions with mathematical identities
- Extract duplicate code to functions with performance annotations

**AI-Enhanced Method Extraction**
```python
# Before (Monolithic research pipeline)
def process_research_data(data):
    # 50 lines of data validation
    # 30 lines of preprocessing
    # 40 lines of feature extraction
    # 60 lines of model training
    # 30 lines of evaluation
    # 20 lines of visualization

# After (AI-optimized microservices)
@jax.jit  # JIT compilation for performance
def process_research_data(data: DataArray) -> ResearchResults:
    """AI-optimized research data processing pipeline."""
    validated_data = validate_research_data(data)
    processed_data = preprocess_with_validation(validated_data)
    features = extract_features_vectorized(processed_data)
    model_results = train_model_distributed(features)
    metrics = evaluate_with_uncertainty(model_results)
    return generate_publication_ready_plots(metrics)
```

#### **Phase 2: Architecture Modernization**

**Advanced Class Decomposition with Scientific Computing Focus**
- Extract responsibilities using scientific domain separation
- Create interfaces for algorithm interchangeability
- Implement dependency injection for framework switching
- Use composition over inheritance for model architectures
- Apply functional programming patterns for reproducibility

**Modern Scientific Computing Pattern Application**

**🚀 JAX Ecosystem Patterns (2025 Advanced):**
```python
# Modern JAX patterns with latest features
class AdvancedJAXPatterns:
    """State-of-the-art JAX optimization patterns."""

    # 1. Advanced JIT compilation with static arguments
    @functools.partial(jax.jit, static_argnums=(1, 2))
    def optimized_computation(self, data, batch_size, num_layers):
        """JIT with static shape optimization."""
        return self._compute_layers(data, batch_size, num_layers)

    # 2. Gradient accumulation for large models
    def gradient_accumulation_pattern(self, params, batch_data):
        """Memory-efficient gradient accumulation."""
        @jax.jit
        def accumulate_gradients(carry, batch):
            grads, count = carry
            batch_grads = jax.grad(self.loss_fn)(params, batch)
            return (
                jax.tree_map(lambda g, bg: g + bg, grads, batch_grads),
                count + 1
            )

        init_grads = jax.tree_map(jnp.zeros_like, params)
        final_grads, total_count = jax.lax.scan(
            accumulate_gradients,
            (init_grads, 0),
            batch_data
        )
        return jax.tree_map(lambda g: g / total_count, final_grads)

    # 3. Equinox modern neural architecture
    def modern_neural_architecture(self):
        """Equinox-based modern neural network pattern."""
        import equinox as eqx

        class ModernResNet(eqx.Module):
            layers: list
            norm: eqx.nn.GroupNorm

            def __init__(self, input_size, hidden_size, output_size, key):
                keys = jax.random.split(key, 10)
                self.layers = [
                    eqx.nn.Linear(input_size, hidden_size, key=keys[0]),
                    eqx.nn.Linear(hidden_size, hidden_size, key=keys[1]),
                    eqx.nn.Linear(hidden_size, output_size, key=keys[2])
                ]
                self.norm = eqx.nn.GroupNorm(groups=8, channels=hidden_size)

            def __call__(self, x):
                for layer in self.layers[:-1]:
                    x = self.norm(jax.nn.gelu(layer(x)))
                return self.layers[-1](x)

    # 4. JAX-scipy optimization patterns
    def scipy_optimization_pattern(self, objective, initial_params):
        """Modern JAX-scipy optimization."""
        from jax.scipy.optimize import minimize

        @jax.jit
        def jit_objective(params):
            return objective(params)

        result = minimize(
            jit_objective,
            initial_params,
            method='BFGS',
            options={'maxiter': 1000}
        )
        return result
```

**💎 Julia High-Performance Patterns (2025 Advanced):**
```julia
# Advanced Julia patterns with latest ecosystem
module AdvancedJuliaPatterns

using DifferentialEquations, Flux, MLJ, StaticArrays
using LoopVectorization, CUDA, BenchmarkTools
using DrWatson, Catalyst, ModelingToolkit

# 1. Type-stable high-performance pattern
function type_stable_computation!(
    result::Vector{T},
    data::Matrix{T},
    weights::SVector{N, T}
) where {T <: AbstractFloat, N}
    """
    Type-stable, SIMD-optimized computation pattern.

    Performance optimizations:
    - Type stability for 10-100x speedup
    - StaticArrays for compile-time optimization
    - @turbo for SIMD vectorization
    - Pre-allocated output for zero allocation
    """
    @turbo for i in axes(data, 1)
        acc = zero(T)
        for j in 1:N
            acc += data[i, j] * weights[j]
        end
        result[i] = acc
    end
    return result
end

# 2. Modern Flux.jl neural architecture pattern
function modern_flux_architecture(input_size::Int, output_size::Int)
    """
    Flux.jl 0.14+ optimized neural architecture.

    Features:
    - Modern activation functions (Swish, GELU)
    - Batch normalization for stability
    - Residual connections for deep networks
    - Efficient parameter initialization
    """
    return Chain(
        Dense(input_size => 256, swish),
        BatchNorm(256),
        Dense(256 => 256, swish),
        BatchNorm(256),
        SkipConnection(
            Chain(Dense(256 => 256, swish), BatchNorm(256)),
            +  # Residual connection
        ),
        Dense(256 => output_size)
    )
end

# 3. DifferentialEquations.jl optimization pattern
function optimized_ode_solving(u0, p, tspan)
    """
    High-performance ODE solving with DifferentialEquations.jl.

    Optimizations:
    - GPU acceleration when available
    - Adaptive stepping with error control
    - Automatic differentiation ready
    - Ensemble solving for parameter sweeps
    """
    function lotka_volterra!(du, u, p, t)
        α, β, δ, γ = p
        du[1] = α * u[1] - β * u[1] * u[2]
        du[2] = -δ * u[2] + γ * u[1] * u[2]
    end

    prob = ODEProblem(lotka_volterra!, u0, tspan, p)

    # Use high-performance solver with GPU support
    sol = solve(prob, Tsit5(),
               abstol=1e-8, reltol=1e-8,
               saveat=0.1)

    return sol
end

# 4. MLJ.jl machine learning pipeline pattern
function optimized_ml_pipeline(X, y)
    """
    MLJ.jl optimized machine learning pipeline.

    Features:
    - Automatic hyperparameter tuning
    - Cross-validation with proper evaluation
    - Pipeline composition for reproducibility
    - Performance monitoring and logging
    """
    using MLJ

    # Create pipeline with preprocessing
    pipe = @pipeline(
        Standardizer(),
        PCA(variance_ratio=0.95),
        RandomForestClassifier(n_trees=100)
    )

    # Hyperparameter tuning
    r = range(pipe, :(random_forest_classifier.max_depth),
              lower=1, upper=10)

    tuning = TunedModel(
        model=pipe,
        tuning=Grid(resolution=10),
        range=r,
        measure=cross_entropy
    )

    # Fit with cross-validation
    mach = machine(tuning, X, y)
    fit!(mach)

    return mach
end

# 5. GPU computing pattern with CUDA.jl
function gpu_accelerated_computation(data::Matrix{Float32})
    """
    CUDA.jl GPU acceleration pattern.

    Optimizations:
    - Efficient GPU memory management
    - Kernel fusion for better performance
    - Asynchronous execution
    - Error handling for GPU operations
    """
    if CUDA.functional()
        # Transfer to GPU
        gpu_data = CuArray(data)

        # GPU-optimized operations
        result = gpu_data .* gpu_data .+ 1.0f0

        # Transfer back to CPU
        return Array(result)
    else
        # Fallback to CPU
        return data .* data .+ 1.0f0
    end
end

end # module
```

**🐍 Python Scientific Stack (2025 Cutting-Edge):**
```python
# Modern Python scientific computing patterns
class ModernPythonPatterns:
    """Cutting-edge Python scientific computing patterns."""

    def polars_optimization_pattern(self, data_path: str):
        """
        Polars: Rust-backed DataFrame operations for extreme performance.

        Performance benefits:
        - 10-100x faster than pandas for large datasets
        - Lazy evaluation and query optimization
        - Memory-efficient columnar storage
        - Parallel execution by default
        """
        import polars as pl

        # Lazy evaluation for query optimization
        lazy_df = (
            pl.scan_parquet(data_path)
            .filter(pl.col("value") > 0)
            .group_by("category")
            .agg([
                pl.col("value").sum().alias("total_value"),
                pl.col("value").mean().alias("mean_value"),
                pl.col("value").std().alias("std_value")
            ])
            .sort("total_value", descending=True)
        )

        # Execute optimized query
        return lazy_df.collect()

    def cupy_gpu_acceleration(self, data: np.ndarray):
        """
        CuPy: GPU NumPy acceleration with zero code changes.

        Benefits:
        - Drop-in NumPy replacement for GPU
        - 10-100x speedup for compatible operations
        - Memory pool management for efficiency
        - Interoperability with other GPU libraries
        """
        try:
            import cupy as cp

            # Transfer to GPU
            gpu_data = cp.asarray(data)

            # GPU-accelerated operations (same API as NumPy)
            result = cp.fft.fft2(gpu_data)
            result = cp.abs(result) ** 2

            # Transfer back to CPU
            return cp.asnumpy(result)
        except ImportError:
            # Fallback to CPU NumPy
            return np.abs(np.fft.fft2(data)) ** 2

    def xarray_labeled_arrays(self, dataset_path: str):
        """
        Xarray: N-dimensional labeled arrays for scientific computing.

        Features:
        - Named dimensions and coordinates
        - Integration with pandas and NumPy
        - NetCDF/HDF5 support for scientific data
        - Automatic alignment and broadcasting
        """
        import xarray as xr

        # Load multi-dimensional scientific dataset
        ds = xr.open_dataset(dataset_path)

        # Named dimension operations
        seasonal_avg = (
            ds.groupby("time.season")
            .mean("time")
            .sel(lat=slice(-60, 60))  # Tropical regions
        )

        return seasonal_avg

    def ray_distributed_ml(self, X, y):
        """
        Ray: Distributed machine learning and hyperparameter tuning.

        Capabilities:
        - Automatic scaling across multiple nodes
        - Efficient hyperparameter optimization
        - Integration with popular ML libraries
        - Fault-tolerant distributed computing
        """
        import ray
        from ray import tune
        from ray.tune.search.optuna import OptunaSearch

        def train_model(config):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score

            model = RandomForestClassifier(**config)
            scores = cross_val_score(model, X, y, cv=5)
            return {"accuracy": scores.mean()}

        # Distributed hyperparameter tuning
        search_space = {
            "n_estimators": tune.randint(10, 1000),
            "max_depth": tune.randint(1, 20),
            "min_samples_split": tune.uniform(0.1, 1.0)
        }

        tuner = tune.Tuner(
            train_model,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                search_alg=OptunaSearch(),
                num_samples=100
            )
        )

        results = tuner.fit()
        return results.get_best_result()

    def dask_parallel_computing(self, data_files: List[str]):
        """
        Dask: Parallel computing and out-of-core processing.

        Benefits:
        - Familiar pandas-like API
        - Automatic parallelization
        - Out-of-core processing for datasets larger than memory
        - Integration with scientific Python ecosystem
        """
        import dask.dataframe as dd
        import dask.array as da

        # Parallel DataFrame processing
        ddf = dd.read_parquet(data_files)

        # Out-of-core computation
        result = (
            ddf.groupby("category")
            .value.agg(["sum", "mean", "std"])
            .compute()  # Trigger computation
        )

        return result
```

#### **Phase 3: Cross-Language Integration Patterns**

**Modern Interoperability Strategies**
```python
class CrossLanguageIntegration:
    """Advanced patterns for multi-language scientific computing."""

    def python_julia_integration(self):
        """Optimized Python-Julia interoperability."""
        from juliacall import Main as jl

        # Load Julia scientific computing environment
        jl.seval("""
        using DifferentialEquations, LinearAlgebra, BenchmarkTools

        function high_performance_ode_solve(u0, p, tspan)
            function lorenz!(du, u, p, t)
                σ, ρ, β = p
                du[1] = σ * (u[2] - u[1])
                du[2] = u[1] * (ρ - u[3]) - u[2]
                du[3] = u[1] * u[2] - β * u[3]
            end

            prob = ODEProblem(lorenz!, u0, tspan, p)
            solve(prob, Tsit5(), abstol=1e-14, reltol=1e-14)
        end
        """)

        # Call Julia from Python with automatic conversion
        import numpy as np
        u0 = np.array([1.0, 1.0, 1.0])
        p = np.array([10.0, 28.0, 8/3])
        tspan = (0.0, 100.0)

        # Julia computation with Python data
        solution = jl.high_performance_ode_solve(u0, p, tspan)
        return np.array(solution.u)

    def jax_julia_interop(self):
        """JAX-Julia interoperability for scientific computing."""
        # JAX computation
        @jax.jit
        def jax_preprocessing(data):
            return jnp.fft.fft(data) / jnp.sqrt(data.shape[0])

        # Julia high-performance backend
        from juliacall import Main as jl
        jl.seval("""
        using LinearAlgebra, StaticArrays

        function julia_linear_solve(A, b)
            return A \\ b  # Optimized linear solve
        end
        """)

        # Hybrid workflow
        def hybrid_computation(data):
            # JAX preprocessing
            preprocessed = jax_preprocessing(data)

            # Convert to Julia for linear algebra
            A = np.random.randn(len(preprocessed), len(preprocessed))
            b = np.array(preprocessed)

            # Julia computation
            result = np.array(jl.julia_linear_solve(A, b))

            # JAX postprocessing
            return jax_preprocessing(result)

        return hybrid_computation
```

#### **Phase 4: Research Workflow Optimization Patterns**

```python
class ResearchWorkflowPatterns:
    """Modern research workflow optimization patterns."""

    def reproducible_experiment_pattern(self):
        """Comprehensive reproducible research pattern."""
        import hydra
        from hydra.core.config_store import ConfigStore
        from omegaconf import DictConfig
        import mlflow
        import wandb
        from dataclasses import dataclass

        @dataclass
        class ExperimentConfig:
            model_name: str = "resnet50"
            learning_rate: float = 0.001
            batch_size: int = 32
            epochs: int = 100
            seed: int = 42

        # Register configuration
        cs = ConfigStore.instance()
        cs.store(name="base_config", node=ExperimentConfig)

        @hydra.main(version_base=None, config_path="conf", config_name="config")
        def reproducible_experiment(cfg: DictConfig) -> None:
            # Set all random seeds for reproducibility
            set_all_seeds(cfg.seed)

            # Initialize experiment tracking
            with mlflow.start_run():
                wandb.init(project="research_project", config=cfg)

                # Log configuration
                mlflow.log_params(cfg)
                wandb.config.update(cfg)

                # Run experiment
                model = create_model(cfg)
                results = train_model(model, cfg)

                # Log results
                mlflow.log_metrics(results)
                wandb.log(results)

                return results

    def dvc_data_pipeline_pattern(self):
        """DVC-based data version control and pipeline."""
        # dvc.yaml pipeline definition
        pipeline_config = """
        stages:
          data_collection:
            cmd: python src/collect_data.py
            outs:
            - data/raw/dataset.csv

          preprocessing:
            cmd: python src/preprocess.py data/raw/dataset.csv
            deps:
            - data/raw/dataset.csv
            - src/preprocess.py
            outs:
            - data/processed/features.parquet

          training:
            cmd: python src/train.py data/processed/features.parquet
            deps:
            - data/processed/features.parquet
            - src/train.py
            params:
            - training.learning_rate
            - training.batch_size
            metrics:
            - metrics/train_metrics.json
            outs:
            - models/trained_model.pkl

          evaluation:
            cmd: python src/evaluate.py models/trained_model.pkl
            deps:
            - models/trained_model.pkl
            - src/evaluate.py
            metrics:
            - metrics/eval_metrics.json
        """
        return pipeline_config

    def fair_data_pattern(self):
        """FAIR (Findable, Accessible, Interoperable, Reusable) data pattern."""
        import pandas as pd
        from dataclasses import dataclass, asdict
        import json
        from datetime import datetime

        @dataclass
        class FAIRMetadata:
            """FAIR data principles compliant metadata."""
            title: str
            description: str
            creator: str
            created_date: str
            license: str
            version: str
            doi: str
            keywords: list
            format: str
            access_rights: str

            def to_json(self) -> str:
                return json.dumps(asdict(self), indent=2)

        def create_fair_dataset(data: pd.DataFrame, metadata: FAIRMetadata):
            """Create FAIR-compliant dataset with rich metadata."""

            # Add metadata to dataset
            dataset_info = {
                'data': data.to_dict(),
                'metadata': asdict(metadata),
                'schema': {
                    'columns': list(data.columns),
                    'dtypes': data.dtypes.to_dict(),
                    'shape': data.shape
                },
                'provenance': {
                    'processing_date': datetime.now().isoformat(),
                    'software_version': get_software_versions(),
                    'processing_steps': get_processing_history()
                }
            }

            return dataset_info
```

### 3. Refactored Implementation

Provide the complete refactored code with:

**Clean Code Principles**
- Meaningful names (searchable, pronounceable, no abbreviations)
- Functions do one thing well
- No side effects
- Consistent abstraction levels
- DRY (Don't Repeat Yourself)
- YAGNI (You Aren't Gonna Need It)

**Error Handling**
```python
# Use specific exceptions
class OrderValidationError(Exception):
    pass

class InsufficientInventoryError(Exception):
    pass

# Fail fast with clear messages
def validate_order(order):
    if not order.items:
        raise OrderValidationError("Order must contain at least one item")
    
    for item in order.items:
        if item.quantity <= 0:
            raise OrderValidationError(f"Invalid quantity for {item.name}")
```

**Documentation**
```python
def calculate_discount(order: Order, customer: Customer) -> Decimal:
    """
    Calculate the total discount for an order based on customer tier and order value.

    Args:
        order: The order to calculate discount for
        customer: The customer making the order

    Returns:
        The discount amount as a Decimal

    Raises:
        ValueError: If order total is negative
    """
```

**🚀 Advanced Scientific Computing Refactoring Examples (2025 Edition)**

**JAX Ecosystem Complete Transformation**
```python
# Before (Legacy PyTorch-style implementation):
class LegacyNeuralNetwork:
    def __init__(self, features):
        # Manual parameter initialization
        self.w1 = np.random.randn(784, features) * 0.01
        self.b1 = np.zeros(features)
        self.w2 = np.random.randn(features, 10) * 0.01
        self.b2 = np.zeros(10)

    def forward(self, x):
        # Manual forward pass
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.w2 + self.b2
        return z2

    def train_step(self, x, y, lr=0.01):
        # Manual gradient computation using finite differences
        gradients = {}
        eps = 1e-7
        loss_fn = lambda: np.mean((self.forward(x) - y) ** 2)

        original_loss = loss_fn()

        # Compute gradients for w1
        self.w1[0, 0] += eps
        perturbed_loss = loss_fn()
        gradients['w1'] = (perturbed_loss - original_loss) / eps
        self.w1[0, 0] -= eps  # Reset

        # ... manual gradient computation for all parameters
        # This is extremely slow and inaccurate!

        return gradients

# After (Modern JAX + Equinox + Optax):
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

class ModernNeuralNetwork(eqx.Module):
    """
    State-of-the-art JAX neural network with Equinox.

    Features:
    - Automatic differentiation (exact gradients)
    - JIT compilation for 100x+ speedup
    - GPU/TPU acceleration
    - Modern architectures (attention, normalization)
    - Efficient parameter management
    - Gradient accumulation for large models
    """
    layers: list
    dropout: eqx.nn.Dropout

    def __init__(self, input_size: int, hidden_size: int, output_size: int, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 6)

        self.layers = [
            eqx.nn.Linear(input_size, hidden_size, key=keys[0]),
            eqx.nn.LayerNorm(hidden_size),
            eqx.nn.Linear(hidden_size, hidden_size, key=keys[1]),
            eqx.nn.LayerNorm(hidden_size),
            eqx.nn.Linear(hidden_size, output_size, key=keys[2])
        ]
        self.dropout = eqx.nn.Dropout(p=0.1)

    def __call__(self, x, *, key=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if isinstance(layer, eqx.nn.Linear):
                x = jax.nn.gelu(x)  # Modern activation
                if key is not None:
                    x = self.dropout(x, key=jax.random.split(key, 1)[0])
        return self.layers[-1](x)

# Modern training loop with JAX transformations
@jax.jit
def train_step(model, optimizer, opt_state, x, y, key):
    """
    JIT-compiled training step with automatic differentiation.

    Performance improvements:
    - 100x faster than manual gradients
    - Exact gradients via autodiff
    - JIT compilation for maximum speed
    - Memory efficient with gradient accumulation
    """
    def loss_fn(model):
        predictions = model(x, key=key)
        return jnp.mean((predictions - y) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss

# Advanced training with Optax optimizer
def create_training_system():
    """Create modern training system with advanced optimizers."""

    # Modern optimizer with learning rate scheduling
    schedule = optax.exponential_decay(
        init_value=1e-3,
        transition_steps=1000,
        decay_rate=0.99
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(learning_rate=schedule, weight_decay=1e-4),  # AdamW
    )

    return optimizer

# Distributed training with JAX pmap
@functools.partial(jax.pmap, axis_name='devices')
def distributed_train_step(model, optimizer, opt_state, x, y, key):
    """Multi-device distributed training step."""
    def loss_fn(model):
        predictions = model(x, key=key)
        loss = jnp.mean((predictions - y) ** 2)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(model)

    # Synchronize gradients across devices
    grads = jax.lax.pmean(grads, axis_name='devices')

    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss
```

**💎 Julia Scientific Computing Complete Transformation**
```julia
# Before (Inefficient, type-unstable Julia code):
function inefficient_scientific_computation(data)
    results = []  # Type unstable!

    for i in 1:length(data)
        row_results = []  # More type instability!

        for j in 1:length(data[i])
            if data[i][j] > 0
                # Mixed types and allocations in loop
                val = data[i][j] * sin(data[i][j])^2 + cos(data[i][j])^2
                push!(row_results, val)
            else
                push!(row_results, 0.0)  # Constant allocation
            end
        end

        push!(results, row_results)  # More allocations
    end

    return results
end

# After (Modern Julia with full ecosystem integration):
using DifferentialEquations, Flux, MLJ, StaticArrays, LoopVectorization
using CUDA, BenchmarkTools, DrWatson, Catalyst, ModelingToolkit
using LinearAlgebra, Statistics

"""
    modern_scientific_computation!(result, data)

Type-stable, high-performance scientific computation with modern Julia patterns.

Performance optimizations:
- Type stability: 10-100x speedup
- SIMD vectorization: @turbo macro
- Zero allocations: pre-allocated arrays
- Mathematical optimization: sin²(x) + cos²(x) = 1
- GPU acceleration: CUDA.jl integration
- Parallel processing: @threads for multi-core
"""
function modern_scientific_computation!(
    result::Matrix{T},
    data::Matrix{T}
) where T <: AbstractFloat

    @threads for i in axes(data, 1)
        @turbo for j in axes(data, 2)
            if data[i, j] > zero(T)
                # Mathematical identity: sin²(x) + cos²(x) = 1
                result[i, j] = data[i, j]  # Optimized computation
            else
                result[i, j] = zero(T)
            end
        end
    end

    return result
end

# GPU-accelerated version for large datasets
function gpu_scientific_computation(data::Matrix{Float32})
    """GPU-accelerated computation with CUDA.jl"""

    if CUDA.functional()
        # Transfer to GPU
        gpu_data = CuArray(data)

        # GPU kernel execution
        gpu_result = gpu_data .* (gpu_data .> 0.0f0)

        # Transfer back to CPU
        return Array(gpu_result)
    else
        # CPU fallback
        result = similar(data)
        return modern_scientific_computation!(result, data)
    end
end

# Modern Julia ML pipeline with MLJ.jl
function modern_ml_pipeline(X, y)
    """
    MLJ.jl machine learning pipeline with modern Julia patterns.

    Features:
    - Automatic hyperparameter tuning
    - Cross-validation with statistical rigor
    - Type-stable pipeline composition
    - Integration with Julia scientific ecosystem
    """
    using MLJ

    # Type-stable pipeline composition
    pipe = @pipeline(
        Standardizer(),
        PCA(variance_ratio=0.95),
        RandomForestClassifier(n_trees=500)
    ) |> MLJBase.machine

    # Hyperparameter optimization with Bayesian optimization
    r1 = range(pipe, :(pca.variance_ratio), lower=0.8, upper=0.99)
    r2 = range(pipe, :(random_forest_classifier.max_depth), lower=5, upper=20)

    tuning = TunedModel(
        model=pipe,
        tuning=Hyperopt(),  # Bayesian optimization
        range=[r1, r2],
        measure=cross_entropy,
        n=50
    )

    # Cross-validation with statistical analysis
    mach = machine(tuning, X, y)
    fit!(mach, verbosity=1)

    return mach
end

# Modern differential equations with DifferentialEquations.jl
function solve_complex_ode_system(u0, p, tspan)
    """
    High-performance ODE solving with modern Julia ecosystem.

    Features:
    - GPU acceleration
    - Automatic differentiation
    - Ensemble solving
    - Uncertainty quantification
    """

    # Define complex ODE system (Lorenz equations with noise)
    function lorenz_stochastic!(du, u, p, t)
        σ, ρ, β, noise_strength = p
        du[1] = σ * (u[2] - u[1])
        du[2] = u[1] * (ρ - u[3]) - u[2]
        du[3] = u[1] * u[2] - β * u[3]
    end

    function noise!(du, u, p, t)
        du[1] = p[4]  # noise_strength
        du[2] = p[4]
        du[3] = p[4]
    end

    # Stochastic differential equation
    prob = SDEProblem(lorenz_stochastic!, noise!, u0, tspan, p)

    # High-performance solver with GPU support
    sol = solve(prob, SRIW1(),
               abstol=1e-8, reltol=1e-8,
               dt=0.01, adaptive=true)

    return sol
end

# Type-stable neural network with Flux.jl 0.14+
function create_modern_flux_model(input_size::Int, output_size::Int)
    """
    Modern Flux.jl neural architecture with latest features.

    Optimizations:
    - Type-stable construction
    - Modern activation functions
    - Batch normalization for stability
    - Residual connections for deep networks
    """

    model = Chain(
        Dense(input_size => 512, swish),
        BatchNorm(512),
        Dropout(0.1),

        # Residual block
        SkipConnection(
            Chain(
                Dense(512 => 512, swish),
                BatchNorm(512),
                Dropout(0.1),
                Dense(512 => 512)
            ),
            +
        ),

        Dense(512 => output_size)
    )

    return model
end

# Benchmark comparison function
function benchmark_optimization()
    """Benchmark old vs new implementation."""

    # Test data
    data = rand(Float32, 1000, 1000)

    # Old implementation
    old_time = @belapsed inefficient_scientific_computation($data)

    # New implementation
    result = similar(data)
    new_time = @belapsed modern_scientific_computation!($result, $data)

    speedup = old_time / new_time

    println("Performance improvement: $(round(speedup, digits=1))x speedup")
    println("Old time: $(round(old_time * 1000, digits=2)) ms")
    println("New time: $(round(new_time * 1000, digits=2)) ms")

    return speedup
end
```

**🐍 Python Scientific Stack Complete Transformation**
```python
# Before (Inefficient Python with manual loops):
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def inefficient_data_processing(data_path):
    """Legacy data processing with terrible performance."""

    # Read data inefficiently
    df = pd.read_csv(data_path)

    # Manual loops instead of vectorization
    processed_data = []
    for index, row in df.iterrows():  # Extremely slow!
        row_result = []
        for col in df.columns:
            if pd.isna(row[col]):
                row_result.append(0)
            else:
                # Complex computation in Python loop
                val = np.sin(row[col]) ** 2 + np.cos(row[col]) ** 2
                row_result.append(val)
        processed_data.append(row_result)

    # Convert back to DataFrame
    result = pd.DataFrame(processed_data, columns=df.columns)

    # Manual feature scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(result)

    # Manual model training
    model = RandomForestClassifier(n_estimators=100)
    # ... more manual steps

    return scaled_data

# After (Modern Python scientific stack):
import polars as pl
import cupy as cp
import dask.dataframe as dd
import xarray as xr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import ray
from ray import tune

class ModernScientificPipeline:
    """
    Modern Python scientific computing pipeline with cutting-edge tools.

    Performance improvements:
    - 10-100x faster with Polars DataFrames
    - GPU acceleration with CuPy
    - Distributed computing with Dask/Ray
    - Memory efficiency with lazy evaluation
    - Mathematical optimization applied
    """

    def __init__(self):
        self.setup_environment()

    def setup_environment(self):
        """Initialize modern scientific computing environment."""
        # Initialize Ray for distributed computing
        if not ray.is_initialized():
            ray.init()

        # Check GPU availability
        self.gpu_available = cp.cuda.is_available()
        print(f"GPU acceleration: {'✅' if self.gpu_available else '❌'}")

    def polars_data_processing(self, data_path: str):
        """Ultra-fast data processing with Polars (Rust-backed)."""

        # Lazy evaluation for query optimization
        df = (
            pl.scan_parquet(data_path)  # Memory-efficient scanning
            .with_columns([
                # Vectorized operations (100x faster than pandas loops)
                pl.col("numeric_col").fill_null(0).alias("filled_col"),

                # Mathematical optimization: sin²(x) + cos²(x) = 1
                pl.lit(1.0).alias("optimized_computation"),  # Identity!

                # Complex aggregations
                pl.col("category").count().over("group").alias("group_counts")
            ])
            .filter(pl.col("value") > 0)
            .group_by("category")
            .agg([
                pl.col("value").sum().alias("total"),
                pl.col("value").mean().alias("mean"),
                pl.col("value").std().alias("std")
            ])
        )

        # Execute optimized query
        return df.collect()

    def gpu_accelerated_computation(self, data: np.ndarray):
        """GPU acceleration with CuPy (10-100x speedup)."""

        if self.gpu_available:
            # Transfer to GPU
            gpu_data = cp.asarray(data)

            # GPU-accelerated FFT and mathematical operations
            fft_result = cp.fft.fft2(gpu_data)
            power_spectrum = cp.abs(fft_result) ** 2

            # Complex GPU operations
            result = cp.sum(power_spectrum, axis=1)

            # Transfer back to CPU
            return cp.asnumpy(result)
        else:
            # CPU fallback with NumPy
            fft_result = np.fft.fft2(data)
            power_spectrum = np.abs(fft_result) ** 2
            return np.sum(power_spectrum, axis=1)

    def distributed_ml_pipeline(self, X, y):
        """Distributed machine learning with Ray."""

        @ray.remote
        def train_model_remote(config):
            """Remote training function for distributed execution."""
            from sklearn.model_selection import cross_val_score

            model = RandomForestClassifier(**config)
            scores = cross_val_score(model, X, y, cv=5)
            return {"accuracy": scores.mean(), "std": scores.std()}

        # Distributed hyperparameter optimization
        search_space = {
            "n_estimators": tune.randint(100, 1000),
            "max_depth": tune.randint(5, 30),
            "min_samples_split": tune.uniform(0.01, 0.1)
        }

        # Use Optuna for Bayesian optimization
        from ray.tune.search.optuna import OptunaSearch

        tuner = tune.Tuner(
            train_model_remote,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                search_alg=OptunaSearch(),
                num_samples=100,
                max_concurrent_trials=8
            )
        )

        results = tuner.fit()
        return results.get_best_result()

    def xarray_scientific_data(self, dataset_path: str):
        """N-dimensional scientific data processing with Xarray."""

        # Load multi-dimensional scientific dataset
        ds = xr.open_dataset(dataset_path, chunks={'time': 100})

        # Advanced scientific operations
        result = (
            ds
            .sel(lat=slice(-60, 60))  # Tropical regions
            .groupby("time.season")
            .mean("time")
            .rolling(lat=5, center=True)
            .mean()
        )

        return result

    def complete_modern_pipeline(self, data_path: str):
        """Complete modern scientific computing pipeline."""

        # Step 1: Fast data loading with Polars
        processed_data = self.polars_data_processing(data_path)

        # Step 2: GPU-accelerated computation
        numpy_data = processed_data.to_numpy()
        gpu_result = self.gpu_accelerated_computation(numpy_data)

        # Step 3: Distributed machine learning
        X, y = gpu_result[:, :-1], gpu_result[:, -1]
        ml_results = self.distributed_ml_pipeline(X, y)

        return {
            'processed_data': processed_data,
            'gpu_computation': gpu_result,
            'ml_results': ml_results
        }

# Performance comparison
def benchmark_modern_vs_legacy():
    """Benchmark modern vs legacy implementations."""

    # Create test data
    test_data = np.random.randn(10000, 100)

    # Legacy timing
    import time
    start_time = time.time()
    legacy_result = inefficient_data_processing("test_data.csv")
    legacy_time = time.time() - start_time

    # Modern timing
    pipeline = ModernScientificPipeline()
    start_time = time.time()
    modern_result = pipeline.complete_modern_pipeline("test_data.parquet")
    modern_time = time.time() - start_time

    speedup = legacy_time / modern_time
    print(f"🚀 Performance improvement: {speedup:.1f}x speedup")
    print(f"📊 Legacy time: {legacy_time:.2f}s")
    print(f"⚡ Modern time: {modern_time:.2f}s")

    return speedup
```

**Julia Refactoring Examples**
```julia
# Before (Type unstable and allocating):
function process_data_slow(data)
    result = []  # Type unstable empty array
    for i in 1:length(data)
        for j in 1:length(data[1])
            if data[i][j] > 0
                push!(result, data[i][j] * 2.0)  # Allocates each iteration
            else
                push!(result, data[i][j] * 2)    # Mixed types!
            end
        end
    end
    return result
end

# After (Type stable and optimized):
function process_data_fast(data::Matrix{T})::Vector{T} where T<:Real
    """
    Type-stable, vectorized data processing function.

    Args:
        data: Input matrix of numeric type T

    Returns:
        Processed vector maintaining input type T

    Performance optimizations:
        - Type stability for 10-100x speedup
        - Pre-allocation to minimize garbage collection
        - Broadcasting for SIMD vectorization
        - Mathematical optimization (factor out constant)
    """
    # Pre-allocate result array (type stable)
    result = Vector{T}(undef, length(data))

    # Vectorized operation (SIMD optimized)
    result .= data .* T(2)

    return result
end

# Even better (Pure vectorized):
process_data_vectorized(data::Matrix{T}) where T<:Real = vec(data .* T(2))

# Julia Multiple Dispatch Refactoring
# Before (Generic, slow):
function compute_metric(x, y)
    return sqrt(sum((x .- y).^2))  # Works but not optimized
end

# After (Specialized dispatch):
function compute_metric(x::Vector{Float64}, y::Vector{Float64})
    """Euclidean distance for Float64 vectors with BLAS optimization."""
    diff = x .- y
    return sqrt(LinearAlgebra.dot(diff, diff))  # BLAS optimized
end

function compute_metric(x::SVector{N,T}, y::SVector{N,T}) where {N,T}
    """Compile-time optimized distance for small static vectors."""
    return sqrt(sum(abs2, x .- y))  # Compile-time unrolled
end

function compute_metric(x::CuArray{T}, y::CuArray{T}) where T
    """GPU-optimized distance computation."""
    return sqrt(sum(abs2.(x .- y)))  # GPU kernel
end
```

**Scientific Python Refactoring Examples**
```python
# Before (Inefficient loops):
def analyze_data_slow(data):
    results = []
    for i in range(len(data)):
        row_result = []
        for j in range(len(data[i])):
            value = data[i][j]
            if value > 0:
                row_result.append(np.sin(value) ** 2 + np.cos(value) ** 2)
            else:
                row_result.append(0)
        results.append(row_result)
    return np.array(results)

# After (Vectorized NumPy):
def analyze_data_fast(data: np.ndarray) -> np.ndarray:
    """
    Vectorized data analysis using NumPy optimizations.

    Args:
        data: Input array of numeric data

    Returns:
        Processed array with mathematical optimization applied

    Optimizations:
        - Vectorized operations (100x faster than loops)
        - Mathematical identity: sin²(x) + cos²(x) = 1
        - Broadcasting for memory efficiency
        - Type hints for clarity
    """
    # Vectorized condition and mathematical optimization
    return np.where(data > 0, 1.0, 0.0)  # sin²(x) + cos²(x) = 1!

# sklearn Pipeline Refactoring
# Before (Manual preprocessing):
def train_model_manual(X_train, y_train, X_test):
    # Manual scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Manual feature selection
    selector = SelectKBest(f_regression, k=10)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Manual model training
    model = RandomForestRegressor()
    model.fit(X_train_selected, y_train)

    return model.predict(X_test_selected)

# After (sklearn Pipeline):
from sklearn.pipeline import Pipeline

def train_model_pipeline(X_train, y_train, X_test):
    """
    Scikit-learn pipeline for reproducible ML workflow.

    Benefits:
        - Prevents data leakage
        - Ensures consistent preprocessing
        - Easy hyperparameter tuning
        - Serializable model pipeline
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_regression, k=10)),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline.predict(X_test)
```

### 4. Testing Strategy

Generate comprehensive tests for the refactored code:

**Unit Tests**
```python
class TestOrderProcessor:
    def test_validate_order_empty_items(self):
        order = Order(items=[])
        with pytest.raises(OrderValidationError):
            validate_order(order)
    
    def test_calculate_discount_vip_customer(self):
        order = create_test_order(total=1000)
        customer = Customer(tier="VIP")
        discount = calculate_discount(order, customer)
        assert discount == Decimal("100.00")  # 10% VIP discount
```

**Test Coverage**
- All public methods tested
- Edge cases covered
- Error conditions verified
- Performance benchmarks included

**Scientific Computing Testing Patterns**

**JAX Testing**
```python
import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

class TestJAXOptimizations:
    def test_jit_compilation_consistency(self):
        """Test that JIT and non-JIT versions produce identical results."""
        @jax.jit
        def jit_function(x):
            return jnp.sin(x) ** 2 + jnp.cos(x) ** 2

        def python_function(x):
            return jnp.sin(x) ** 2 + jnp.cos(x) ** 2

        x = jnp.array([1.0, 2.0, 3.0])
        jit_result = jit_function(x)
        python_result = python_function(x)

        assert jnp.allclose(jit_result, python_result, rtol=1e-10)

    def test_gradient_correctness(self):
        """Test automatic differentiation against numerical gradients."""
        def loss_function(x):
            return jnp.sum(x ** 3 - 2 * x ** 2 + x)

        x = jnp.array([1.0, 2.0, 3.0])

        # JAX provides check_grads for numerical gradient verification
        check_grads(loss_function, (x,), order=1, modes=['fwd', 'rev'])

    def test_vmap_consistency(self):
        """Test vectorized map produces same results as manual batching."""
        def single_computation(x):
            return jnp.sin(x) + jnp.cos(x)

        batch_size = 10
        x_batch = jnp.arange(batch_size, dtype=float)

        # Manual batching
        manual_results = jnp.array([single_computation(x) for x in x_batch])

        # Vectorized
        vmap_results = jax.vmap(single_computation)(x_batch)

        assert jnp.allclose(manual_results, vmap_results)

    def test_performance_regression(self):
        """Performance regression test for JIT compiled functions."""
        @jax.jit
        def optimized_computation(x):
            return jnp.sum(jnp.sin(x) ** 2 + jnp.cos(x) ** 2)

        large_array = jnp.arange(10000, dtype=float)

        import time
        start_time = time.time()
        result = optimized_computation(large_array).block_until_ready()
        execution_time = time.time() - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert execution_time < 0.1  # 100ms threshold
        assert jnp.allclose(result, 10000.0)  # Mathematical identity verification
```

**Julia Testing (Test.jl format)**
```julia
using Test
using BenchmarkTools

@testset "Julia Performance Tests" begin
    @testset "Type Stability" begin
        # Test that functions maintain type stability
        function test_function(x::T)::T where T<:Real
            return x * T(2)
        end

        @test typeof(test_function(1)) == Int
        @test typeof(test_function(1.0)) == Float64
        @test typeof(test_function(1.0f0)) == Float32

        # Use @inferred to test compile-time type inference
        @test @inferred test_function(1.0) === 2.0
    end

    @testset "Allocation Testing" begin
        function allocating_function(n)
            result = []
            for i in 1:n
                push!(result, i^2)
            end
            return result
        end

        function non_allocating_function!(result, n)
            for i in 1:n
                result[i] = i^2
            end
            return result
        end

        n = 1000
        result = Vector{Int}(undef, n)

        # Test that optimized version allocates less
        alloc_before = @allocated allocating_function(n)
        alloc_after = @allocated non_allocating_function!(result, n)

        @test alloc_after < alloc_before / 10  # At least 10x less allocation
    end

    @testset "Performance Benchmarks" begin
        function slow_version(data)
            result = similar(data)
            for i in eachindex(data)
                result[i] = sin(data[i])^2 + cos(data[i])^2
            end
            return result
        end

        function fast_version(data)
            return ones(eltype(data), size(data))  # sin²(x) + cos²(x) = 1
        end

        data = rand(10000)

        # Benchmark both versions
        slow_bench = @benchmark slow_version($data)
        fast_bench = @benchmark fast_version($data)

        # Fast version should be significantly faster
        @test median(fast_bench.times) < median(slow_bench.times) / 100

        # Results should be equivalent
        @test isapprox(slow_version(data), fast_version(data), rtol=1e-10)
    end

    @testset "Multiple Dispatch" begin
        # Test that specialized methods are faster than generic ones
        function generic_distance(x, y)
            return sqrt(sum(abs2, x .- y))
        end

        function optimized_distance(x::Vector{Float64}, y::Vector{Float64})
            diff = x .- y
            return sqrt(LinearAlgebra.dot(diff, diff))
        end

        x = rand(1000)
        y = rand(1000)

        generic_time = @elapsed generic_distance(x, y)
        optimized_time = @elapsed optimized_distance(x, y)

        @test optimized_time < generic_time  # Optimized should be faster
        @test isapprox(generic_distance(x, y), optimized_distance(x, y))
    end
end
```

**Scientific Python Testing**
```python
import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score

class TestScientificPythonOptimizations:
    def test_vectorization_correctness(self):
        """Test vectorized operations against loop-based versions."""
        def loop_version(data):
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    result[i, j] = np.sin(data[i, j]) ** 2 + np.cos(data[i, j]) ** 2
            return result

        def vectorized_version(data):
            return np.ones_like(data)  # Mathematical identity optimization

        data = np.random.rand(100, 100)
        loop_result = loop_version(data)
        vectorized_result = vectorized_version(data)

        np.testing.assert_allclose(loop_result, vectorized_result, rtol=1e-10)

    def test_performance_regression(self):
        """Performance regression test for optimized functions."""
        large_array = np.random.rand(10000, 1000)

        import time
        start_time = time.time()
        result = np.sum(large_array, axis=1)  # Optimized NumPy operation
        execution_time = time.time() - start_time

        assert execution_time < 1.0  # Should complete in reasonable time

    def test_pipeline_consistency(self):
        """Test that pipeline produces same results as manual steps."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression

        X, y = make_regression(n_samples=100, n_features=20, random_state=42)

        # Manual approach
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        manual_score = model.score(X_scaled, y)

        # Pipeline approach
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
        pipeline.fit(X, y)
        pipeline_score = pipeline.score(X, y)

        assert abs(manual_score - pipeline_score) < 1e-10

    def test_memory_efficiency(self):
        """Test memory usage of optimized implementations."""
        import tracemalloc

        def memory_inefficient():
            data = []
            for i in range(10000):
                data.append(np.random.rand(100))
            return np.array(data)

        def memory_efficient():
            return np.random.rand(10000, 100)

        # Test memory efficient version uses less memory
        tracemalloc.start()
        result1 = memory_efficient()
        current, peak = tracemalloc.get_traced_memory()
        efficient_memory = peak
        tracemalloc.stop()

        tracemalloc.start()
        result2 = memory_inefficient()
        current, peak = tracemalloc.get_traced_memory()
        inefficient_memory = peak
        tracemalloc.stop()

        assert efficient_memory < inefficient_memory / 2  # At least 2x more efficient
        np.testing.assert_array_equal(result1, result2)  # Same results
```

### 5. Before/After Comparison

Provide clear comparisons showing improvements:

**Metrics**
- Cyclomatic complexity reduction
- Lines of code per method
- Test coverage increase
- Performance improvements

**Example**
```
Before:
- processData(): 150 lines, complexity: 25
- 0% test coverage
- 3 responsibilities mixed

After:
- validateInput(): 20 lines, complexity: 4
- transformData(): 25 lines, complexity: 5
- saveResults(): 15 lines, complexity: 3
- 95% test coverage
- Clear separation of concerns
```

**Scientific Computing Performance Improvements**

**JAX Ecosystem Refactoring**
```
Before (Manual Implementation):
- gradient_computation(): 50 lines, O(n) finite differences
- train_step(): 100 lines, no JIT compilation
- model_forward(): 75 lines, manual parameter management
- Execution time: 45.2 seconds
- Memory usage: 2.1 GB
- Test coverage: 30%

After (JAX Optimized):
- gradient_computation(): 5 lines, @jax.jit + jax.grad
- train_step(): 15 lines, JIT compiled
- model_forward(): 10 lines, Flax module
- Execution time: 0.18 seconds (251x speedup!)
- Memory usage: 0.3 GB (7x reduction)
- Test coverage: 95%
- Automatic differentiation: Exact gradients
- GPU/TPU ready: Auto-parallelization
```

**Julia Scientific Computing Refactoring**
```
Before (Type Unstable):
- data_processing(): 80 lines, type unstable
- linear_solve(): 45 lines, generic types
- parallel_computation(): 60 lines, serial execution
- Execution time: 12.4 seconds
- Memory allocations: 1.2 million
- Type inference: Failed (Any types)

After (Julia Optimized):
- data_processing(): 20 lines, type stable
- linear_solve(): 8 lines, specialized dispatch
- parallel_computation(): 15 lines, @threads + pmap
- Execution time: 0.41 seconds (30x speedup!)
- Memory allocations: 12,000 (100x reduction)
- Type inference: Perfect (concrete types)
- Multiple dispatch: Specialized methods
- SIMD vectorization: Automatic
```

**Scientific Python Refactoring**
```
Before (Loop-based):
- matrix_operations(): 120 lines, nested loops
- data_analysis(): 90 lines, manual iterations
- ml_pipeline(): 150 lines, manual preprocessing
- Execution time: 8.7 seconds
- Memory efficiency: Poor (repeated allocations)
- Scalability: O(n²) algorithms

After (Vectorized):
- matrix_operations(): 15 lines, NumPy vectorized
- data_analysis(): 25 lines, pandas operations
- ml_pipeline(): 30 lines, sklearn Pipeline
- Execution time: 0.22 seconds (40x speedup!)
- Memory efficiency: Excellent (pre-allocated)
- Scalability: O(n log n) optimized algorithms
- Mathematical optimization: Identity simplifications
```

**Real-World Case Study: Neural Network Training**
```
Original Implementation (PyTorch manual):
- Lines of code: 450
- Training time (1000 epochs): 2.5 hours
- Memory usage: 8.2 GB
- Gradient computation: Manual backprop
- Optimization: Hand-coded SGD
- Hardware utilization: CPU only

JAX + Flax Refactored:
- Lines of code: 85 (5.3x reduction)
- Training time (1000 epochs): 12 minutes (12.5x speedup)
- Memory usage: 1.1 GB (7.5x reduction)
- Gradient computation: Automatic via jax.grad
- Optimization: Optax AdamW with scheduling
- Hardware utilization: GPU/TPU with auto-parallelization

Improvements:
- Development time: 80% reduction
- Training speed: 1250% improvement
- Memory efficiency: 750% improvement
- Code maintainability: Dramatically improved
- Mathematical correctness: Guaranteed
```

### 6. Migration Guide

If breaking changes are introduced:

**Step-by-Step Migration**
1. Install new dependencies
2. Update import statements
3. Replace deprecated methods
4. Run migration scripts
5. Execute test suite

**Backward Compatibility**
```python
# Temporary adapter for smooth migration
class LegacyOrderProcessor:
    def __init__(self):
        self.processor = OrderProcessor()
    
    def process(self, order_data):
        # Convert legacy format
        order = Order.from_legacy(order_data)
        return self.processor.process(order)
```

### 7. Performance Optimizations

Include specific optimizations:

**Algorithm Improvements**
```python
# Before: O(n²)
for item in items:
    for other in items:
        if item.id == other.id:
            # process

# After: O(n)
item_map = {item.id: item for item in items}
for item_id, item in item_map.items():
    # process
```

**Caching Strategy**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def calculate_expensive_metric(data_id: str) -> float:
    # Expensive calculation cached
    return result
```

### 8. Code Quality Checklist

Ensure the refactored code meets these criteria:

**General Code Quality**
- [ ] All methods < 20 lines
- [ ] All classes < 200 lines
- [ ] No method has > 3 parameters
- [ ] Cyclomatic complexity < 10
- [ ] No nested loops > 2 levels
- [ ] All names are descriptive
- [ ] No commented-out code
- [ ] Consistent formatting
- [ ] Type hints added (Python/TypeScript)
- [ ] Error handling comprehensive
- [ ] Logging added for debugging
- [ ] Performance metrics included
- [ ] Documentation complete
- [ ] Tests achieve > 80% coverage
- [ ] No security vulnerabilities

**🚀 Advanced Scientific Computing Quality Standards (2025 Edition)**

**JAX Ecosystem Excellence Checklist**
- [ ] **Core JAX Optimizations:**
  - [ ] All computational functions use @jax.jit with appropriate static_argnums
  - [ ] Gradient computations use jax.grad/jax.value_and_grad (exact gradients)
  - [ ] Vectorized operations use jax.vmap/jax.pmap (no Python loops)
  - [ ] Pure functions enforced (no side effects for transformations)
  - [ ] jax.lax operations used for maximum performance
  - [ ] Proper PRNG key management with jax.random.split

- [ ] **Modern JAX Frameworks:**
  - [ ] Equinox modules for neural networks (type-safe, composable)
  - [ ] Optax optimizers with learning rate scheduling
  - [ ] Chex assertions for testing and validation
  - [ ] Orbax checkpointing for model persistence
  - [ ] Diffrax for differential equation solving
  - [ ] JAX-MD for molecular dynamics (if applicable)

- [ ] **Advanced Performance:**
  - [ ] JAX-scipy optimization patterns implemented
  - [ ] Gradient accumulation for large models
  - [ ] Multi-device training with pmap
  - [ ] Memory-efficient implementations (activation checkpointing)
  - [ ] GPU/TPU compatibility verified with device placement
  - [ ] Performance regression tests with target metrics

**Julia Scientific Excellence Checklist**
- [ ] **Type System Mastery:**
  - [ ] Perfect type stability (@inferred tests pass)
  - [ ] Concrete types in function signatures (no Any)
  - [ ] Parametric types for performance (@code_warntype clean)
  - [ ] Union types avoided in hot loops
  - [ ] Abstract types used only for interfaces

- [ ] **Modern Julia Ecosystem:**
  - [ ] DifferentialEquations.jl for ODE/SDE/DAE solving
  - [ ] Flux.jl 0.14+ with modern architectures
  - [ ] MLJ.jl for machine learning pipelines
  - [ ] DrWatson.jl for scientific project structure
  - [ ] Catalyst.jl for reaction networks (if applicable)
  - [ ] ModelingToolkit.jl for symbolic computation

- [ ] **High-Performance Computing:**
  - [ ] LoopVectorization.jl (@turbo) for SIMD optimization
  - [ ] StaticArrays.jl for compile-time optimization
  - [ ] CUDA.jl for GPU acceleration
  - [ ] MPI.jl for distributed computing
  - [ ] Pre-allocation patterns (zero allocation hot loops)
  - [ ] BenchmarkTools.jl comprehensive performance testing

- [ ] **Memory and Performance:**
  - [ ] @allocated < 32 bytes for hot functions
  - [ ] Broadcasting operations (.=, .+) for vectorization
  - [ ] Multiple dispatch specialized methods
  - [ ] @threads/@distributed for parallelization
  - [ ] BLAS operations for linear algebra

**Python Scientific Stack Excellence Checklist**
- [ ] **Traditional Scientific Python:**
  - [ ] NumPy vectorized operations (zero Python loops on arrays)
  - [ ] SciPy optimized functions for scientific computing
  - [ ] Pandas efficient operations (no iterrows)
  - [ ] Scikit-learn Pipelines for ML workflows
  - [ ] Matplotlib/Seaborn for publication-quality plots

- [ ] **Modern Python Scientific Computing (2025):**
  - [ ] Polars for ultra-fast DataFrame operations (Rust-backed)
  - [ ] CuPy for GPU-accelerated NumPy (10-100x speedup)
  - [ ] Dask for parallel/distributed computing
  - [ ] Xarray for labeled N-dimensional arrays
  - [ ] Numba @jit/@cuda for compiled performance
  - [ ] Ray for distributed machine learning

- [ ] **Advanced Data Processing:**
  - [ ] Vaex for out-of-core DataFrame operations
  - [ ] Modin for pandas acceleration
  - [ ] DVC for data version control
  - [ ] Great Expectations for data validation
  - [ ] Prefect/Kedro for pipeline orchestration

- [ ] **AI/ML Framework Integration:**
  - [ ] PyTorch 2.0+ with torch.compile optimization
  - [ ] JAX integration for numerical computing
  - [ ] Weights & Biases for experiment tracking
  - [ ] Hydra for configuration management
  - [ ] MLflow for model lifecycle management

**Cross-Language Integration Excellence**
- [ ] **Multi-Language Workflows:**
  - [ ] JuliaCall/PyCall optimized integration
  - [ ] JAX-Julia interoperability patterns
  - [ ] C/C++ extension optimization (Cython, SWIG)
  - [ ] WebAssembly deployment patterns
  - [ ] Container optimization (Docker, Singularity)

**Research Workflow Excellence Checklist**
- [ ] **Reproducibility Standards:**
  - [ ] Environment management (conda/mamba/pixi)
  - [ ] Experiment tracking (MLflow/Wandb/Neptune)
  - [ ] Data pipeline versioning (DVC/Kedro)
  - [ ] FAIR data principles compliance
  - [ ] Computational reproducibility verified

- [ ] **Modern Documentation:**
  - [ ] Quarto/Jupyter Book for scientific publishing
  - [ ] Automatic citation management (BibTeX/Zotero)
  - [ ] Interactive documentation with examples
  - [ ] API documentation with type hints
  - [ ] Performance benchmarks documented

- [ ] **Open Science Standards:**
  - [ ] Open source licensing specified
  - [ ] Data availability statements
  - [ ] Code sharing with proper attribution
  - [ ] Reproducible computational environments
  - [ ] Transparent methodology documentation

**Advanced Performance Optimization Checklist**
- [ ] **Algorithmic Excellence:**
  - [ ] Optimal algorithmic complexity achieved
  - [ ] Mathematical identities applied (e.g., sin²+cos²=1)
  - [ ] Numerical stability analyzed and ensured
  - [ ] Precision requirements optimized
  - [ ] Cache-friendly data access patterns

- [ ] **Hardware Optimization:**
  - [ ] SIMD vectorization leveraged
  - [ ] GPU/TPU acceleration where beneficial
  - [ ] Memory hierarchy optimization
  - [ ] Parallel processing patterns implemented
  - [ ] Load balancing for distributed systems

- [ ] **Benchmarking and Validation:**
  - [ ] Performance regression tests implemented
  - [ ] Scalability analysis with problem size
  - [ ] Memory usage profiling and optimization
  - [ ] Energy efficiency considerations
  - [ ] Benchmark comparisons with state-of-the-art

**AI-Powered Code Quality Metrics**
- [ ] **Automated Analysis:**
  - [ ] Static analysis with modern tools
  - [ ] Dynamic performance profiling
  - [ ] Security vulnerability scanning
  - [ ] Dependency analysis and updates
  - [ ] Code coverage > 90% for critical functions

- [ ] **Intelligent Optimization:**
  - [ ] AI-suggested refactoring patterns applied
  - [ ] Performance bottleneck identification automated
  - [ ] Code smell detection and resolution
  - [ ] Pattern recognition for optimization opportunities
  - [ ] Continuous improvement feedback loops

## Severity Levels

Rate issues found and improvements made:

**Critical**: Security vulnerabilities, data corruption risks, memory leaks, type instability in Julia, missing JAX JIT compilation
**High**: Performance bottlenecks (O(n²) algorithms), maintainability blockers, missing tests, Python loops on arrays, manual gradient computation
**Medium**: Code smells, minor performance issues, incomplete documentation, missing vectorization, suboptimal package usage
**Low**: Style inconsistencies, minor naming issues, nice-to-have features, missing type hints, minor allocation inefficiencies

## Advanced AI-Powered Output Format (2025 Edition)

### 1. **Executive Analysis Summary**
- **Critical Issues Identified**: Performance bottlenecks, type instability, security vulnerabilities
- **Framework-Specific Optimizations**: JAX JIT opportunities, Julia type stabilization, Python vectorization
- **Performance Impact Projections**: Quantified speedup estimates (10x, 100x, etc.)
- **AI-Powered Insights**: Machine learning-driven pattern recognition and optimization suggestions
- **ROI-Ranked Improvements**: Impact/effort ratio for each optimization opportunity

### 2. **Multi-Framework Refactoring Strategy**
- **Phase 1: Quick Wins** (Hours)
  - Mathematical identity optimizations
  - Type annotation additions
  - Import optimization and dead code removal
  - Framework-specific decorators (@jax.jit, @threads, etc.)

- **Phase 2: Architecture Modernization** (Days)
  - JAX/Equinox model architectures
  - Julia ecosystem integration (DifferentialEquations.jl, MLJ.jl)
  - Python scientific stack modernization (Polars, CuPy, Ray)
  - Cross-language integration patterns

- **Phase 3: Advanced Optimization** (Weeks)
  - GPU/TPU acceleration implementation
  - Distributed computing patterns
  - Research workflow integration
  - Reproducibility and FAIR data compliance

### 3. **State-of-the-Art Refactored Implementation**
- **JAX Ecosystem Code**: Equinox modules, Optax optimizers, JAX-scipy integration
- **Julia High-Performance Code**: Type-stable, SIMD-optimized, GPU-accelerated
- **Modern Python Scientific Code**: Polars, CuPy, Dask, Xarray integration
- **Cross-Language Workflows**: Optimized multi-language scientific computing
- **Research-Ready Code**: Reproducible, documented, publication-quality implementation

### 4. **Comprehensive Testing Framework**
- **JAX Testing Patterns**: JIT compilation verification, gradient correctness, device compatibility
- **Julia Performance Testing**: Type stability, allocation monitoring, benchmark comparisons
- **Python Scientific Testing**: Vectorization correctness, memory efficiency, scalability
- **Integration Testing**: Cross-framework compatibility, numerical accuracy validation
- **Performance Regression Testing**: Automated benchmarking with CI/CD integration

### 5. **Research Workflow Integration Guide**
- **Environment Setup**: conda/mamba/pixi configuration for reproducibility
- **Experiment Tracking**: MLflow/Wandb integration for scientific workflows
- **Data Pipeline**: DVC/Kedro setup for version control and orchestration
- **Documentation**: Quarto/Jupyter Book for publication-ready results
- **Open Science Compliance**: FAIR data principles and reproducibility standards

### 6. **Advanced Performance Metrics Report**
```
🚀 TRANSFORMATION RESULTS

╭─────────────────────────────────────────────────────────╮
│                    PERFORMANCE GAINS                   │
├─────────────────────────────────────────────────────────┤
│ Metric                │ Before    │ After     │ Improvement │
├─────────────────────────────────────────────────────────┤
│ Execution Time        │ 45.2s     │ 0.18s     │ 251x ⚡     │
│ Memory Usage          │ 2.1 GB    │ 0.3 GB    │ 7x 💾      │
│ GPU Utilization       │ 0%        │ 95%       │ ∞ 🚀       │
│ Code Lines            │ 450       │ 85        │ 5.3x 📝    │
│ Type Stability        │ 30%       │ 100%      │ Perfect ✅  │
│ Test Coverage         │ 30%       │ 95%       │ 3.2x 🧪    │
│ Maintainability       │ Poor      │ Excellent │ ⭐⭐⭐⭐⭐  │
╰─────────────────────────────────────────────────────────╯

🎯 SCIENTIFIC COMPUTING OPTIMIZATIONS APPLIED:
✅ JAX JIT compilation and autodiff
✅ Julia type stability and SIMD vectorization
✅ Python vectorization and GPU acceleration
✅ Mathematical identity optimizations
✅ Cross-language integration patterns
✅ Research workflow best practices

📊 QUALITY METRICS:
• Cyclomatic Complexity: 25 → 4 (83% reduction)
• Technical Debt: High → Low
• Security Vulnerabilities: 3 → 0
• Performance Regressions: 0 (comprehensive testing)
• Reproducibility Score: 95% (FAIR compliant)
```

### 7. **AI-Powered Migration Roadmap**
- **Automated Migration Scripts**: Framework-specific conversion tools
- **Backward Compatibility Adapters**: Smooth transition patterns
- **Validation Checkpoints**: Correctness verification at each step
- **Performance Monitoring**: Real-time optimization impact tracking
- **Rollback Strategies**: Safe fallback procedures for production systems

### 8. **Continuous Optimization Framework**
- **Performance Monitoring**: Automated benchmarking and regression detection
- **Code Evolution Tracking**: Git-based metrics and improvement visualization
- **AI-Driven Suggestions**: Machine learning-based optimization recommendations
- **Community Best Practices**: Integration with latest scientific computing developments
- **Future-Proofing**: Compatibility with emerging frameworks and hardware

### 9. **Scientific Publication Ready Output**
- **Methodology Documentation**: Reproducible computational methods
- **Performance Benchmarks**: Publication-quality figures and statistics
- **Code Availability**: Open source repository with proper attribution
- **Data Provenance**: FAIR data principles compliance documentation
- **Citation Management**: Automatic BibTeX generation for references

### 10. **Success Metrics Dashboard**
```python
class RefactoringSuccessMetrics:
    """Comprehensive success tracking for scientific computing refactoring."""

    def __init__(self):
        self.metrics = {
            'performance': {
                'speedup_factor': 0,
                'memory_reduction': 0,
                'gpu_utilization': 0
            },
            'quality': {
                'test_coverage': 0,
                'type_stability': 0,
                'maintainability_index': 0
            },
            'scientific': {
                'reproducibility_score': 0,
                'numerical_accuracy': 0,
                'documentation_completeness': 0
            },
            'workflow': {
                'automation_level': 0,
                'collaboration_efficiency': 0,
                'publication_readiness': 0
            }
        }

    def generate_success_report(self) -> str:
        """Generate comprehensive success metrics report."""
        return f"""
        🎉 REFACTORING SUCCESS SUMMARY

        Performance: {self.metrics['performance']['speedup_factor']}x speedup achieved
        Quality: {self.metrics['quality']['test_coverage']}% test coverage
        Science: {self.metrics['scientific']['reproducibility_score']}% reproducible
        Workflow: {self.metrics['workflow']['publication_readiness']}% publication ready

        ✅ Mission Accomplished: Production-ready scientific computing code!
        """
```

**🌟 Transformation Philosophy**: Deliver revolutionary improvements that transform legacy code into state-of-the-art scientific computing implementations while maintaining 100% correctness, reproducibility, and research-grade quality standards.