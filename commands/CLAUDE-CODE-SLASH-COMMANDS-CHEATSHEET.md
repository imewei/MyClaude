# Claude Code Slash Commands - Complete Cheatsheet

A comprehensive guide to all available slash commands in Claude Code for scientific computing, AI development, and software engineering.

## Table of Contents
- [🧠 AI Analysis & Thinking](#-ai-analysis--thinking)
- [🔧 Code Quality & Analysis](#-code-quality--analysis)
- [🧪 Testing & Debugging](#-testing--debugging)
- [📝 Git & Project Management](#-git--project-management)
- [🔍 Verification & Quality Control](#-verification--quality-control)
- [🐍 Python Optimization](#-python-optimization)
- [⚡ JAX Ecosystem](#-jax-ecosystem)
- [🔬 Julia Scientific Computing](#-julia-scientific-computing)
- [📚 Documentation](#-documentation)
- [🏗️ Common Flags](#️-common-flags)

---

## 🧠 AI Analysis & Thinking

### `/think-ultra [problem] [options]`
Revolutionary quantum-level analytical thinking engine for complex problems.

**Options:**
- `--depth=quantum|ultra|comprehensive` - Analysis depth level
- `--mode=research|discovery|optimization` - Thinking mode
- `--paradigm=multi|cross|meta` - Analysis paradigm
- `--export-insights` - Export insights for reuse

**Examples:**
```bash
/think-ultra "How to optimize distributed training across 1000 GPUs"
/think-ultra "Design a quantum-classical hybrid algorithm" --depth=quantum
/think-ultra "Multi-modal AI architecture" --mode=research --export-insights
```

### `/think-harder [problem]`
Next-generation deep analytical thinking for complex technical challenges.

**Examples:**
```bash
/think-harder "Design fault-tolerant distributed system"
/think-harder "Optimize memory usage in large language models"
```

### `/reflection [options]`
Advanced AI instruction optimization with meta-cognitive analysis.

**Options:**
- `--type=comprehensive|focused|scientific` - Reflection type
- `--analysis=deep|surface|meta` - Analysis level
- `--optimize=performance|accuracy|collaboration` - Optimization focus
- `--export-patterns` - Export learned patterns

**Examples:**
```bash
/reflection --type=scientific --analysis=deep
/reflection --optimize=performance --export-patterns
```

### `/reflection-harder [options]`
Advanced AI-powered session analysis and learning capture.

**Options:**
- `--type=scientific|general|research` - Reflection type
- `--depth=comprehensive|focused|quick` - Analysis depth
- `--ai-insights` - Include AI insights
- `--export-knowledge` - Export captured knowledge

---

## 🔧 Code Quality & Analysis

### `/optimize`
Comprehensive code optimization and performance analysis for Python, Julia, JAX ecosystem.

**Examples:**
```bash
/optimize
```

### `/code_analysis [path] [options]`
Perform comprehensive code analysis with AI-powered insights.

**Options:**
- `--focus=all|security|performance|maintainability|jax` - Analysis focus
- `--language=auto|python|julia|jax` - Target language
- `--format=detailed|json|markdown` - Output format
- `--ai-insights` - Include AI-powered insights

**Examples:**
```bash
/code_analysis src/models --focus=performance --ai-insights
/code_analysis . --language=python --format=json
/code_analysis neural_network.py --focus=jax
```

### `/check-code-quality [path] [options]`
Advanced code quality analyzer for scientific computing projects.

**Options:**
- `--fix` - Auto-fix detected issues
- `--research` - Research-grade analysis
- `--gpu-analysis` - GPU optimization analysis
- `--julia` - Julia-specific analysis
- `--benchmark` - Include benchmarking
- `--ci-mode` - CI/CD integration mode

**Examples:**
```bash
/check-code-quality src/ --fix --gpu-analysis
/check-code-quality . --research --benchmark
```

### `/refactor-clean [file] [options]`
Advanced code refactoring with AI-powered analysis.

**Options:**
- `--language=auto` - Auto-detect language
- `--focus=performance|maintainability|scientific` - Refactoring focus
- `--framework=jax|julia|python` - Target framework
- `--ai-optimize` - AI-powered optimization
- `--interactive` - Interactive refactoring

**Examples:**
```bash
/refactor-clean model.py --focus=performance --framework=jax
/refactor-clean algorithm.jl --ai-optimize --interactive
```

### `/clean-codebase [path] [options]`
Intelligent codebase cleanup with duplication detection.

**Options:**
- `--dry-run` - Preview changes without applying
- `--aggressive` - More aggressive cleanup
- `--language=auto|python|julia|mixed` - Target language
- `--interactive` - Interactive cleanup
- `--backup` - Create backup before cleanup
- `--report` - Generate cleanup report

**Examples:**
```bash
/clean-codebase . --dry-run --report
/clean-codebase src/ --aggressive --backup
```

### `/clean-project [path] [options]`
Intelligent project cleanup and structure optimization.

**Options:**
- `--dry-run` - Preview changes
- `--aggressive` - Aggressive cleanup
- `--type=python|julia|mixed|auto` - Project type
- `--interactive` - Interactive mode
- `--backup` - Create backup
- `--git-integration` - Git integration
- `--report` - Generate report

---

## 🧪 Testing & Debugging

### `/generate-tests [target] [options]`
Generate comprehensive test suites with AI optimization.

**Options:**
- `--type=all|unit|integration|performance|cases|jax|scientific|gpu` - Test type
- `--framework=auto|pytest|julia|jax|flax|polars|xarray` - Test framework

**Examples:**
```bash
/generate-tests model.py --type=unit --framework=pytest
/generate-tests neural_network.jl --type=scientific --framework=julia
/generate-tests . --type=integration --framework=jax
```

### `/run-all-tests [options]`
Comprehensive test execution with intelligent failure resolution.

**Options:**
- `--scope=SCOPE` - Test scope
- `--profile` - Include profiling
- `--benchmark` - Run benchmarks
- `--scientific` - Scientific computing tests
- `--gpu` - GPU-accelerated tests
- `--parallel` - Parallel execution
- `--reproducible` - Ensure reproducibility
- `--coverage` - Coverage analysis
- `--report` - Generate test report

**Examples:**
```bash
/run-all-tests --scientific --gpu --coverage
/run-all-tests --benchmark --parallel --report
```

### `/debug [options]`
Advanced debugging engine for scientific computing.

**Options:**
- `--issue=TYPE` - Issue type
- `--gpu` - GPU debugging
- `--julia` - Julia debugging
- `--research` - Research workflow debugging
- `--jupyter` - Jupyter notebook debugging
- `--profile` - Include profiling
- `--monitor` - Real-time monitoring
- `--logs` - Log analysis
- `--auto-fix` - Auto-fix issues
- `--report` - Generate debug report

**Examples:**
```bash
/debug --gpu --profile --auto-fix
/debug --julia --research --monitor
```

### `/python-debug-prof [options]`
Profile Python/JAX code for performance bottlenecks.

**Options:**
- `--jax-profiler` - Use JAX profiler
- `--cprofile` - Use cProfile
- `--memory` - Memory profiling
- `--suggest-opts` - Suggest optimizations

**Examples:**
```bash
/python-debug-prof --jax-profiler --memory --suggest-opts
/python-debug-prof --cprofile --suggest-opts
```

---

## 📝 Git & Project Management

### `/commit [options]`
Intelligent commit engine with AI-powered message generation.

**Options:**
- `--all` - Add all changes
- `--staged` - Commit staged changes only
- `--amend` - Amend last commit
- `--interactive` - Interactive commit
- `--split` - Split large commits
- `--template=TYPE` - Use commit template
- `--ai-message` - AI-generated message
- `--validate` - Validate before commit
- `--push` - Push after commit

**Examples:**
```bash
/commit --all --ai-message --push
/commit --staged --validate
/commit --interactive --split
```

### `/fix-commit-errors [hash] [options]`
Intelligent GitHub Actions & commit error resolution.

**Options:**
- `--auto-fix` - Automatically fix errors
- `--rerun` - Rerun failed actions
- `--debug` - Debug mode
- `--max-cycles=10` - Maximum fix cycles
- `--aggressive` - Aggressive fixing
- `--emergency` - Emergency mode

**Examples:**
```bash
/fix-commit-errors abc123 --auto-fix --rerun
/fix-commit-errors --aggressive --max-cycles=5
```

### `/fix-github-issue [issue] [options]`
GitHub issue analysis and resolution engine.

**Options:**
- `--draft` - Create draft PR
- `--branch=<name>` - Create specific branch
- `--auto-fix` - Auto-fix the issue
- `--interactive` - Interactive mode
- `--emergency` - Emergency resolution

**Examples:**
```bash
/fix-github-issue 123 --auto-fix --branch=fix-issue-123
/fix-github-issue https://github.com/user/repo/issues/456 --draft
```

---

## 🔍 Verification & Quality Control

### `/recheck [options]`
Revolutionary re-examination engine with intelligent verification.

**Options:**
- `--scope=TYPE` - Verification scope
- `--interactive` - Interactive mode
- `--auto-fix` - Auto-fix issues
- `--comprehensive` - Comprehensive check
- `--requirements` - Check requirements
- `--implementation` - Check implementation
- `--integration` - Check integration
- `--experience` - User experience check
- `--report` - Generate report
- `--gaps` - Identify gaps

**Examples:**
```bash
/recheck --comprehensive --auto-fix --report
/recheck --requirements --implementation --gaps
```

### `/double-check [options]`
Intelligent verification with multi-perspective validation.

**Options:**
- `--type=TYPE` - Check type
- `--interactive` - Interactive mode
- `--comprehensive` - Comprehensive validation
- `--requirements` - Requirements check
- `--code` - Code validation
- `--docs` - Documentation check
- `--security` - Security validation
- `--report` - Generate report
- `--auto-fix` - Auto-fix issues

**Examples:**
```bash
/double-check --comprehensive --security --report
/double-check --code --docs --auto-fix
```

---

## 🐍 Python Optimization

### `/python-type-hint [options]`
Add comprehensive type hints to Python/JAX code.

**Options:**
- `--strict` - Strict type checking
- `--jax-arrays` - JAX array typing
- `--scientific` - Scientific computing types

**Examples:**
```bash
/python-type-hint --strict --jax-arrays
/python-type-hint --scientific
```

---

## ⚡ JAX Ecosystem

### Core JAX Commands

#### `/jax-init`
Initialize JAX project with essential imports and PRNG setup.

**Examples:**
```bash
/jax-init
```

#### `/jax-jit [function]`
Apply JIT compilation to JAX functions with optimization benefits.

**Examples:**
```bash
/jax-jit train_step
/jax-jit loss_function
```

#### `/jax-grad [function] [options]`
Compute gradients using JAX automatic differentiation.

**Options:**
- `--higher-order` - Higher-order derivatives
- `--value-and-grad` - Return both value and gradient

**Examples:**
```bash
/jax-grad loss_fn --value-and-grad
/jax-grad objective --higher-order
```

#### `/jax-vmap [function] [options]`
Vectorize functions over batches with axis specification.

**Options:**
- `--in-axes` - Input axis specification
- `--out-axes` - Output axis specification

**Examples:**
```bash
/jax-vmap process_batch --in-axes=0
/jax-vmap transform --in-axes=0 --out-axes=1
```

#### `/jax-pmap [function] [options]`
Parallelize across multiple devices with mesh setup.

**Options:**
- `--axis-name` - Parallel axis name
- `--devices` - Device specification

**Examples:**
```bash
/jax-pmap train_step --axis-name=batch
/jax-pmap inference --devices=gpu
```

#### `/jax-debug [options]`
Debug JAX code with specialized tools.

**Options:**
- `--disable-jit` - Disable JIT for debugging
- `--check-tracers` - Check for tracer leaks
- `--print-values` - Print intermediate values

**Examples:**
```bash
/jax-debug --disable-jit --print-values
/jax-debug --check-tracers
```

### JAX Advanced Features

#### `/jax-mixed-prec [options]`
Enable mixed precision in JAX with bfloat16.

**Options:**
- `--precision-level` - Precision level
- `--fallback-ops` - Float64 fallback operations
- `--monitoring` - Precision monitoring
- `--optimization` - Precision optimization

**Examples:**
```bash
/jax-mixed-prec --precision-level=bfloat16 --fallback-ops
/jax-mixed-prec --monitoring --optimization
```

#### `/jax-cache-opt [options]`
Implement JIT caching strategies with optimization.

**Options:**
- `--cache-policy` - Caching policy
- `--eviction-strategy` - Cache eviction strategy
- `--memory-limit` - Memory limit for cache
- `--profiling` - Cache profiling

**Examples:**
```bash
/jax-cache-opt --cache-policy=lru --memory-limit=8GB
/jax-cache-opt --eviction-strategy=fifo --profiling
```

#### `/jax-data-load [options]`
Set up data loading pipelines optimized for JAX.

**Options:**
- `--framework` - Data framework (Grain/TF)
- `--batch-size` - Batch size
- `--shuffle` - Enable shuffling
- `--prefetch` - Enable prefetching

**Examples:**
```bash
/jax-data-load --framework=grain --batch-size=32 --shuffle
/jax-data-load --framework=tf --prefetch
```

#### `/jax-sparse-jac [options]`
Compute sparse Jacobians for efficient optimization.

**Options:**
- `--sparsity-pattern` - Sparsity pattern
- `--estimation-method` - Jacobian estimation method
- `--nlsq-integration` - NLSQ integration
- `--memory-efficient` - Memory-efficient computation

**Examples:**
```bash
/jax-sparse-jac --sparsity-pattern=banded --memory-efficient
/jax-sparse-jac --estimation-method=forward --nlsq-integration
```

### JAX Machine Learning

#### `/jax-flax-model [options]`
Define neural networks using Flax Linen.

**Options:**
- `--architecture` - Model architecture
- `--layers` - Layer specification
- `--activation` - Activation functions

**Examples:**
```bash
/jax-flax-model --architecture=transformer --layers=12
/jax-flax-model --architecture=cnn --activation=gelu
```

#### `/jax-equinox-model [options]`
Define functional neural networks with Equinox.

**Options:**
- `--architecture` - Model architecture
- `--functional` - Functional programming style
- `--stateful` - Stateful components

**Examples:**
```bash
/jax-equinox-model --architecture=mlp --functional
/jax-equinox-model --architecture=rnn --stateful
```

#### `/jax-ml-train [options]`
Complete ML training loop with Flax and Optax.

**Options:**
- `--model-type` - Model type
- `--optimizer` - Optimizer choice
- `--epochs` - Number of epochs

**Examples:**
```bash
/jax-ml-train --model-type=transformer --optimizer=adamw --epochs=100
/jax-ml-train --model-type=cnn --optimizer=sgd
```

#### `/jax-optax-optimizer [options]`
Set up optimizers with Optax.

**Options:**
- `--optimizer-type` - Optimizer type
- `--learning-rate` - Learning rate
- `--schedule` - Learning rate schedule

**Examples:**
```bash
/jax-optax-optimizer --optimizer-type=adamw --learning-rate=1e-4
/jax-optax-optimizer --optimizer-type=sgd --schedule=cosine
```

### JAX Scientific Computing

#### `/jax-nlsq-fit [options]`
Nonlinear least-squares curve fitting with GPU acceleration.

**Options:**
- `--model-type` - Model type for fitting
- `--gpu-accel` - GPU acceleration
- `--chunking` - Data chunking strategy
- `--algorithm` - Fitting algorithm

**Examples:**
```bash
/jax-nlsq-fit --model-type=gaussian --gpu-accel --chunking
/jax-nlsq-fit --algorithm=levenberg-marquardt
```

#### `/jax-nlsq-large [options]`
Optimize NLSQ for massive datasets.

**Options:**
- `--dataset-size` - Dataset size
- `--memory-limit` - Memory constraint
- `--algorithm` - Algorithm choice
- `--chunking-strategy` - Chunking strategy

**Examples:**
```bash
/jax-nlsq-large --dataset-size=1TB --memory-limit=32GB --chunking-strategy=adaptive
/jax-nlsq-large --algorithm=trust-region
```

#### `/jax-numpyro-prob [options]`
Set up probabilistic models with Numpyro.

**Options:**
- `--model-type` - Probabilistic model type
- `--inference` - Inference method
- `--sampling` - Sampling strategy

**Examples:**
```bash
/jax-numpyro-prob --model-type=bayesian-nn --inference=nuts
/jax-numpyro-prob --model-type=hierarchical --sampling=hmc
```

#### `/jax-orbax-checkpoint [options]`
Handle model checkpointing with Orbax.

**Options:**
- `--save` - Save checkpoint
- `--restore` - Restore checkpoint
- `--async` - Async operations
- `--distributed` - Distributed training

**Examples:**
```bash
/jax-orbax-checkpoint --save --async
/jax-orbax-checkpoint --restore --distributed
```

---

## 🔬 Julia Scientific Computing

### `/julia-ad-grad [options]`
Generate Julia automatic differentiation code using Zygote.jl.

**Options:**
- `--higher-order` - Higher-order derivatives
- `--vectorize` - Vectorized operations
- `--performance` - Performance optimization

**Examples:**
```bash
/julia-ad-grad --higher-order --vectorize
/julia-ad-grad --performance
```

### `/julia-jit-like [options]`
Optimize Julia functions mimicking JAX JIT using Enzyme.jl.

**Options:**
- `--type-stability` - Type stability analysis
- `--precompile` - Precompilation optimization
- `--enzyme-ad` - Enzyme automatic differentiation
- `--benchmark` - Benchmarking

**Examples:**
```bash
/julia-jit-like --type-stability --precompile
/julia-jit-like --enzyme-ad --benchmark
```

### `/julia-prob-model [options]`
Build probabilistic models in Julia with Turing.jl.

**Options:**
- `--mcmc` - MCMC sampling
- `--variational` - Variational inference
- `--parallel` - Parallel computation
- `--distributed` - Distributed computing

**Examples:**
```bash
/julia-prob-model --mcmc --parallel
/julia-prob-model --variational --distributed
```

---

## 📚 Documentation

### `/update-docs [options]`
Revolutionary documentation generation with AI reasoning.

**Options:**
- `--type=TYPE` - Documentation type
- `--quantum-ai` - Quantum-level AI analysis
- `--multi-modal` - Multi-modal content
- `--research-grade` - Research-grade quality
- `--auto-optimize` - Auto-optimization
- `--real-time` - Real-time updates
- `--collaborative` - Collaborative features
- `--publication-ready` - Publication quality
- `--interactive-docs` - Interactive documentation

**Examples:**
```bash
/update-docs --research-grade --publication-ready
/update-docs --multi-modal --interactive-docs --real-time
```

---

## 🏗️ Common Flags

Most commands support these common options:

### Execution Control
- `--dry-run` - Preview changes without applying them
- `--interactive` - Interactive mode with user prompts
- `--auto-fix` - Automatically fix detected issues
- `--emergency` - Emergency/urgent mode

### Analysis & Reporting
- `--report` - Generate detailed reports
- `--comprehensive` - Comprehensive analysis
- `--ai-insights` - Include AI-powered insights
- `--benchmark` - Include benchmarking data

### Technical Options
- `--language=auto|python|julia|jax` - Target language
- `--framework=jax|julia|python|flax|pytorch` - Target framework
- `--gpu` - GPU-specific operations
- `--parallel` - Parallel execution
- `--distributed` - Distributed computing

### Quality & Validation
- `--validate` - Validate before execution
- `--backup` - Create backup before changes
- `--git-integration` - Git integration features
- `--ci-mode` - CI/CD integration mode

---

## Quick Start Examples

### Scientific Computing Workflow
```bash
# Initialize JAX project
/jax-init

# Generate tests for your model
/generate-tests model.py --type=scientific --framework=jax

# Optimize your code
/optimize

# Run comprehensive tests
/run-all-tests --scientific --gpu --coverage

# Commit with AI message
/commit --all --ai-message --push
```

### Code Quality Pipeline
```bash
# Analyze code quality
/check-code-quality . --research --gpu-analysis --report

# Clean up codebase
/clean-codebase . --dry-run --report

# Refactor for performance
/refactor-clean model.py --focus=performance --framework=jax

# Double-check everything
/double-check --comprehensive --security --report
```

### JAX Development Workflow
```bash
# Set up mixed precision
/jax-mixed-prec --precision-level=bfloat16 --monitoring

# Create ML training loop
/jax-ml-train --model-type=transformer --optimizer=adamw

# Optimize with JIT
/jax-jit train_step

# Debug if needed
/jax-debug --disable-jit --check-tracers
```

---

*This cheatsheet covers all available slash commands as of the latest Claude Code version. For the most up-to-date information, refer to the official Claude Code documentation.*