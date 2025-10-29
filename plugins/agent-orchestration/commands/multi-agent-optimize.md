---
description: Coordinate multiple specialized agents for code optimization and review tasks with intelligent orchestration, resource allocation, and multi-dimensional analysis including scientific computing
allowed-tools: Bash(find:*), Bash(grep:*), Bash(git:*), Bash(python:*), Bash(julia:*), Bash(npm:*), Bash(cargo:*)
argument-hint: <target-path> [--agents=agent1,agent2] [--focus=performance,quality,research,scientific] [--parallel]
color: magenta
agents:
  primary:
    - multi-agent-orchestrator
    - systems-architect
  conditional:
    - agent: hpc-numerical-coordinator
      trigger: pattern "numpy|scipy|pandas|matplotlib|scientific.*computing|numerical|simulation" OR argument "--focus=scientific"
    - agent: jax-pro
      trigger: pattern "jax|flax|@jit|@vmap|@pmap|grad\\(|optax"
    - agent: neural-architecture-engineer
      trigger: pattern "torch|pytorch|tensorflow|keras|neural.*network|deep.*learning"
    - agent: correlation-function-expert
      trigger: pattern "correlation|fft|spectral.*analysis|statistical.*physics"
    - agent: simulation-expert
      trigger: pattern "lammps|gromacs|molecular.*dynamics|md.*simulation|ase"
    - agent: code-quality
      trigger: argument "--focus=quality" OR pattern "test|quality|lint"
    - agent: research-intelligence
      trigger: argument "--focus=research" OR pattern "research|publication|analysis"
  orchestrated: true
---

# Multi-Agent Optimization Toolkit

## Role: AI-Powered Multi-Agent Performance Engineering Specialist

### Context
The Multi-Agent Optimization Tool is an advanced AI-driven framework designed to holistically improve system performance through intelligent, coordinated agent-based optimization. Leveraging cutting-edge AI orchestration techniques, this tool provides a comprehensive approach to performance engineering across multiple domains including web applications, distributed systems, and scientific computing.

### Core Capabilities
- Intelligent multi-agent coordination and orchestration
- Performance profiling and bottleneck identification
- Adaptive optimization strategies across domains
- Cross-domain performance optimization (web, backend, scientific, ML)
- Scientific computing and numerical algorithm optimization
- Cost and efficiency tracking
- Conflict resolution and meta-analysis
- Parallel agent execution with synthesis

## Arguments Handling
The tool processes optimization arguments with flexible input parameters:
- `$TARGET`: Primary system/application to optimize (default: `$ARGUMENTS`)
- `--agents`: Specific agents to deploy (comma-separated)
- `--focus`: Focus areas (performance, quality, research, scientific)
- `--parallel`: Enable maximum parallelization
- `--auto-execute`: Execute optimizations automatically (default: analysis only)

## 1. Multi-Agent Performance Profiling

### Profiling Strategy
- Distributed performance monitoring across system layers
- Real-time metrics collection and analysis
- Continuous performance signature tracking

#### Profiling Agents
1. **Database Performance Agent**
   - Query execution time analysis
   - Index utilization tracking
   - Resource consumption monitoring

2. **Application Performance Agent**
   - CPU and memory profiling
   - Algorithmic complexity assessment
   - Concurrency and async operation analysis

3. **Frontend Performance Agent**
   - Rendering performance metrics
   - Network request optimization
   - Core Web Vitals monitoring

### Profiling Code Example
```python
def multi_agent_profiler(target_system):
    agents = [
        DatabasePerformanceAgent(target_system),
        ApplicationPerformanceAgent(target_system),
        FrontendPerformanceAgent(target_system)
    ]

    performance_profile = {}
    for agent in agents:
        performance_profile[agent.__class__.__name__] = agent.profile()

    return aggregate_performance_metrics(performance_profile)
```

## 2. Context Window Optimization

### Optimization Techniques
- Intelligent context compression
- Semantic relevance filtering
- Dynamic context window resizing
- Token budget management

### Context Compression Algorithm
```python
def compress_context(context, max_tokens=4000):
    # Semantic compression using embedding-based truncation
    compressed_context = semantic_truncate(
        context,
        max_tokens=max_tokens,
        importance_threshold=0.7
    )
    return compressed_context
```

## 3. Agent Coordination Efficiency

### Coordination Principles
- Parallel execution design
- Minimal inter-agent communication overhead
- Dynamic workload distribution
- Fault-tolerant agent interactions

### Orchestration Framework
```python
class MultiAgentOrchestrator:
    def __init__(self, agents):
        self.agents = agents
        self.execution_queue = PriorityQueue()
        self.performance_tracker = PerformanceTracker()

    def optimize(self, target_system):
        # Parallel agent execution with coordinated optimization
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(agent.optimize, target_system): agent
                for agent in self.agents
            }

            for future in concurrent.futures.as_completed(futures):
                agent = futures[future]
                result = future.result()
                self.performance_tracker.log(agent, result)
```

## 4. Parallel Execution Optimization

### Key Strategies
- Asynchronous agent processing
- Workload partitioning
- Dynamic resource allocation
- Minimal blocking operations

## 5. Cost Optimization Strategies

### LLM Cost Management
- Token usage tracking
- Adaptive model selection
- Caching and result reuse
- Efficient prompt engineering

### Cost Tracking Example
```python
class CostOptimizer:
    def __init__(self):
        self.token_budget = 100000  # Monthly budget
        self.token_usage = 0
        self.model_costs = {
            'gpt-4': 0.03,
            'claude-3-sonnet': 0.015,
            'claude-3-haiku': 0.0025
        }

    def select_optimal_model(self, complexity):
        # Dynamic model selection based on task complexity and budget
        pass
```

## 6. Latency Reduction Techniques

### Performance Acceleration
- Predictive caching
- Pre-warming agent contexts
- Intelligent result memoization
- Reduced round-trip communication

## 7. Quality vs Speed Tradeoffs

### Optimization Spectrum
- Performance thresholds
- Acceptable degradation margins
- Quality-aware optimization
- Intelligent compromise selection

## 8. Monitoring and Continuous Improvement

### Observability Framework
- Real-time performance dashboards
- Automated optimization feedback loops
- Machine learning-driven improvement
- Adaptive optimization strategies

## 9. Scientific Computing Stack Detection

### Language & Framework Auto-Detection

#### Scientific Computing Stacks
- **Python Scientific**: Detect NumPy, SciPy, JAX, Pandas
  ```bash
  python -c "import numpy, scipy, jax; print('Scientific stack detected')"
  ```
- **JAX Ecosystem**: Detect JAX, Flax, Optax for GPU acceleration
- **Julia/SciML**: Detect DifferentialEquations.jl, SciML ecosystem
- **PyTorch/TensorFlow**: Deep learning frameworks
- **Quantum Computing**: Qiskit, Cirq, PennyLane

#### Domain-Specific Pattern Detection
- **Neutron/X-ray Scattering**: SANS, SAXS, diffraction keywords
- **Soft Matter Physics**: Polymer, colloid, rheology patterns
- **Stochastic Processes**: Monte Carlo, Gillespie, Langevin
- **Correlation Analysis**: Autocorrelation, pair distribution, FFT
- **Molecular Dynamics**: LAMMPS, GROMACS, MD simulation patterns
- **Neural Networks**: CNN, RNN, transformer architectures

### Scientific Agent Registry

#### HPC & Numerical Agents
- **hpc-numerical-coordinator**: Numerical algorithm optimization
- **jax-pro**: JAX transformations (jit, vmap, pmap, grad)
- **correlation-function-expert**: FFT-based spectral analysis
- **simulation-expert**: Molecular dynamics and multiscale simulations

#### ML & Deep Learning Agents
- **neural-architecture-engineer**: Model architecture optimization
- **ai-systems-architect**: ML pipeline and serving optimization

#### Scientific Workflow Agents
- **research-intelligence**: Research methodology and strategy
- **scientific-code-adoptor**: Legacy Fortran/MATLAB modernization

## 10. Scientific Computing Optimization Patterns

### Numerical Algorithm Optimization
```python
# BEFORE: Python loops (slow)
def compute_correlation(data):
    n = len(data)
    result = []
    for i in range(n):
        for j in range(n):
            result.append(data[i] * data[j])
    return result

# AFTER: NumPy vectorization (50x faster)
def compute_correlation(data):
    return np.outer(data, data).flatten()

# ADVANCED: JAX JIT compilation (500x faster)
import jax
import jax.numpy as jnp

@jax.jit
def compute_correlation(data):
    return jnp.outer(data, data).flatten()
```

### JAX Transformation Optimization
```python
# Batch processing with vmap
@jax.jit
def process_batch(items):
    return jax.vmap(expensive_function)(items)
# Expected: 50x speedup over Python loops

# Gradient computation with automatic differentiation
compute_gradient = jax.grad(loss_function)
# Benefits: Exact gradients, higher-order derivatives
```

### GPU Acceleration Patterns
```python
# Memory-efficient GPU operations
@jax.jit
def gpu_computation(large_array):
    # In-place operations for memory efficiency
    return large_array.at[indices].set(new_values)
```

### Numerical Stability
```python
# Log-sum-exp trick for numerical stability
def stable_softmax(logits):
    max_logit = jnp.max(logits)
    exp_logits = jnp.exp(logits - max_logit)
    return exp_logits / jnp.sum(exp_logits)
```

## 11. Cross-Cutting Optimization Synthesis

### Meta-Analysis Framework
When multiple agents identify the same bottleneck:
1. **Convergent Recommendations**: Higher priority (multiple agents agree)
2. **Complementary Strategies**: Layer optimizations (vectorize + JIT + GPU)
3. **Conflicting Approaches**: Resolve via constraint analysis (memory vs speed)

### Example: Comprehensive Optimization Strategy
```
Pattern: Performance bottleneck in compute_correlation()

Agents identified by:
- Scientific Computing Master: Vectorization opportunity
- JAX Pro: JIT compilation candidate
- Systems Architect: Hot loop accounting for 65% runtime

Synthesized Strategy:
1. Phase 1: NumPy vectorization → 50x speedup (low risk)
2. Phase 2: JAX JIT compilation → additional 10x (medium risk)
3. Phase 3: GPU migration → additional 5x (requires GPU)

Expected Total: 2500x improvement
```

## Reference Workflows

### Workflow 1: Scientific Computing Optimization
1. Detect scientific stack (NumPy, JAX, Julia)
2. Profile numerical hotspots
3. Deploy scientific agents (hpc-numerical-coordinator, jax-pro)
4. Apply vectorization and JIT compilation
5. Validate numerical accuracy
6. Measure performance improvements

### Workflow 2: E-Commerce Platform Optimization
1. Initial performance profiling
2. Agent-based optimization
3. Cost and performance tracking
4. Continuous improvement cycle

### Workflow 3: ML Training Pipeline Enhancement
1. Detect ML framework (PyTorch/JAX/TensorFlow)
2. Deploy neural-architecture-engineer
3. Optimize training loop (mixed precision, gradient accumulation)
4. Implement caching and data loading optimization
5. Validate model accuracy unchanged

### Workflow 4: Enterprise API Performance Enhancement
1. Comprehensive system analysis
2. Multi-layered agent optimization
3. Iterative performance refinement
4. Cost-efficient scaling strategy

## Key Considerations
- Always measure before and after optimization
- Maintain system stability during optimization
- Balance performance gains with resource consumption
- Implement gradual, reversible changes
- **For scientific code**: Validate numerical accuracy after optimization
- **For ML models**: Ensure model performance unchanged
- **For simulations**: Verify energy conservation and physical properties

Target Optimization: $ARGUMENTS

## Execution Modes

### Analysis Only (Default)
```bash
/multi-agent-optimize src/
# Analyzes and provides recommendations, no changes
```

### Targeted Agents
```bash
/multi-agent-optimize src/ --agents=jax-pro,hpc-numerical-coordinator
# Deploy only specific agents
```

### Focus Areas
```bash
/multi-agent-optimize src/ --focus=scientific,performance
# Prioritize scientific computing and performance optimization
```

### Parallel Execution
```bash
/multi-agent-optimize src/ --parallel
# Maximum parallelization of agent analyses
```