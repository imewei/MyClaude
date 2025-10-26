---
description: Coordinate multiple specialized agents for code optimization and review tasks with intelligent orchestration, resource allocation, and multi-dimensional analysis
allowed-tools: Bash(find:*), Bash(grep:*), Bash(git:*), Bash(python:*), Bash(julia:*), Bash(npm:*), Bash(cargo:*)
argument-hint: <target-path> [--agents=agent1,agent2] [--focus=performance,quality,research] [--parallel]
color: magenta
agents:
  primary:
    - systems-architect
    - code-quality
  conditional:
    - agent: hpc-numerical-coordinator
      trigger: pattern "performance|optimization|numerical" OR argument "--focus=performance"
    - agent: research-intelligence
      trigger: argument "--focus=research" OR pattern "research|analysis"
  orchestrated: true
---

# Multi-Agent Optimization Orchestration System

## Phase 0: Project Discovery & Agent Selection

### Repository Context
- Working directory: !`pwd`
- Project root: !`git rev-parse --show-toplevel 2>/dev/null || pwd`
- Total files: !`find . -type f 2>/dev/null | wc -l`
- Code files: !`find . -name "*.py" -o -name "*.jl" -o -name "*.js" -o -name "*.ts" -o -name "*.rs" -o -name "*.cpp" -o -name "*.java" 2>/dev/null | grep -v node_modules | wc -l`
- Repository size: !`du -sh . 2>/dev/null | cut -f1`

### Language & Framework Detection

#### Scientific Computing Stack
- **Python Scientific**: !`python -c "import numpy, scipy, jax 2>/dev/null; print('NumPy, SciPy, JAX')" 2>/dev/null || python -c "import numpy, scipy 2>/dev/null; print('NumPy, SciPy')" 2>/dev/null || echo "Not detected"`
- **JAX Ecosystem**: !`python -c "import jax, flax, optax 2>/dev/null; print(f'JAX {jax.__version__}')" 2>/dev/null || echo "Not installed"`
- **Julia/SciML**: !`julia -e 'using Pkg; println(filter(p -> p.name in ["DifferentialEquations", "SciML"], values(Pkg.dependencies())))' 2>/dev/null || echo "Not detected"`
- **PyTorch**: !`python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null || echo "Not installed"`
- **TensorFlow**: !`python -c "import tensorflow as tf; print(f'TF {tf.__version__}')" 2>/dev/null || echo "Not installed"`

#### Quantum Computing
- **Qiskit**: !`python -c "import qiskit; print(f'Qiskit {qiskit.__version__}')" 2>/dev/null || echo "Not installed"`
- **Cirq**: !`python -c "import cirq; print(f'Cirq {cirq.__version__}')" 2>/dev/null || echo "Not installed"`
- **PennyLane**: !`python -c "import pennylane; print(f'PennyLane {pennylane.__version__}')" 2>/dev/null || echo "Not installed"`

#### Web & Application Stack
- **Frontend**: !`find . -name "package.json" -exec grep -l "react\|vue\|angular\|svelte" {} \; 2>/dev/null | head -1`
- **Backend**: !`find . -name "*.js" -o -name "*.ts" 2>/dev/null | head -1 | xargs grep -l "express\|fastify\|nest" 2>/dev/null || echo "Not detected"`
- **Database**: !`grep -r "postgresql\|mongodb\|redis\|mysql" --include="*.{py,js,ts,env}" 2>/dev/null | head -3`

#### Infrastructure & DevOps
- **CI/CD**: !`find .github .gitlab-ci.yml .circleci -type f 2>/dev/null | wc -l` configs
- **IaC**: !`find . -name "*.tf" -o -name "cloudformation.yml" 2>/dev/null | wc -l` files

### Domain-Specific Detection

#### Scientific Domains
- **Neutron Scattering**: !`grep -ri "neutron\|scattering\|SANS\|SAXS" --include="*.{py,jl}" 2>/dev/null | wc -l` mentions
- **X-ray Analysis**: !`grep -ri "xray\|x-ray\|diffraction\|crystallography" --include="*.{py,jl}" 2>/dev/null | wc -l` mentions
- **Soft Matter**: !`grep -ri "polymer\|colloid\|rheology\|soft.matter" --include="*.{py,jl}" 2>/dev/null | wc -l` mentions
- **Stochastic Processes**: !`grep -ri "stochastic\|brownian\|langevin\|gillespie" --include="*.{py,jl}" 2>/dev/null | wc -l` mentions
- **Correlation Functions**: !`grep -ri "correlation\|autocorrelation\|pair.distribution" --include="*.{py,jl}" 2>/dev/null | wc -l` mentions

#### Neural Networks & ML
- **Deep Learning**: !`grep -ri "neural.network\|deep.learning\|CNN\|RNN\|transformer" --include="*.{py,jl}" 2>/dev/null | wc -l` mentions
- **Reinforcement Learning**: !`grep -ri "reinforcement\|policy\|Q.learning\|actor.critic" --include="*.{py,jl}" 2>/dev/null | wc -l` mentions
- **Scientific ML**: !`grep -ri "PINN\|physics.informed\|neural.ODE\|scientific.ML" --include="*.{py,jl}" 2>/dev/null | wc -l` mentions

---

## Phase 1: Intelligent Agent Selection

### Agent Registry

#### Tier 1: Orchestration & Systems (Always Active)
```yaml
multi-agent-orchestrator:
  role: "Master Coordinator"
  capabilities:
    - Workflow coordination
    - Resource allocation
    - Meta-analysis synthesis
    - Performance monitoring
    - Cross-agent communication
  activation: always
  priority: highest

command-systems-engineer:
  role: "Workflow Optimization"
  capabilities:
    - Command optimization
    - Pipeline engineering
    - Automation enhancement
    - Tool integration
  activation: always
  priority: high
```

#### Tier 2: Core Technical Agents (Domain-Triggered)
```yaml
hpc-numerical-coordinator:
  role: "Scientific Computing Optimization"
  capabilities:
    - Numerical algorithm optimization
    - Scientific workflow enhancement
    - Computational efficiency
    - Research methodology
  triggers:
    - numpy|scipy|matplotlib detected
    - .py files with scientific patterns
    - research keywords found
  priority: high

jax-pro:
  role: "JAX & GPU Optimization"
  capabilities:
    - JAX optimization (jit, vmap, pmap)
    - GPU acceleration
    - Gradient computation optimization
    - Scientific ML workflows
  triggers:
    - jax|flax|optax imports
    - @jit|@vmap decorators
    - GPU computing patterns
  priority: high

neural-architecture-engineer:
  role: "Deep Learning Optimization"
  capabilities:
    - Model architecture optimization
    - Training efficiency
    - Inference optimization
    - Neural network debugging
  triggers:
    - torch|tensorflow|keras imports
    - neural network architectures
    - training loops detected
  priority: high

advanced-quantum-computing-expert:
  role: "Quantum Computing Optimization"
  capabilities:
    - Quantum circuit optimization
    - Hybrid quantum-classical systems
    - Quantum algorithm enhancement
    - Quantum ML optimization
  triggers:
    - qiskit|cirq|pennylane imports
    - quantum circuit definitions
    - variational algorithms
  priority: high

systems-architect:
  role: "Architecture & Design"
  capabilities:
    - System design optimization
    - Architecture patterns
    - Scalability enhancement
    - Distributed systems
  triggers:
    - microservices|architecture patterns
    - large codebase (>50 files)
    - distributed systems code
  priority: high

ai-systems-architect:
  role: "AI System Design"
  capabilities:
    - AI pipeline architecture
    - Model serving optimization
    - ML system scalability
    - AI infrastructure
  triggers:
    - ML pipelines detected
    - model serving code
    - AI system patterns
  priority: high

fullstack-developer:
  role: "Full-Stack Optimization"
  capabilities:
    - Frontend optimization
    - Backend efficiency
    - API optimization
    - Application performance
  triggers:
    - web framework detected
    - API endpoints found
    - frontend code present
  priority: medium

devops-security-engineer:
  role: "DevSecOps & Infrastructure"
  capabilities:
    - CI/CD optimization
    - Security hardening
    - Infrastructure automation
    - Deployment optimization
  triggers:
    - CI/CD configs present
    - Security concerns found
  priority: high

code-quality:
  role: "Quality Assurance"
  capabilities:
    - Code quality analysis
    - Testing optimization
    - Quality metrics
    - Best practices enforcement
  triggers:
    - code quality issues detected
    - test coverage < 80%
    - linting errors present
  priority: high

documentation-architect:
  role: "Documentation Excellence"
  capabilities:
    - Documentation optimization
    - API documentation
    - Technical writing
    - Knowledge management
  triggers:
    - missing/outdated docs
    - API without docs
    - complex code undocumented
  priority: medium

research-intelligence:
  role: "Research Strategy"
  capabilities:
    - Research methodology
    - Innovation discovery
    - Knowledge synthesis
    - Strategic planning
  triggers:
    - research project detected
    - publications directory
    - experimental code
  priority: high
```

#### Tier 3: Domain Specialists (Pattern-Triggered)
```yaml
data-engineering-coordinator:
  triggers:
    - pandas|dask|spark detected
    - data pipeline code
    - ETL workflows
  priority: medium

ml-pipeline-coordinator:
  triggers:
    - sklearn|xgboost|lightgbm
    - ML algorithms present
    - model training code
  priority: medium

visualization-interface:
  triggers:
    - matplotlib|plotly|d3 detected
    - visualization code
    - dashboard present
  priority: low

database-workflow-engineer:
  triggers:
    - SQL queries present
    - ORM usage detected
    - database schemas found
  priority: medium

correlation-function-expert:
  triggers:
    - correlation|autocorrelation keywords
    - statistical analysis code
    - time series analysis
  priority: low

neutron-soft-matter-expert:
  triggers:
    - neutron|SANS|SAXS keywords
    - scattering analysis
    - soft matter simulations
  priority: low

xray-soft-matter-expert:
  triggers:
    - xray|diffraction keywords
    - crystallography code
    - structure analysis
  priority: low

nonequilibrium-stochastic-expert:
  triggers:
    - stochastic|nonequilibrium keywords
    - Monte Carlo simulations
    - kinetic theory code
  priority: low

scientific-code-adoptor:
  triggers:
    - legacy scientific code
    - Fortran|F90 code present
    - modernization needed
  priority: medium
```

### Agent Selection Algorithm

```python
def select_agents(project_analysis):
    """
    Intelligent agent selection based on project characteristics

    Returns: List of agents with priority scores
    """

    selected_agents = []

    # Tier 1: Always active
    selected_agents.extend([
        ('multi-agent-orchestrator', 1.0),
        ('command-systems-engineer', 1.0)
    ])

    # Tier 2: Domain-triggered scoring
    for agent, config in AGENT_REGISTRY.items():
        score = calculate_relevance_score(agent, project_analysis)

        if score > 0.3:  # Relevance threshold
            selected_agents.append((agent, score))

    # Sort by priority and score
    selected_agents.sort(key=lambda x: x[1], reverse=True)

    # Limit to top 8 agents to avoid coordination overhead
    return selected_agents[:8]


def calculate_relevance_score(agent, project_analysis):
    """
    Calculate 0-1 relevance score for agent

    Factors:
    - Keyword matches (0.3 weight)
    - File type matches (0.2 weight)
    - Complexity needs (0.2 weight)
    - Existing issues (0.3 weight)
    """
    score = 0.0

    # Check triggers
    if agent_triggers_match(agent, project_analysis):
        score += 0.4

    # Check complexity
    if requires_specialization(agent, project_analysis):
        score += 0.3

    # Check issue severity
    if has_critical_issues(agent.domain, project_analysis):
        score += 0.3

    return min(score, 1.0)
```

---

## Phase 2: Multi-Agent Workflow Orchestration

### Workflow Phases

**Phase 1: Parallel Initial Analysis** (All agents independently)
```
┌─────────────────────────────────────────────────────────────┐
│ PARALLEL ANALYSIS PHASE                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Agent 1: Scientific Computing ──→ Analysis Report 1       │
│  Agent 2: JAX Pro             ──→ Analysis Report 2        │
│  Agent 3: Code Quality        ──→ Analysis Report 3        │
│  Agent 4: Systems Architect   ──→ Analysis Report 4        │
│  Agent 5: DevOps Security     ──→ Analysis Report 5        │
│  Agent 6: Neural Networks     ──→ Analysis Report 6        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Phase 2: Orchestrator Synthesis** (Meta-analysis)
```
┌─────────────────────────────────────────────────────────────┐
│ META-ANALYSIS SYNTHESIS                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  All Reports ──→ Multi-Agent Orchestrator                  │
│                          │                                  │
│                          ↓                                  │
│                  Conflict Resolution                        │
│                  Priority Ranking                           │
│                  Resource Allocation                        │
│                  Implementation Plan                        │
│                          │                                  │
│                          ↓                                  │
│                  Unified Strategy ──→                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Phase 3: Collaborative Deep Dive** (Sequential with handoffs)
```
┌─────────────────────────────────────────────────────────────┐
│ DEEP DIVE PHASE                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Critical Issue 1 ──→ Agent A + Agent B (collaboration)    │
│  Critical Issue 2 ──→ Agent C + Agent D (collaboration)    │
│  Critical Issue 3 ──→ Agent E (solo deep analysis)         │
│                                                             │
│  Each team produces detailed optimization plan             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Phase 4: Implementation Coordination** (Orchestrated application)
```
┌─────────────────────────────────────────────────────────────┐
│ IMPLEMENTATION PHASE                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Batch 1: Low-risk optimizations (parallel)                │
│  Batch 2: Medium-risk changes (sequential with validation) │
│  Batch 3: High-risk refactors (careful orchestration)      │
│                                                             │
│  Continuous validation and rollback capability             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Orchestration Protocol

**Message Passing System**:
```yaml
agent_communication:
  discovery:
    - agent broadcasts capabilities
    - orchestrator maps agent to tasks

  analysis:
    - each agent analyzes independently
    - reports to orchestrator with findings

  collaboration:
    - agents request collaboration for complex issues
    - orchestrator facilitates agent-to-agent communication

  conflict_resolution:
    - orchestrator detects conflicting recommendations
    - convenes agent committee for resolution
    - applies ultrathink reasoning for final decision

  implementation:
    - orchestrator sequences implementation
    - agents validate their domain post-change
    - continuous monitoring and adjustment
```

---

## Phase 3: UltraThink Meta-Analysis Layer

### Multi-Dimensional Reasoning Framework

**1. Cross-Domain Pattern Recognition**

```python
class MetaAnalyzer:
    """
    Synthesizes insights across multiple agent analyses
    """

    def identify_cross_cutting_concerns(self, agent_reports):
        """
        Find patterns that multiple agents identified

        Example:
        - Scientific Computing Agent: "Needs vectorization"
        - JAX Pro: "Can benefit from vmap"
        - Performance Agent: "Hot loop detected"

        Synthesis: "Critical vectorization opportunity with
                    JAX vmap - high impact optimization"
        """

        concerns = {}

        for report in agent_reports:
            for issue in report.issues:
                pattern = extract_pattern(issue)

                if pattern in concerns:
                    concerns[pattern].append({
                        'agent': report.agent,
                        'severity': issue.severity,
                        'recommendation': issue.solution
                    })
                else:
                    concerns[pattern] = [...]

        # Rank by number of agents mentioning + severity
        return rank_concerns(concerns)

    def detect_conflicts(self, agent_reports):
        """
        Identify contradictory recommendations

        Example:
        - Agent A: "Optimize for speed, use aggressive caching"
        - Agent B: "Optimize for memory, minimize caching"

        Resolution: Determine primary constraint (speed vs memory)
                   based on project requirements
        """
        conflicts = []

        for i, report1 in enumerate(agent_reports):
            for j, report2 in enumerate(agent_reports[i+1:]):
                if recommendations_conflict(report1, report2):
                    conflicts.append({
                        'agents': [report1.agent, report2.agent],
                        'issue': describe_conflict(report1, report2),
                        'resolution_strategy': determine_resolution(...)
                    })

        return conflicts

    def synthesize_strategy(self, agent_reports, conflicts):
        """
        Create unified optimization strategy

        Process:
        1. Identify high-impact, low-risk quick wins
        2. Resolve conflicts based on project priorities
        3. Sequence optimizations to avoid interference
        4. Allocate resources based on expected ROI
        5. Create validation checkpoints
        """

        strategy = {
            'quick_wins': [],
            'major_optimizations': [],
            'long_term_refactors': [],
            'risk_mitigation': [],
            'success_metrics': []
        }

        # Apply ultrathink reasoning
        for issue in aggregate_issues(agent_reports):
            category = categorize_by_impact_and_risk(issue)
            strategy[category].append(issue)

        return strategy
```

**2. Impact vs. Effort Analysis**

```
Impact-Effort Matrix:

High Impact │ ⭐ DO FIRST          │ 📋 PLAN CAREFULLY    │
            │                     │                      │
            │ • JAX JIT critical  │ • Architecture       │
            │   functions         │   refactor           │
            │ • Vectorize hot     │ • Distributed        │
            │   loops             │   computing          │
            │ • Fix memory leaks  │ • ML pipeline        │
            │                     │   redesign           │
            │─────────────────────│──────────────────────│
            │ ✅ QUICK WINS       │ ❌ AVOID             │
Low Impact  │                     │                      │
            │ • Format code       │ • Premature          │
            │ • Update comments   │   optimization       │
            │ • Naming cleanup    │ • Over-engineering   │
            │                     │                      │
            └─────────────────────┴──────────────────────┘
              Low Effort             High Effort
```

**3. Dependency Graph Analysis**

```python
def build_optimization_dag(optimizations):
    """
    Create dependency graph for optimization sequence

    Example:
    A: Fix data loading → enables → B: Vectorize computation
    B: Vectorize → enables → C: GPU acceleration
    D: Refactor API → independent

    Result: Parallel: [A, D], Sequential: A→B→C
    """

    graph = nx.DiGraph()

    for opt in optimizations:
        graph.add_node(opt.id, **opt.metadata)

        # Add dependencies
        for dependency in opt.requires:
            graph.add_edge(dependency, opt.id)

        # Add conflicts (mutex)
        for conflict in opt.conflicts_with:
            graph.add_edge(opt.id, conflict, type='mutex')

    # Find optimal execution order
    execution_plan = topological_sort(graph)
    parallel_batches = identify_parallel_opportunities(graph)

    return execution_plan, parallel_batches
```

**4. Risk Assessment Framework**

```python
def assess_risk(optimization, project_state):
    """
    Multi-factor risk assessment

    Factors:
    - Code complexity (cyclomatic complexity, dependencies)
    - Test coverage (higher coverage = lower risk)
    - Change scope (lines changed, files affected)
    - Reversibility (easy to rollback?)
    - Team expertise (familiar patterns?)
    - Production impact (user-facing?)
    """

    risk_score = 0.0

    # Complexity risk
    if optimization.complexity > 10:
        risk_score += 0.2

    # Coverage risk
    if project_state.test_coverage < 0.8:
        risk_score += 0.3

    # Scope risk
    if optimization.files_affected > 10:
        risk_score += 0.2

    # Reversibility risk
    if not optimization.easily_reversible:
        risk_score += 0.2

    # Production risk
    if optimization.affects_production:
        risk_score += 0.3

    return {
        'score': min(risk_score, 1.0),
        'level': categorize_risk(risk_score),
        'mitigation': suggest_mitigation(optimization, risk_score)
    }
```

---

## Phase 4: Agent-Specific Analysis Templates

### Scientific Computing Master Analysis

```markdown
## Scientific Computing Optimization Report

### Numerical Algorithm Analysis
- **Algorithm Complexity**: O(n²) → Opportunity for O(n log n) improvement
- **Numerical Stability**: 3 functions at risk of catastrophic cancellation
- **Precision**: Mixed float32/float64 usage - recommend standardization
- **Convergence**: 2 iterative methods without convergence checks

### Performance Hotspots (Profiling Results)
1. **Function: compute_correlation** (65% of runtime)
   - Current: Pure Python loops
   - Recommendation: NumPy vectorization
   - Expected Speedup: 50x

2. **Function: solve_linear_system** (20% of runtime)
   - Current: General solver for sparse matrix
   - Recommendation: Specialized sparse solver
   - Expected Speedup: 10x

### Scientific Workflow Optimization
- **Data Pipeline**: Currently sequential, can parallelize 3 stages
- **Reproducibility**: Missing random seeds in 5 functions
- **Validation**: 40% of functions lack numerical validation tests

### Recommendations (Priority Order)
1. ⭐ Vectorize compute_correlation with NumPy (Impact: High, Effort: Low)
2. ⭐ Use sparse solver for linear systems (Impact: High, Effort: Medium)
3. 📋 Add convergence checks to iterative methods (Impact: Medium, Effort: Low)
4. 📋 Implement parallel data pipeline (Impact: High, Effort: High)
```

### JAX Pro Analysis

```markdown
## JAX Optimization Report

### JIT Compilation Opportunities
- **Functions suitable for @jit**: 12 identified
  - Pure functions: 8 (ready for JIT)
  - Functions with side effects: 4 (need refactoring)

- **Expected Speedup**: 10-100x for JIT-compiled functions

### Vectorization Analysis (vmap)
- **Batch operations detected**: 5 locations
  - Currently using Python loops
  - Can use vmap for automatic vectorization
  - Expected Speedup: 10-50x

### Gradient Computation
- **Differentiable functions**: 8 identified
- **Current approach**: Manual derivatives (error-prone)
- **Recommendation**: Use jax.grad
- **Benefits**:
  - Correctness guarantee
  - Support for higher-order derivatives
  - Reverse-mode AD for efficient gradients

### GPU Acceleration Potential
- **Suitable for GPU**: 6 functions
  - Matrix operations: 4 functions
  - Element-wise operations: 2 functions
- **Expected GPU Speedup**: 50-200x

### Memory Optimization
- **Large array allocations**: 3 functions
- **Recommendation**: Use in-place operations with `.at[]` syntax
- **Memory Savings**: ~60% reduction

### Specific Optimizations

#### High Priority
```python
# BEFORE
def batch_process(items):
    results = []
    for item in items:
        results.append(expensive_function(item))
    return jnp.array(results)

# AFTER
@jit
def batch_process(items):
    return vmap(expensive_function)(items)
# Expected: 50x speedup
```

#### Medium Priority
```python
# BEFORE
def compute_gradient(f, x):
    eps = 1e-5
    grad = jnp.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.at[i].add(eps)
        x_minus = x.at[i].add(-eps)
        grad = grad.at[i].set((f(x_plus) - f(x_minus)) / (2*eps))
    return grad

# AFTER
compute_gradient = jax.grad(f)
# Benefits: Exact gradients, faster, supports higher-order
```
```

### Code Quality Master Analysis

```markdown
## Code Quality Report

### Quality Metrics
- **Maintainability Index**: 62/100 (Target: >70)
- **Cyclomatic Complexity**: Average 8.5 (Target: <10)
- **Code Duplication**: 12% (Target: <5%)
- **Test Coverage**: 65% (Target: >90%)

### Issues by Severity

#### Critical (Fix Immediately)
1. **No error handling in 8 critical functions**
   - Functions: data_loader, process_batch, save_results
   - Risk: Silent failures, data corruption
   - Fix: Add try-except with proper logging

2. **Type hints missing in 45% of functions**
   - Impact: No static type checking, harder to maintain
   - Fix: Add type hints using mypy

#### High Priority
3. **Code duplication in validation logic** (5 instances)
   - Lines duplicated: ~150
   - Fix: Extract to shared validation module

4. **Complex functions exceeding 50 lines** (12 functions)
   - Largest: 180 lines
   - Fix: Refactor into smaller, testable units

#### Medium Priority
5. **Inconsistent naming conventions** (30 instances)
6. **Missing docstrings** (40% of public functions)
7. **Hardcoded magic numbers** (25 instances)

### Testing Strategy Gaps
- **Missing test categories**:
  - Edge case tests: 0/15 functions
  - Integration tests: 0
  - Performance regression tests: 0

- **Test quality issues**:
  - 20 tests with no assertions
  - 15 tests always passing (not testing anything)
  - No property-based tests

### Refactoring Opportunities
1. **Extract 3 classes from procedural code**
2. **Implement dependency injection** (5 tightly coupled modules)
3. **Apply strategy pattern** (4 if-else chains)

### CI/CD Quality Gates
- ❌ No automatic testing on PR
- ❌ No code coverage requirements
- ❌ No linting in CI
- ✅ Basic build check exists

### Recommendations
1. ⭐ Add error handling to critical functions (1 hour)
2. ⭐ Achieve 80% test coverage (1 week)
3. 📋 Refactor complex functions (2 weeks)
4. 📋 Setup CI quality gates (2 hours)
```

### Systems Architect Analysis

```markdown
## System Architecture Report

### Current Architecture Assessment
- **Pattern**: Monolithic with some modular components
- **Scalability**: Limited - bottlenecks in data processing
- **Maintainability**: Medium - some coupling issues
- **Performance**: CPU-bound, not utilizing available resources

### Architecture Recommendations

#### 1. Modular Refactoring
```
Current:
┌─────────────────────────────────┐
│     Monolithic Application      │
│  (data, compute, API, storage)  │
└─────────────────────────────────┘

Proposed:
┌──────────┐  ┌──────────┐  ┌──────────┐
│   Data   │→ │ Compute  │→ │   API    │
│  Layer   │  │  Layer   │  │  Layer   │
└──────────┘  └──────────┘  └──────────┘
      │            │              │
      └────────────┴──────────────┘
                   ↓
            ┌──────────┐
            │ Storage  │
            └──────────┘
```

#### 2. Distributed Computing Integration
- **Current**: Single-machine processing
- **Proposed**: Dask/Ray for distributed computation
- **Expected**: 5-10x throughput improvement

#### 3. Caching Strategy
- **L1**: Function-level memoization (functools.lru_cache)
- **L2**: Redis for shared computation results
- **L3**: Disk cache for large arrays
- **Expected**: 70% cache hit rate → 3x speedup

#### 4. Async I/O Pattern
- **Current**: Synchronous blocking I/O
- **Proposed**: Async with asyncio
- **Impact**: Concurrent I/O operations, better resource utilization
```

---

## Phase 5: Conflict Resolution & Synthesis

### Conflict Detection

```python
class ConflictResolver:
    """
    Resolves contradictory recommendations from agents
    """

    def resolve(self, conflict):
        """
        Resolution strategies:
        1. Priority-based (domain expert wins)
        2. Evidence-based (more evidence wins)
        3. Risk-based (lower risk preferred)
        4. Committee-based (agent discussion)
        5. User-choice (present options to user)
        """

        if conflict.type == 'optimization_tradeoff':
            return self.resolve_tradeoff(conflict)
        elif conflict.type == 'implementation_approach':
            return self.resolve_approach(conflict)
        elif conflict.type == 'priority_disagreement':
            return self.resolve_priority(conflict)

    def resolve_tradeoff(self, conflict):
        """
        Example Conflict:
        - Agent A: "Optimize for speed → use more memory"
        - Agent B: "Optimize for memory → accept slower speed"

        Resolution:
        1. Identify project constraints (speed critical? memory limited?)
        2. Calculate pareto frontier
        3. Recommend balanced solution or let user decide
        """

        # Analyze constraints
        constraints = analyze_constraints(project_state)

        if constraints['memory'] == 'critical':
            return {
                'decision': 'prioritize_memory',
                'rationale': 'Memory constraints are critical',
                'implementation': conflict.agent_b.recommendation
            }
        elif constraints['speed'] == 'critical':
            return {
                'decision': 'prioritize_speed',
                'rationale': 'Performance is critical',
                'implementation': conflict.agent_a.recommendation
            }
        else:
            # Calculate balanced solution
            return {
                'decision': 'balanced_approach',
                'rationale': 'No critical constraint, balance both',
                'implementation': find_pareto_optimal_solution(conflict)
            }
```

### Example Conflict Resolution

```markdown
## Conflict: Memory vs Speed Optimization

### Agents Involved
- **Scientific Computing Master**: Recommends aggressive caching for 10x speedup
- **Systems Architect**: Warns of memory exhaustion with current caching strategy

### Analysis
- **Memory Available**: 32GB
- **Current Usage**: 28GB (87%)
- **Proposed Cache Size**: 8GB
- **Result**: Would cause OOM errors

### Resolution Strategy
```python
# Hybrid approach
def optimized_caching_strategy():
    """
    Balance speed and memory using adaptive cache
    """
    # Use LRU cache with size limit
    from functools import lru_cache
    from cachetools import LRUCache

    # Limit cache to 2GB (safe memory headroom)
    cache = LRUCache(maxsize=calculate_safe_cache_size())

    @cached(cache)
    def expensive_computation(x):
        return compute(x)

    # Benefits:
    # - 5x speedup (vs 10x with unlimited cache)
    # - Safe memory usage
    # - Automatic eviction of old entries
```

### Outcome
- **Speedup Achieved**: 5x (vs desired 10x)
- **Memory Safety**: Guaranteed
- **Recommendation**: Acceptable tradeoff, revisit if more RAM available
```

---

## Phase 6: Implementation Orchestration

### Execution Plan Generation

```python
class ImplementationOrchestrator:
    """
    Coordinates implementation of optimizations
    """

    def generate_execution_plan(self, optimizations):
        """
        Create safe, efficient execution plan

        Considerations:
        - Dependencies between optimizations
        - Risk levels
        - Testing requirements
        - Rollback strategies
        """

        plan = {
            'phases': [],
            'validation_points': [],
            'rollback_triggers': []
        }

        # Phase 1: Quick wins (low risk, high impact)
        quick_wins = filter_by_criteria(
            optimizations,
            risk='low',
            impact='high'
        )
        plan['phases'].append({
            'name': 'Quick Wins',
            'optimizations': quick_wins,
            'execution': 'parallel',
            'validation': 'after_each'
        })

        # Phase 2: Major optimizations (sequential)
        major_opts = filter_by_criteria(
            optimizations,
            risk='medium',
            impact='high'
        )
        plan['phases'].append({
            'name': 'Major Optimizations',
            'optimizations': major_opts,
            'execution': 'sequential',
            'validation': 'comprehensive_after_each'
        })

        # Phase 3: Refactors (careful)
        refactors = filter_by_criteria(
            optimizations,
            risk='high',
            impact='high'
        )
        plan['phases'].append({
            'name': 'Architectural Refactors',
            'optimizations': refactors,
            'execution': 'sequential_with_review',
            'validation': 'full_suite_plus_manual'
        })

        return plan
```

### Validation Framework

```python
class ValidationOrchestrator:
    """
    Ensures each optimization works correctly
    """

    def validate(self, optimization, scope='full'):
        """
        Multi-level validation

        Levels:
        1. Unit tests (affected functions)
        2. Integration tests (affected workflows)
        3. Performance benchmarks (verify improvement)
        4. Regression tests (ensure no breaks)
        5. Manual review (for high-risk changes)
        """

        results = {
            'passed': False,
            'tests': {},
            'performance': {},
            'issues': []
        }

        # Run unit tests
        results['tests']['unit'] = run_unit_tests(
            affected_files=optimization.files
        )

        # Run integration tests
        results['tests']['integration'] = run_integration_tests(
            affected_modules=optimization.modules
        )

        # Benchmark performance
        if optimization.claims_speedup:
            results['performance'] = benchmark_and_compare(
                before=baseline,
                after=current,
                expected_improvement=optimization.expected_speedup
            )

        # Check for regressions
        results['tests']['regression'] = run_full_test_suite()

        # Determine if validation passed
        results['passed'] = all([
            results['tests']['unit'].passed,
            results['tests']['integration'].passed,
            results['tests']['regression'].passed,
            results['performance'].meets_expectations
        ])

        return results
```

---

## Phase 7: Comprehensive Reporting

### Multi-Agent Optimization Report

```markdown
# Multi-Agent Optimization Report

Generated: {timestamp}
Target: {target_path}
Active Agents: {agent_count}

## Executive Summary

### Agents Deployed
1. ✅ **Multi-Agent Orchestrator** - Coordination and synthesis
2. ✅ **Scientific Computing Master** - Numerical optimization
3. ✅ **JAX Pro** - GPU acceleration and JAX optimization
4. ✅ **Code Quality Master** - Quality assurance
5. ✅ **Systems Architect** - Architecture optimization
6. ✅ **DevOps Security Engineer** - Infrastructure optimization

### Overall Assessment
- **Project Health**: 7.5/10
- **Optimization Potential**: HIGH
- **Critical Issues**: 3
- **Quick Win Opportunities**: 12
- **Expected Performance Improvement**: 10-50x (with all optimizations)

---

## Agent Findings Summary

### 🔬 Scientific Computing Master

**Priority Findings**:
1. ⭐ **Vectorization Opportunity** (compute_correlation)
   - Current: Python loops (65% of runtime)
   - Solution: NumPy vectorization
   - Impact: 50x speedup, 0.5 day effort
   - Status: READY FOR IMPLEMENTATION

2. 📋 **Numerical Stability Issues**
   - 3 functions at risk of catastrophic cancellation
   - Solution: Reformulate using stable algorithms
   - Impact: Correctness improvement
   - Status: REQUIRES ANALYSIS

**Metrics**:
- Functions analyzed: 45
- Hotspots identified: 8
- Optimization opportunities: 15
- Expected overall speedup: 10-20x

---

### ⚡ JAX Pro

**Priority Findings**:
1. ⭐ **JIT Compilation** (12 functions)
   - Pure functions ready for @jit
   - Impact: 10-100x speedup
   - Effort: 2 hours (simple decorators)
   - Status: READY FOR IMPLEMENTATION

2. ⭐ **Batch Vectorization** (vmap)
   - 5 batch operations using Python loops
   - Solution: Replace with vmap
   - Impact: 10-50x speedup
   - Status: READY FOR IMPLEMENTATION

3. 📋 **GPU Migration**
   - 6 functions suitable for GPU
   - Impact: 50-200x speedup (if GPU available)
   - Effort: 1 week (testing on GPU)
   - Status: REQUIRES GPU ACCESS

**Metrics**:
- JIT candidates: 12
- vmap opportunities: 5
- GPU-suitable: 6
- Expected speedup: 20-100x

---

### ✅ Code Quality Master

**Priority Findings**:
1. 🔴 **Critical: No error handling** (8 functions)
   - Risk: Silent failures, data corruption
   - Solution: Add try-except blocks
   - Effort: 4 hours
   - Status: URGENT

2. ⭐ **Test Coverage: 65%** (Target: 90%)
   - Missing: Edge cases, integration tests
   - Solution: Generate comprehensive test suite
   - Effort: 1 week
   - Status: HIGH PRIORITY

3. 📋 **Code Duplication: 12%**
   - ~150 lines duplicated in validation
   - Solution: Extract shared module
   - Effort: 4 hours
   - Status: MEDIUM PRIORITY

**Metrics**:
- Maintainability Index: 62/100 → Target 80/100
- Cyclomatic Complexity: 8.5 → Target <10
- Test Coverage: 65% → Target 90%
- Code Duplication: 12% → Target <5%

---

### 🏗️ Systems Architect

**Priority Findings**:
1. 📋 **Architecture: Modular Refactoring**
   - Current: Monolithic structure
   - Proposed: Layered architecture
   - Impact: Better maintainability, scalability
   - Effort: 2 weeks
   - Status: LONG-TERM REFACTOR

2. ⭐ **Caching Strategy**
   - Missing computation caching
   - Solution: Multi-level cache (memory + disk)
   - Impact: 3x speedup (70% hit rate)
   - Effort: 2 days
   - Status: HIGH PRIORITY

3. 📋 **Distributed Computing**
   - Can parallelize 3 workflow stages
   - Solution: Integrate Dask/Ray
   - Impact: 5-10x throughput
   - Effort: 1 week
   - Status: HIGH IMPACT

**Metrics**:
- Scalability Score: 4/10 → Target 8/10
- Module Coupling: High → Target Low
- Resource Utilization: 30% → Target 80%

---

### 🔒 DevOps Security Engineer

**Priority Findings**:
1. ⭐ **CI/CD Optimization**
   - Current: 15 min build time
   - Solution: Caching + parallel jobs
   - Impact: 5 min build time
   - Status: READY FOR IMPLEMENTATION

2. 🔴 **Security: Dependency Vulnerabilities**
   - 5 high-severity CVEs found
   - Solution: Update dependencies
   - Effort: 2 hours
   - Status: URGENT

3. 📋 **Infrastructure as Code**
   - Manual deployment process
   - Solution: Terraform + Ansible
   - Impact: Reproducible infrastructure
   - Effort: 1 week
   - Status: MEDIUM PRIORITY

---

## Meta-Analysis (Orchestrator Synthesis)

### Cross-Cutting Patterns

**Pattern 1: Performance Bottleneck in compute_correlation**
- Identified by: Scientific Computing Master, JAX Pro, Systems Architect
- Convergent recommendation: Vectorize with NumPy, then JIT with JAX
- **Consensus Priority**: ⭐⭐⭐ HIGHEST
- Expected impact: 50x speedup (vectorization) × 10x (JIT) = 500x total

**Pattern 2: Missing Error Handling**
- Identified by: Code Quality Master, DevOps Security Engineer
- Impact: Production stability risk
- **Consensus Priority**: 🔴 CRITICAL
- Must fix before optimizations

**Pattern 3: Test Coverage Gaps**
- Identified by: Code Quality Master, Scientific Computing Master
- Impact: Cannot safely refactor
- **Consensus Priority**: ⭐ HIGH
- Blocks high-risk optimizations

### Conflict Resolutions

**Conflict 1: Memory vs Speed (Caching)**
- Agent A (Scientific): Unlimited cache for 10x speedup
- Agent B (Systems): Memory limited, will OOM
- **Resolution**: LRU cache with 2GB limit → 5x speedup (safe)
- **Rationale**: Memory safety > maximum speed

**Conflict 2: JAX Migration Scope**
- Agent A (JAX Pro): Migrate all 12 functions immediately
- Agent B (Systems): Phase migration to minimize risk
- **Resolution**: Phase 1 (6 functions) → validate → Phase 2 (6 more)
- **Rationale**: Incremental reduces risk

---

## Unified Optimization Strategy

### Phase 1: Critical Fixes (Week 1) - URGENT
```
┌─────────────────────────────────────────┐
│ CRITICAL PATH                           │
├─────────────────────────────────────────┤
│ 1. Add error handling (4 hours)        │
│ 2. Fix security CVEs (2 hours)         │
│ 3. Setup CI quality gates (2 hours)    │
│ 4. Add basic test coverage (2 days)    │
└─────────────────────────────────────────┘
Outcome: Stable, safe codebase
```

### Phase 2: Quick Wins (Week 2) - HIGH IMPACT
```
┌─────────────────────────────────────────┐
│ PERFORMANCE QUICK WINS                  │
├─────────────────────────────────────────┤
│ 1. Vectorize compute_correlation        │
│    → 50x speedup (0.5 day)             │
│                                         │
│ 2. Add JIT to 12 functions             │
│    → 10-100x speedup (2 hours)         │
│                                         │
│ 3. Implement vmap for batches          │
│    → 10-50x speedup (4 hours)          │
│                                         │
│ 4. Add caching layer                    │
│    → 3x speedup (2 days)               │
└─────────────────────────────────────────┘
Expected: 100-500x combined speedup
```

### Phase 3: Major Optimizations (Weeks 3-4) - TRANSFORMATIVE
```
┌─────────────────────────────────────────┐
│ ARCHITECTURAL IMPROVEMENTS              │
├─────────────────────────────────────────┤
│ 1. Modular refactoring (1 week)        │
│ 2. Distributed computing (1 week)      │
│ 3. GPU migration (if available)        │
│ 4. Comprehensive test suite (1 week)   │
└─────────────────────────────────────────┘
Expected: 10x throughput, better maintainability
```

### Phase 4: Long-Term (Months 2-3) - STRATEGIC
```
┌─────────────────────────────────────────┐
│ STRATEGIC REFACTORS                     │
├─────────────────────────────────────────┤
│ 1. Full architecture redesign          │
│ 2. Infrastructure as Code              │
│ 3. Advanced monitoring                  │
│ 4. Documentation overhaul              │
└─────────────────────────────────────────┘
Expected: Production-ready, scalable system
```

---

## Implementation Plan

### Batch 1: Immediate (Day 1-2)
- [ ] Add error handling to 8 critical functions
- [ ] Update dependencies (fix CVEs)
- [ ] Setup CI quality gates

### Batch 2: Quick Wins (Day 3-5)
- [ ] Vectorize compute_correlation with NumPy
- [ ] Add @jit decorators to 12 functions
- [ ] Implement vmap for 5 batch operations
- [ ] Run benchmarks to verify improvements

### Batch 3: Major (Week 2-4)
- [ ] Implement caching strategy
- [ ] Achieve 80% test coverage
- [ ] Begin modular refactoring
- [ ] Setup distributed computing

### Validation Checkpoints
After each batch:
1. ✅ All tests pass
2. ✅ Performance benchmarks meet targets
3. ✅ No regressions introduced
4. ✅ Code review by domain experts

---

## Expected Outcomes

### Performance Improvements
```
Metric                  | Before    | After     | Improvement
------------------------|-----------|-----------|-------------
compute_correlation     | 10.0s     | 0.02s     | 500x ⚡⚡⚡
Overall runtime         | 15.3s     | 0.5s      | 30x  ⚡⚡
Memory usage           | 28GB      | 8GB       | 71%  ↓
Test coverage          | 65%       | 90%       | +25% ✅
Build time             | 15min     | 5min      | 67%  ↓
Maintainability Index  | 62/100    | 85/100    | +37% ✅
```

### Quality Metrics
```
- Code duplication: 12% → 3%
- Cyclomatic complexity: 8.5 → 5.2
- Security vulnerabilities: 5 → 0
- Documentation coverage: 40% → 95%
```

### Business Impact
```
- Development velocity: +40% (less time debugging)
- Production incidents: -80% (better error handling)
- New feature delivery: +60% (cleaner architecture)
- Team satisfaction: +50% (better tools and code quality)
```

---

## Risk Assessment & Mitigation

### High-Risk Items
1. **GPU Migration** (if attempted)
   - Risk: Different numerical results
   - Mitigation: Comprehensive validation, tolerance testing

2. **Architecture Refactor**
   - Risk: Breaking changes, long delivery time
   - Mitigation: Incremental refactor, feature flags

3. **Distributed Computing**
   - Risk: Complexity, debugging difficulty
   - Mitigation: Start with simple parallelization

### Rollback Plan
- All changes in git branches
- Automated rollback on test failure
- Feature flags for major changes
- Database migrations reversible

---

## Agent Collaboration Insights

### Successful Collaborations
1. **Scientific Computing + JAX Pro**
   - Identified compute_correlation as top priority
   - Collaborated on hybrid NumPy→JAX migration
   - Result: 500x speedup strategy

2. **Code Quality + DevOps**
   - Aligned on CI/CD quality gates
   - Collaborated on security fixes
   - Result: Comprehensive quality pipeline

### Conflict Resolution Success
1. **Memory vs Speed Trade-off**
   - Balanced approach achieved 5x speedup safely
   - Avoided potential OOM issues
   - Can revisit when more RAM available

---

## Next Steps

### Immediate Actions (This Week)
1. 🔴 Review and approve optimization strategy
2. 🔴 Create feature branch: `optimization/multi-agent`
3. ⭐ Begin Batch 1 (critical fixes)
4. ⭐ Setup performance benchmarking

### Follow-Up (Next Week)
1. 📋 Execute Batch 2 (quick wins)
2. 📋 Monitor performance improvements
3. 📋 Adjust strategy based on results

### Long-Term (Month 2+)
1. 📊 Architecture refactor
2. 📊 Advanced optimizations
3. 📊 Continuous improvement

---

## Appendix: Detailed Agent Reports

[Full reports from each agent available in separate files]

- `reports/hpc-numerical-coordinator.md`
- `reports/jax-pro.md`
- `reports/code-quality.md`
- `reports/systems-architect.md`
- `reports/devops-security.md`

---

**Report Generated By**: Multi-Agent Orchestrator
**Total Analysis Time**: 45 minutes
**Agents Deployed**: 6
**Optimizations Identified**: 47
**Priority Optimizations**: 15
**Expected ROI**: 30-500x performance improvement
```

---

## Your Task: Execute Multi-Agent Optimization

**Arguments Received**: `$ARGUMENTS`

### Execution Sequence

**Step 1: Project Analysis**
```bash
# Analyze codebase
analyze_project $ARGUMENTS

# Detect languages, frameworks, patterns
detect_stack
```

**Step 2: Agent Selection**
```bash
# Select relevant agents based on project
agents = select_agents(project_analysis)

# Display selected agents
echo "Deploying ${#agents[@]} specialized agents..."
```

**Step 3: Parallel Analysis**
```bash
# Launch all agents in parallel
for agent in "${agents[@]}"; do
    launch_agent_analysis $agent $ARGUMENTS &
done

# Wait for all analyses to complete
wait
```

**Step 4: Meta-Analysis**
```bash
# Synthesize all agent reports
multi_agent_orchestrator synthesize_reports

# Detect conflicts and resolve
resolve_conflicts

# Generate unified strategy
create_optimization_strategy
```

**Step 5: Generate Report**
```bash
# Create comprehensive report
generate_report > multi-agent-optimization-report.md

# Create implementation plan
generate_implementation_plan
```

**Step 6: Execute (if approved)**
```bash
if [ "$AUTO_EXECUTE" = true ]; then
    execute_optimization_plan
    validate_all_changes
    generate_results_report
fi
```

---

## Execution Modes

### 1. Analysis Only (Default)
```bash
/multi-agent-optimize src/
# Analyzes and provides recommendations, no changes
```

### 2. Targeted Agents
```bash
/multi-agent-optimize src/ --agents=jax-pro,hpc-numerical-coordinator
# Deploy only specific agents
```

### 3. Focus Areas
```bash
/multi-agent-optimize src/ --focus=performance,quality
# Prioritize specific optimization dimensions
```

### 4. Parallel Execution
```bash
/multi-agent-optimize src/ --parallel
# Maximum parallelization of agent analyses
```

---

Now execute comprehensive multi-agent optimization with intelligent orchestration! 🎯
