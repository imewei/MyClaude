# Agent Orchestration

Multi-agent workflow coordination with production-ready optimization workflows, systematic agent improvement methodologies, and comprehensive documentation featuring real-world examples with 50-200x performance improvements.

**Version:** 1.0.7 | **Category:** AI & ML | **License:** MIT

[Full Documentation →](https://myclaude.readthedocs.io/en/latest/plugins/agent-orchestration.html) | [Changelog →](./CHANGELOG.md)

## What's New in v1.0.7

This release implements **Opus 4.5 optimization** with enhanced token efficiency and standardized documentation.

### Key Improvements

- **Format Standardization**: All components now include consistent YAML frontmatter with version, maturity, specialization, and description fields
- **Token Efficiency**: 40-50% line reduction through tables over prose, minimal code examples, and structured sections
- **Enhanced Discoverability**: Clear "Use when..." trigger phrases for better Claude Code activation
- **Actionable Checklists**: Task-oriented guidance for common workflows
- **Cross-Reference Tables**: Quick-reference format for delegation and integration patterns


## Overview

The Agent Orchestration plugin provides production-ready tools for coordinating multiple AI agents, optimizing code performance, and systematically improving agent capabilities. With v1.0.2, both slash commands have been restructured with executable workflows, comprehensive documentation, and real-world case studies demonstrating 50-208x performance improvements.

### Key Features

✨ **Advanced Agent Coordination**
- Multi-agent workflow orchestration with DAG-based execution
- Intelligent task allocation and team assembly
- Inter-agent communication protocols (message broker, request/response, broadcast)
- Fault-tolerant execution with retry logic and graceful degradation

🎯 **Performance Optimization**
- Comprehensive metrics collection (P50/P95/P99 latencies, success rates)
- Multi-tier caching strategies (LRU, hot/warm/cold tiers)
- Load balancing algorithms (round-robin, least-loaded, weighted)
- Performance profiling and bottleneck analysis

🧠 **Intelligent Decision Making**
- Chain-of-thought reasoning for complex orchestration decisions
- Constitutional AI principles for self-correction
- Extensive few-shot examples demonstrating best practices
- Clear triggering criteria and anti-patterns

## Recent Updates (v1.0.2)

### /multi-agent-optimize Command
- ✅ Added executable workflow logic for --mode=scan (quick bottleneck detection)
- ✅ Implemented YAML frontmatter with graceful fallback for missing agents
- ✅ Reduced token usage by 19% (382→311 lines)
- ✅ Created comprehensive documentation ecosystem (12+ external docs)
- ✅ Added 3 real-world case studies with measured results:
  - MD Simulation: 4.5 hours → 1.3 minutes (208x speedup)
  - JAX Training: 8 hours → 9.6 minutes (50x speedup)
  - API Performance: 120 → 1,200 req/sec (10x throughput)
- **Documentation**: 3 domain-specific pattern libraries (scientific, ML, web)

### /improve-agent Command
- ✅ Added executable workflow logic for --mode=check (health assessment)
- ✅ Implemented YAML frontmatter with execution modes
- ✅ Reduced token usage by 20% (291→234 lines)
- ✅ Created 4 phase-specific methodology guides
- ✅ Added customer support case study: 72% → 91% success rate (26% improvement)
- **Methodology**: Complete 4-phase improvement workflow (analysis, prompts, testing, deployment)

**Impact Metrics:**
- 19-20% token reduction across both commands (~512 tokens saved per invocation)
- 4 comprehensive case studies with real measured results
- 12+ external documentation files for maintainability
- Executable logic for immediate usability

## Agents (2)

### context-manager (v1.0.3)

**Status:** active | **Maturity:** 95%

Elite AI context engineering specialist mastering dynamic context management, vector databases, knowledge graphs, and intelligent memory systems. Orchestrates context across multi-agent workflows, enterprise AI systems, and long-running projects with 2024/2025 best practices.

**Key Capabilities:**
- Advanced RAG system architecture with multi-tier caching
- Vector database optimization (Pinecone, Weaviate, ChromaDB)
- Knowledge graph construction and traversal
- Session context management with hierarchical compression
- Multi-agent context coordination

**New in v1.0.1:**
- Triggering criteria with decision tree
- Constitutional AI framework with safety checks
- Chain-of-thought reasoning for architecture decisions
- 3 comprehensive examples (multi-agent coordination, RAG optimization, context overflow)

### multi-agent-orchestrator (v1.0.3)

**Status:** active | **Maturity:** 95%

Multi-agent orchestrator specializing in workflow coordination and distributed systems. Expert in agent team assembly and task allocation for scalable collaboration. Delegates all domain-specific work to specialist agents.

**Key Capabilities:**
- DAG-based workflow orchestration with parallel execution
- Agent capability matching and team assembly
- Complex dependency management and synchronization
- Intelligent task allocation across specialized agents
- Fault-tolerant execution with automatic retries

**New in v1.0.1:**
- 5-step chain-of-thought orchestration process
- 5 Constitutional AI principles for validation
- 4 comprehensive few-shot examples (good/bad orchestration, dependency handling, unclear requirements)
- Advanced orchestration patterns (pipeline, fan-out/fan-in, conditional, iterative)
- Anti-patterns section with common mistakes

## Commands (2)

### `/multi-agent-optimize` (v1.0.3)

**Status:** active | **Lines:** 311 (-19%)

Multi-agent code optimization with executable workflows (scan/analyze/apply modes), graceful fallback for missing agents, and comprehensive documentation with real case studies showing 50-208x speedups.

**Syntax:**
```bash
# Quick scan (2-5 min): Identify bottlenecks
/multi-agent-optimize src/ --mode=scan

# Deep analysis (10-30 min): Generate optimization recommendations
/multi-agent-optimize src/simulation/ --mode=analyze --focus=scientific --parallel

# Apply optimizations: Implement changes with validation
/multi-agent-optimize src/ --mode=apply --quick-wins
```

**New in v1.0.2:**
- ✅ Executable workflow logic for --mode=scan with step-by-step instructions
- ✅ YAML frontmatter with graceful fallback messages for 7 conditional agents
- ✅ 19% token reduction through external documentation references
- ✅ 3 real-world case studies: MD simulation (208x), JAX training (50x), API (10x)
- ✅ Domain-specific pattern libraries (scientific, ML, web performance)

### `/improve-agent` (v1.0.3)

**Status:** active | **Lines:** 234 (-20%)

Systematic agent improvement through 4-phase methodology (analysis, prompt engineering, testing, deployment) with executable workflows and A/B testing framework.

**Syntax:**
```bash
# Health check (2-5 min): Identify top 3 issues
/improve-agent customer-support --mode=check

# Single phase execution (10-30 min): Targeted improvements
/improve-agent customer-support --phase=2 --focus=tool-selection

# Full optimization (1-2 hours): Complete 4-phase cycle
/improve-agent customer-support --mode=optimize
```

**New in v1.0.2:**
- ✅ Executable workflow logic for --mode=check with Task tool integration
- ✅ YAML frontmatter with execution modes and conditional agent triggering
- ✅ 20% token reduction through modular documentation structure
- ✅ 4 phase-specific guides (analysis, prompts, testing, deployment)
- ✅ Customer support case study: 72% → 91% success rate (+26%)

## Skills (2)

### multi-agent-coordination

Production-ready multi-agent coordination with team assembly, workflow orchestration, DAG-based execution, inter-agent messaging, and fault-tolerant coordination patterns.

**Use this skill when:**
- Coordinating multiple specialized AI agents across technical domains
- Designing DAG-based workflow orchestration systems
- Implementing agent team assembly with capability matching
- Building inter-agent communication protocols
- Orchestrating parallel agent execution with synchronization
- Implementing fault-tolerant multi-agent systems

**Includes:**
- Agent capability matching and registry
- Workflow engine with DAG validation
- Message broker for inter-agent communication
- Team builder for optimal agent selection
- Complete Python implementations

### agent-performance-optimization

Comprehensive agent performance monitoring, optimization, and tuning including metrics collection, caching strategies (LRU, multi-tier), load balancing (round-robin, least-loaded, weighted), and performance profiling.

**Use this skill when:**
- Analyzing agent execution bottlenecks
- Implementing metrics collection (P50/P95/P99 latencies)
- Building multi-tier caching for expensive operations
- Setting up load balancing across agent instances
- Tracking success/failure rates and error patterns
- Scaling agent systems for production workloads

**Includes:**
- MetricsCollector with percentile calculations
- LRU cache with TTL support
- Multi-tier caching (hot/warm/cold)
- Load balancer with multiple strategies
- Complete Python implementations

## Quick Start

### Installation

```bash
# Add the marketplace
/plugin marketplace add imewei/MyClaude

# Install the plugin
/plugin install agent-orchestration@scientific-computing-workflows
```

### Basic Usage

**1. Using the Context Manager**

Ask Claude to design a context management system:
```
Design a RAG system with multi-tier caching for 1M+ documents using the @context-manager agent
```

**2. Using the Multi-Agent Orchestrator**

Coordinate multiple agents for a complex task:
```
Build a production ML system with web interface, API, training pipeline, deployment, and monitoring
```

The orchestrator will automatically:
- Analyze task complexity (5+ domains)
- Select appropriate specialist agents
- Design execution workflow with dependencies
- Coordinate parallel execution where possible
- Handle failures with fallback strategies

**3. Optimizing Agent Performance**

```bash
/multi-agent-optimize src/ --focus=performance --parallel
```

**4. Improving Agent Definitions**

```bash
/improve-agent plugins/my-plugin/agents/my-agent.md
```

## Use Cases

### Code Performance Optimization
- **Molecular Dynamics**: Optimized LAMMPS preprocessing 208x (4.5h → 1.3min)
  - cKDTree neighbor search, NumPy vectorization, Numba JIT
  - See: `docs/examples/md-simulation-optimization.md`
- **JAX Training Pipeline**: Accelerated neural network training 50x (8h → 9.6min)
  - @jit compilation, GPU data pipeline, Optax optimizer, vmap batching
  - See: `docs/examples/jax-training-optimization.md`
- **REST API**: Increased throughput 10x (120 → 1,200 req/sec)
  - N+1 query fixes, Redis caching, connection pooling, gzip compression
  - See: `docs/examples/api-performance-optimization.md`

### Agent Capability Improvement
- **Customer Support Agent**: Improved success rate 26% (72% → 91%)
  - Few-shot examples, chain-of-thought reasoning, Constitutional AI
  - Reduced user corrections 48% (2.3 → 1.2 per task)
  - See: `docs/examples/customer-support-optimization.md`

### Pattern Libraries
- **Scientific Computing**: NumPy/JAX/SciPy optimization patterns (10 patterns)
  - Vectorization, JIT compilation, broadcasting, sparse matrices
- **ML Training**: PyTorch/TensorFlow optimization (5 patterns)
  - Mixed precision, gradient accumulation, DataLoader tuning, quantization
- **Web Performance**: Backend/frontend optimization (5 patterns)
  - N+1 elimination, Redis caching, code splitting, image optimization

## Integration

This plugin integrates with:

**AI/ML Plugins:**
- `deep-learning` - Neural network architecture coordination
- `machine-learning` - ML pipeline orchestration
- `jax-implementation` - JAX optimization coordination

**Development Plugins:**
- `backend-development` - API and microservices coordination
- `frontend-mobile-development` - UI development coordination
- `python-development` - Python project orchestration

**Quality Plugins:**
- `comprehensive-review` - Multi-agent code review
- `quality-engineering` - QA workflow coordination
- `unit-testing` - Test generation across agents

See [full documentation](https://myclaude.readthedocs.io/en/latest/plugins/agent-orchestration.html) for detailed integration patterns.

## Architecture

```
agent-orchestration/
├── agents/
│   ├── context-manager.md (v1.0.3)
│   │   ├── Triggering criteria
│   │   ├── Constitutional AI framework
│   │   ├── Chain-of-thought reasoning
│   │   └── 3 comprehensive examples
│   └── multi-agent-orchestrator.md (v1.0.3)
│       ├── 5-step orchestration process
│       ├── 5 Constitutional AI principles
│       ├── 4 comprehensive examples
│       └── Advanced patterns & anti-patterns
├── commands/
│   ├── multi-agent-optimize.md (v1.0.3)
│   │   ├── Executable workflow logic (scan/analyze/apply)
│   │   ├── YAML frontmatter with graceful fallbacks
│   │   └── 3 real-world case studies (50-208x speedups)
│   └── improve-agent.md (v1.0.3)
│       ├── Executable workflow logic (check/phase/optimize)
│       ├── 4 phase-specific methodology guides
│       └── Customer support case study (26% improvement)
├── docs/
│   ├── Phase guides (4): analysis, prompts, testing, deployment
│   ├── Pattern libraries (3): scientific, ML, web performance
│   └── examples/ (4 case studies with measured results)
└── skills/
    ├── multi-agent-coordination/ (v1.0.3)
    │   ├── Team assembly (AgentRegistry, TeamBuilder)
    │   ├── Workflow orchestration (WorkflowEngine, DAG)
    │   └── Inter-agent messaging (MessageBroker)
    └── agent-performance-optimization/ (v1.0.3)
        ├── Metrics collection (PerformanceMetrics)
        ├── Caching (LRUCache, TieredCache)
        └── Load balancing (LoadBalancer)
```

## Best Practices

### Agent Orchestration
1. **Use orchestration for 5+ agent tasks** - Direct invocation is more efficient for simple tasks
2. **Map dependencies clearly** - Create visual dependency graphs before execution
3. **Maximize parallelization** - Run independent tasks concurrently
4. **Plan for failures** - Implement fallback strategies and retry logic
5. **Validate before execution** - Use Constitutional AI principles for self-critique

### Performance Optimization
1. **Track percentiles, not just averages** - P95/P99 reveal tail latencies
2. **Implement multi-tier caching** - Hot/warm/cold tiers for different access patterns
3. **Use appropriate load balancing** - Least-loaded for variable tasks, round-robin for uniform
4. **Monitor error patterns** - Track error types for root cause analysis
5. **Size caches appropriately** - Balance memory usage with hit rate

### Context Management
1. **Design before implementing** - Use chain-of-thought reasoning for architecture decisions
2. **Implement safety checks** - Constitutional AI validation prevents production issues
3. **Test with realistic data** - Use representative datasets for performance testing
4. **Monitor context window** - Implement hierarchical compression for large contexts
5. **Document integration points** - Clear handoffs between agents prevent confusion

## Performance Metrics

**All Components (v1.0.3):**

**Agents:**

**Context-Manager:**
- Maturity: 65% → 95% (+30%)
- Added: 6 critical sections
- Examples: 3 comprehensive scenarios
- Expected: More accurate architecture decisions, better safety checks

**Multi-Agent-Orchestrator:**
- Maturity: 78% → 95% (+17%)
- Added: 8 critical sections
- Examples: 4 comprehensive scenarios
- Expected: 25-30% reduction in over-orchestration, 40% better dependency identification

**Commands:**

**multi-agent-optimize:**
- Token reduction: 19% (382→311 lines, ~284 tokens saved)
- Documentation: 3 pattern libraries + 3 case studies
- Case studies: 50-208x speedups with measured results

**improve-agent:**
- Token reduction: 20% (291→234 lines, ~228 tokens saved)
- Documentation: 4 phase guides + 1 case study
- Case study: 26% success rate improvement

**Skills:**
- multi-agent-coordination: Production-ready coordination patterns
- agent-performance-optimization: Comprehensive monitoring and tuning

## Documentation

### Plugin Documentation
- [Full Plugin Documentation](https://myclaude.readthedocs.io/en/latest/plugins/agent-orchestration.html)
- [Changelog](./CHANGELOG.md) - Version history and improvements
- [Agent Definitions](./agents/) - Detailed agent specifications

### Build Documentation Locally

```bash
cd docs/
make html
open _build/html/index.html
```

## Contributing

We welcome contributions! To improve this plugin:

1. **Submit examples** - Real-world usage scenarios help improve agents
2. **Report issues** - Flag cases where agents underperform
3. **Suggest improvements** - Propose new capabilities or refinements
4. **Share performance data** - Metrics help optimize agent behavior

See the [contribution guidelines](https://myclaude.readthedocs.io/en/latest/contributing.html) for details.

## Version History

- **v1.0.3** (2025-11-06) - Version consolidation release: corrected version numbering (v1.0.2 → v1.0.1 for agents), maintained all v1.0.2 improvements, ensured version consistency across documentation
- **v1.0.2** (2025-11-06) - Command optimization release: restructured both slash commands with executable logic, 19-20% token reduction, 12+ external docs, 4 real case studies
- **v1.0.1** (2025-01-29) - Agent prompt engineering release: major improvements to both agents (context-manager, multi-agent-orchestrator) with Constitutional AI, chain-of-thought reasoning, and comprehensive examples
- **v1.0.0** - Initial release with basic agent definitions

See [CHANGELOG.md](./CHANGELOG.md) for detailed version history.

## License

MIT License - see [LICENSE](../../LICENSE) for details

## Author

Wei Chen

---

*For questions, issues, or feature requests, please visit the [plugin documentation](https://myclaude.readthedocs.io/en/latest/plugins/agent-orchestration.html).*
