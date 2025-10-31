# Agent Orchestration

Multi-agent workflow coordination, distributed systems, and intelligent task allocation for complex multi-domain projects with advanced prompt engineering and Constitutional AI frameworks.

**Version:** 2.0.0 | **Category:** AI & ML | **License:** MIT

[Full Documentation →](https://myclaude.readthedocs.io/en/latest/plugins/agent-orchestration.html) | [Changelog →](./CHANGELOG.md)

## Overview

The Agent Orchestration plugin provides production-ready tools for coordinating multiple AI agents, optimizing agent performance, and building sophisticated multi-agent workflows. With v2.0.0, both core agents have been enhanced with Constitutional AI frameworks, chain-of-thought reasoning, and comprehensive few-shot examples for superior performance.

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

## Recent Updates (v2.0.0)

### Context-Manager Agent
- ✅ Added triggering criteria section with clear use cases and anti-patterns
- ✅ Implemented Constitutional AI framework with safety guardrails
- ✅ Added chain-of-thought reasoning framework (4 steps)
- ✅ Included 3 comprehensive few-shot examples
- ✅ Enhanced behavioral traits and output format standards
- **Maturity**: 65% → 95% (+30% improvement)

### Multi-Agent-Orchestrator Agent
- ✅ Added 5-step chain-of-thought orchestration process
- ✅ Implemented 5 Constitutional AI principles for validation
- ✅ Included 4 comprehensive few-shot examples
- ✅ Added 3 standardized output format templates
- ✅ Documented 4 advanced orchestration patterns
- ✅ Included anti-patterns section with common mistakes
- ✅ Enhanced decision framework and delegation strategy
- **Maturity**: 78% → 95% (+17% improvement)

**Expected Improvements:**
- 25-30% reduction in over-orchestration
- 40% improvement in dependency identification
- 50% better error handling
- 35% clearer user communication

## Agents (2)

### context-manager (v2.0.0)

**Status:** active | **Maturity:** 95%

Elite AI context engineering specialist mastering dynamic context management, vector databases, knowledge graphs, and intelligent memory systems. Orchestrates context across multi-agent workflows, enterprise AI systems, and long-running projects with 2024/2025 best practices.

**Key Capabilities:**
- Advanced RAG system architecture with multi-tier caching
- Vector database optimization (Pinecone, Weaviate, ChromaDB)
- Knowledge graph construction and traversal
- Session context management with hierarchical compression
- Multi-agent context coordination

**New in v2.0.0:**
- Triggering criteria with decision tree
- Constitutional AI framework with safety checks
- Chain-of-thought reasoning for architecture decisions
- 3 comprehensive examples (multi-agent coordination, RAG optimization, context overflow)

### multi-agent-orchestrator (v2.0.0)

**Status:** active | **Maturity:** 95%

Multi-agent orchestrator specializing in workflow coordination and distributed systems. Expert in agent team assembly and task allocation for scalable collaboration. Delegates all domain-specific work to specialist agents.

**Key Capabilities:**
- DAG-based workflow orchestration with parallel execution
- Agent capability matching and team assembly
- Complex dependency management and synchronization
- Intelligent task allocation across specialized agents
- Fault-tolerant execution with automatic retries

**New in v2.0.0:**
- 5-step chain-of-thought orchestration process
- 5 Constitutional AI principles for validation
- 4 comprehensive few-shot examples (good/bad orchestration, dependency handling, unclear requirements)
- Advanced orchestration patterns (pipeline, fan-out/fan-in, conditional, iterative)
- Anti-patterns section with common mistakes

## Commands (2)

### `/multi-agent-optimize`

**Status:** active

Coordinate multiple specialized agents for code optimization and review tasks with intelligent orchestration, resource allocation, and multi-dimensional analysis.

**Syntax:**
```bash
/multi-agent-optimize <target-path> [--agents=agent1,agent2] [--focus=performance,quality,research,scientific] [--parallel]
```

**Features:**
- Multi-agent team coordination
- Performance and quality optimization
- Scientific computing support
- Parallel execution option

### `/improve-agent`

**Status:** active

Systematically improve agent definitions through performance analysis, prompt engineering, and continuous iteration.

**Syntax:**
```bash
/improve-agent <path-to-agent.md>
```

**Features:**
- Performance baseline analysis
- Chain-of-thought enhancement
- Few-shot example optimization
- Constitutional AI integration
- A/B testing framework

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

### Complex Multi-Domain Projects
- Full-stack application development with ML components
- Enterprise system architecture with multiple services
- Scientific computing workflows across HPC + ML + visualization

### Performance Optimization
- Identifying and resolving agent bottlenecks
- Implementing caching strategies for expensive operations
- Scaling multi-agent systems for production loads

### Context Management
- RAG systems with vector databases and knowledge graphs
- Long-running multi-agent sessions with context preservation
- Enterprise AI systems with distributed context

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
│   ├── context-manager.md (v2.0.0)
│   │   ├── Triggering criteria
│   │   ├── Constitutional AI framework
│   │   ├── Chain-of-thought reasoning
│   │   └── 3 comprehensive examples
│   └── multi-agent-orchestrator.md (v2.0.0)
│       ├── 5-step orchestration process
│       ├── 5 Constitutional AI principles
│       ├── 4 comprehensive examples
│       └── Advanced patterns & anti-patterns
├── commands/
│   ├── multi-agent-optimize.md
│   └── improve-agent.md
└── skills/
    ├── multi-agent-coordination/
    │   ├── Team assembly (AgentRegistry, TeamBuilder)
    │   ├── Workflow orchestration (WorkflowEngine, DAG)
    │   └── Inter-agent messaging (MessageBroker)
    └── agent-performance-optimization/
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

**Context-Manager v2.0.0:**
- Maturity: 65% → 95% (+30%)
- Added: 6 critical sections
- Examples: 3 comprehensive scenarios
- Expected: More accurate architecture decisions, better safety checks

**Multi-Agent-Orchestrator v2.0.0:**
- Maturity: 78% → 95% (+17%)
- Added: 8 critical sections
- Examples: 4 comprehensive scenarios
- Expected: 25-30% reduction in over-orchestration, 40% better dependency identification

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

- **v2.0.0** (2025-01-29) - Major prompt engineering improvements for both agents
- **v1.0.0** - Initial release with basic agent definitions

See [CHANGELOG.md](./CHANGELOG.md) for detailed version history.

## License

MIT License - see [LICENSE](../../LICENSE) for details

## Author

Wei Chen

---

*For questions, issues, or feature requests, please visit the [plugin documentation](https://myclaude.readthedocs.io/en/latest/plugins/agent-orchestration.html).*
