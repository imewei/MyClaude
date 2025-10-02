---
title: "Multi-Agent Optimize"
description: "Multi-agent system for code optimization and review using specialized agents"
category: optimization
subcategory: multi-agent
complexity: advanced
argument-hint: "[--mode=optimize|review|hybrid|research] [--agents=all|core|scientific|ai|engineering|domain-specific] [--focus=performance|security|quality|architecture|research|innovation] [--implement] [--orchestrate] [target]"
allowed-tools: Read, Write, Edit, Grep, Glob, TodoWrite, Bash, Task
model: inherit
tags: multi-agent, optimization, review, performance, security, quality
dependencies: []
related: [optimize, think-ultra, check-code-quality, refactor-clean, debug]
workflows: [multi-agent-analysis, code-review, optimization-pipeline]
version: "2.0"
last-updated: "2025-09-28"
---

# Multi-Agent Optimize

Coordinate multiple specialized agents for code optimization and review tasks.

## Quick Start

```bash
# Basic multi-agent optimization
/multi-agent-optimize src/

# Performance-focused optimization
/multi-agent-optimize --mode=optimize --focus=performance

# Hybrid analysis with core agents
/multi-agent-optimize --agents=core --mode=hybrid myproject/

# Complete ecosystem optimization
/multi-agent-optimize --agents=all --orchestrate complex_project/
```

## Usage

```bash
/multi-agent-optimize [options] [target]
```

**Parameters:**
- `options` - Agent selection, operation mode, and focus configuration
- `target` - File, directory, or project to optimize (moved to end for better UX)

## Options

- `--mode=<mode>`: Operation mode (optimize, review, hybrid, research)
- `--agents=<agents>`: Agent selection (all, core, scientific, ai, engineering, domain-specific)
- `--focus=<focus>`: Focus area (performance, security, quality, architecture, research, innovation)
- `--parallel`: Run agents in parallel for faster execution
- `--interactive`: Interactive mode for agent selection and coordination
- `--implement`: Automatically implement optimization recommendations using agent insights
- `--validate`: Validate implementation results through testing and verification
- `--rollback`: Enable rollback capability for failed implementations
- `--orchestrate`: Enable advanced multi-agent orchestration with workflow coordination

## Operation Modes

- `optimize` - Focus on performance and efficiency improvements
- `review` - Focus on code quality and best practices review
- `hybrid` - Combined optimization and review (default)
- `research` - Focus on research-grade analysis, innovation, and breakthrough discovery

## Personal Agent Architecture

### Multi-Agent Orchestration
- **`multi-agent-orchestrator`** - Workflow coordination, resource management, and intelligent task allocation
- **`command-systems-engineer`** - Command system optimization and workflow engineering

### Scientific Computing & Research
- **`scientific-computing-master`** - Python, Julia/SciML, JAX ecosystem optimization and numerical computing
- **`research-intelligence-master`** - Advanced research analysis, innovation discovery, and knowledge synthesis
- **`jax-pro`** - JAX-specific optimization, GPU acceleration, and scientific ML workflows
- **`neural-networks-master`** - Deep learning model optimization and neural network architecture
- **`advanced-quantum-computing-expert`** - Quantum computing optimization and quantum-classical hybrid systems

### Engineering & Architecture
- **`ai-systems-architect`** - AI system design, scalability, and intelligent architecture
- **`systems-architect`** - System design, architecture patterns, and scalability optimization
- **`fullstack-developer`** - Full-stack optimization, web performance, and application architecture
- **`devops-security-engineer`** - DevSecOps optimization, infrastructure security, and CI/CD enhancement

### Quality & Documentation
- **`code-quality-master`** - Code quality analysis, testing strategies, and automated quality assurance
- **`documentation-architect`** - Technical documentation, educational content, and knowledge management

### Domain Specialists
- **`data-professional`** - Data engineering, analytics optimization, and data pipeline enhancement
- **`ai-ml-specialist`** - Machine learning optimization, model performance, and ML pipeline enhancement
- **`visualization-interface-master`** - UI/UX optimization, visualization design, and interface performance
- **`database-workflow-engineer`** - Database optimization, query performance, and data workflow engineering

### Scientific Domain Experts
- **`correlation-function-expert`** - Statistical analysis and correlation optimization
- **`neutron-soft-matter-expert`** - Neutron scattering and soft matter simulation optimization
- **`xray-soft-matter-expert`** - X-ray analysis and soft matter computational optimization
- **`nonequilibrium-stochastic-expert`** - Nonequilibrium systems and stochastic process optimization
- **`scientific-code-adoptor`** - Legacy scientific code modernization and optimization

## Focus Areas

- `performance` - Speed, memory usage, and efficiency optimization
- `security` - Security vulnerabilities and hardening
- `quality` - Code quality, maintainability, and best practices
- `architecture` - System design and structural improvements
- `research` - Research-grade analysis, innovation, and breakthrough discovery
- `innovation` - Cutting-edge optimization techniques and emerging technologies

## Agent Selection Strategies

### `all` - Complete Multi-Agent Ecosystem
Activates all available personal agents for comprehensive analysis across all domains. Ideal for complex, multi-dimensional optimization challenges requiring diverse expertise.

### `core` - Essential Multi-Agent Team
- `multi-agent-orchestrator` - Workflow coordination
- `code-quality-master` - Quality analysis
- `systems-architect` - Architecture optimization
- `scientific-computing-master` - Performance optimization

### `scientific` - Scientific Computing Focus
- `scientific-computing-master` - Numerical computing and scientific workflows
- `jax-pro` - JAX ecosystem optimization
- `research-intelligence-master` - Research methodology
- `neural-networks-master` - ML/AI optimization
- `advanced-quantum-computing-expert` - Quantum computing

### `ai` - AI/ML Optimization Team
- `ai-systems-architect` - AI system design
- `ai-ml-specialist` - Machine learning optimization
- `neural-networks-master` - Deep learning architecture
- `data-professional` - Data pipeline optimization
- `jax-pro` - Scientific ML workflows

### `engineering` - Software Engineering Focus
- `systems-architect` - System design
- `fullstack-developer` - Application optimization
- `devops-security-engineer` - Infrastructure and security
- `code-quality-master` - Quality engineering
- `database-workflow-engineer` - Data optimization

### `domain-specific` - Specialized Domain Experts
Activates domain-specific experts based on codebase analysis:
- `correlation-function-expert` - Statistical computing
- `neutron-soft-matter-expert` - Neutron scattering simulations
- `xray-soft-matter-expert` - X-ray analysis workflows
- `nonequilibrium-stochastic-expert` - Stochastic processes
- `scientific-code-adoptor` - Legacy code modernization

## Examples

```bash
# Complete multi-agent ecosystem analysis
/multi-agent-optimize myapp/ --mode=hybrid --agents=all --orchestrate

# Scientific computing optimization with specialized agents
/multi-agent-optimize simulation.py --mode=optimize --agents=scientific --focus=performance

# AI/ML system optimization
/multi-agent-optimize ml_pipeline/ --mode=optimize --agents=ai --implement

# Security-focused review with DevSecOps expertise
/multi-agent-optimize webapp/ --mode=review --focus=security --agents=engineering

# Research-grade analysis with breakthrough discovery
/multi-agent-optimize research_code/ --mode=research --agents=scientific --focus=innovation

# Interactive agent selection with orchestration
/multi-agent-optimize myproject/ --interactive --orchestrate --parallel

# Domain-specific optimization for scientific computing
/multi-agent-optimize neutron_analysis/ --agents=domain-specific --focus=performance

# Full ecosystem optimization with implementation
/multi-agent-optimize codebase/ --mode=hybrid --agents=all --implement --validate

# JAX-specific optimization with scientific computing focus
/multi-agent-optimize jax_model.py --agents=scientific --focus=performance --implement

# Architecture review with engineering team
/multi-agent-optimize system/ --mode=review --focus=architecture --agents=engineering

# Legacy code modernization with scientific focus
/multi-agent-optimize legacy_fortran/ --agents=scientific --mode=optimize --implement

# Multi-agent research analysis with innovation focus
/multi-agent-optimize novel_algorithm/ --mode=research --agents=all --focus=innovation --orchestrate
```

## Advanced Multi-Agent Orchestration (23 Personal Agents)

### Phase 1: Intelligent Agent Activation
**Agent Selection Algorithm:**
- **Codebase Analysis**: Automatic detection of relevant domains and technologies
- **Smart Agent Matching**: Map discovered technologies to appropriate agents from the 23-agent library
- **Capability Assessment**: Evaluate which combinations of agents provide optimal coverage
- **Resource Optimization**: Balance comprehensive analysis with execution efficiency

**Agent Activation Patterns:**
- **Parallel Streams**: Group related agents for concurrent analysis
- **Sequential Specialization**: Chain agents for dependent analysis workflows
- **Cross-Domain Synthesis**: Coordinate agents across different expertise domains

### Phase 2: Distributed Analysis
**Multi-Agent Coordination:**
- **`multi-agent-orchestrator`** manages workflow coordination and resource allocation
- **Domain Clusters**: Scientific, AI/ML, Engineering, Quality, and Specialized domain groups
- **Real-time Communication**: Inter-agent communication for shared insights and coordination
- **Conflict Resolution**: Automatic resolution of conflicting recommendations

**Analysis Dimensions (23-Agent Coverage):**
- **Scientific Computing**: Numerical accuracy, performance, research methodology
- **AI/ML Systems**: Model optimization, data pipelines, ML workflows
- **Software Engineering**: Architecture, quality, security, performance
- **Specialized Domains**: Quantum computing, soft matter, stochastic processes, visualization
- **Infrastructure**: DevOps, databases, command systems, workflow engineering

### Phase 3: Intelligent Synthesis
**Multi-Dimensional Synthesis:**
- **Cross-Agent Correlation**: Identify patterns and synergies across all 23 agent analyses
- **Impact Assessment**: Weighted scoring based on agent expertise and recommendation confidence
- **Dependency Mapping**: Understand implementation dependencies across recommendations
- **Risk Analysis**: Evaluate implementation risks from multiple expert perspectives

**Synthesis Algorithms:**
- **Consensus Building**: Identify areas of agent agreement for high-confidence recommendations
- **Expertise Weighting**: Prioritize recommendations based on agent domain expertise
- **Innovation Discovery**: Highlight novel insights that emerge from multi-agent collaboration

### Phase 4: Strategic Implementation Planning
**23-Agent Implementation Strategy:**
- **Multi-Track Planning**: Parallel implementation tracks for different optimization domains
- **Agent-Guided Implementation**: Specific agents oversee implementation in their domains
- **Quality Gates**: Multi-agent validation at each implementation stage
- **Rollback Coordination**: Distributed rollback capabilities with agent-specific recovery plans

### Phase 5: 23-Agent Coordinated Implementation (--implement)
**Advanced Auto-Implementation Process:**
- **Multi-Agent Priority Assessment**: Consensus-based ranking using all 23 agents' expertise
- **Domain-Specific Risk Analysis**: Each agent evaluates risks in their domain of expertise
- **Distributed Backup Strategy**: Agent-coordinated backup of domain-specific components
- **Orchestrated Implementation**: `multi-agent-orchestrator` coordinates implementation across domains
- **Real-Time Agent Validation**: Continuous validation by relevant agents during implementation
- **Multi-Track Implementation**: Parallel implementation tracks managed by different agent groups
- **Cross-Agent Quality Gates**: Multi-agent approval required for critical changes
- **Intelligent Rollback**: Agent-specific rollback procedures with cross-domain impact analysis
- **Comprehensive Performance Measurement**: Multi-dimensional performance validation by specialized agents
- **Innovation Integration**: Implementation of breakthrough discoveries from agent collaboration

**Agent-Specific Implementation Roles:**
- **Scientific Agents**: Numerical accuracy validation and scientific computing optimization
- **Engineering Agents**: Code quality, architecture, and system reliability
- **AI/ML Agents**: Model performance, data pipeline optimization, and ML workflow enhancement
- **Domain Specialists**: Specialized optimization in quantum, soft matter, and stochastic domains
- **Infrastructure Agents**: DevOps, security, and system-level optimization coordination

## 23-Agent System Output

### Comprehensive Analysis Reports
- **Multi-Agent Analysis Matrix** - Detailed findings from all activated agents (up to 23 agents)
- **Cross-Domain Recommendations** - Prioritized improvements synthesized across all agent domains
- **Agent Consensus Report** - Areas of agreement and conflict resolution across agents
- **Innovation Discoveries** - Novel insights and breakthrough opportunities identified through agent collaboration
- **Domain-Specific Deep Dives** - Specialized analysis from domain expert agents

### Strategic Implementation Planning
- **Multi-Track Implementation Roadmap** - Parallel implementation strategies across different domains
- **Agent-Guided Action Plans** - Step-by-step guidance with agent oversight assignments
- **Risk-Benefit Analysis** - Multi-agent risk assessment with mitigation strategies
- **Quick Wins Portfolio** - Immediate improvements categorized by domain and impact
- **Long-term Innovation Strategy** - Research-grade strategic improvements for breakthrough potential

### With --implement (23-Agent Coordination)
- **Orchestrated Implementation Report** - Detailed summary of changes applied by each agent domain
- **Multi-Agent Validation Results** - Comprehensive validation from all relevant agents
- **Domain-Specific Performance Metrics** - Performance improvements measured by specialized agents
- **Cross-Agent Quality Assurance** - Quality validation reports from all agent perspectives
- **Intelligent Rollback Documentation** - Agent-specific rollback procedures and coordination plans
- **Innovation Implementation Summary** - Successfully implemented breakthrough discoveries

### With --orchestrate
- **Agent Coordination Logs** - Detailed logs of inter-agent communication and decision making
- **Resource Utilization Reports** - Efficiency metrics for 23-agent coordination
- **Workflow Optimization Insights** - Improvements to multi-agent collaboration patterns
- **Emergent Intelligence Reports** - Insights that emerged from agent collaboration beyond individual agent capabilities

## Advanced 23-Agent Workflows

### Complete Ecosystem Analysis & Optimization
```bash
# 1. Full 23-agent ecosystem analysis
/multi-agent-optimize codebase/ --mode=hybrid --agents=all --orchestrate --parallel

# 2. Implement orchestrated optimizations across all domains
/multi-agent-optimize codebase/ --implement --validate --agents=all --orchestrate

# 3. Comprehensive verification with double-check
/double-check "23-agent optimization results" --deep-analysis --auto-complete
```

### Scientific Computing Research Pipeline
```bash
# 1. Research-grade analysis with scientific agents
/multi-agent-optimize research_simulation/ --mode=research --agents=scientific --focus=innovation

# 2. Apply scientific computing optimizations
/multi-agent-optimize research_simulation/ --implement --agents=scientific --validate

# 3. Validate with scientific testing and documentation
/run-all-tests --scientific --reproducible --agents=scientific
/update-docs research_simulation/ --type=research --agents=all
```

### AI/ML System Optimization Pipeline
```bash
# 1. AI/ML system analysis with specialized agents
/multi-agent-optimize ml_pipeline/ --mode=optimize --agents=ai --focus=performance --orchestrate

# 2. Implement AI/ML optimizations with validation
/multi-agent-optimize ml_pipeline/ --implement --agents=ai --validate --parallel

# 3. Performance validation and model testing
/run-all-tests --type=performance --gpu --agents=ai
/generate-tests ml_pipeline/ --type=performance --agents=ai
```

### Legacy Modernization with Domain Experts
```bash
# 1. Legacy code analysis with domain specialists
/multi-agent-optimize legacy_fortran/ --mode=review --agents=domain-specific --focus=architecture

# 2. Modernization with scientific computing agents
/adopt-code legacy_fortran/ --target=python --agents=scientific --optimize

# 3. Quality assurance with engineering agents
/multi-agent-optimize modernized_code/ --mode=review --agents=engineering --implement
```

### Cross-Domain Innovation Discovery
```bash
# 1. Innovation analysis across all 23 agents
/multi-agent-optimize novel_algorithm/ --mode=research --agents=all --focus=innovation --orchestrate

# 2. Breakthrough implementation with multi-agent coordination
/multi-agent-optimize novel_algorithm/ --implement --agents=all --focus=innovation --validate

# 3. Research documentation and reflection
/update-docs novel_algorithm/ --type=research --agents=all
/reflection --type=scientific --agents=research --breakthrough-mode
```

### Security & Architecture Review with Engineering Team
```bash
# 1. Security and architecture analysis
/multi-agent-optimize enterprise_app/ --mode=review --focus=security --agents=engineering --orchestrate

# 2. DevSecOps optimization implementation
/multi-agent-optimize enterprise_app/ --implement --agents=engineering --focus=security

# 3. Comprehensive validation and CI/CD setup
/ci-setup --type=security --agents=devops
/run-all-tests --security --agents=engineering
```

### Quantum Computing Optimization
```bash
# 1. Quantum computing analysis with specialized agents
/multi-agent-optimize quantum_algorithm/ --mode=optimize --agents=domain-specific --focus=performance

# 2. Quantum-classical hybrid optimization
/multi-agent-optimize quantum_algorithm/ --implement --agents=scientific --focus=innovation

# 3. Advanced validation and research documentation
/run-all-tests --type=scientific --reproducible
/update-docs quantum_algorithm/ --type=research --agents=all
```

## Emergent Intelligence: 23-Agent Synergies

### Cross-Domain Innovation Patterns
The combination of 23 specialized personal agents creates unique emergent capabilities that exceed the sum of individual agent expertise:

#### **Scientific-AI Convergence**
- `scientific-computing-master` + `ai-systems-architect` + `neural-networks-master` → Physics-Informed Neural Networks optimization
- `jax-pro` + `advanced-quantum-computing-expert` → Quantum-classical hybrid ML workflows

#### **Research-Engineering Bridge**
- `research-intelligence-master` + `systems-architect` + `code-quality-master` → Research-grade software engineering practices
- `documentation-architect` + `scientific-computing-master` → Publication-ready computational documentation

#### **Domain-Specific Innovation**
- `neutron-soft-matter-expert` + `xray-soft-matter-expert` + `correlation-function-expert` → Multi-technique experimental data analysis
- `nonequilibrium-stochastic-expert` + `neural-networks-master` → Stochastic neural differential equations

#### **Infrastructure-Research Integration**
- `devops-security-engineer` + `scientific-computing-master` + `database-workflow-engineer` → Scientific computing infrastructure at scale
- `command-systems-engineer` + `multi-agent-orchestrator` → Self-optimizing computational workflows

### 23-Agent Breakthrough Discovery Modes

#### **Innovation Cascade Analysis**
```bash
# Breakthrough discovery through 23-agent collaboration
/multi-agent-optimize breakthrough_research/ --mode=research --agents=all --focus=innovation --orchestrate
```
**Agent Cascade Pattern:**
1. **Domain Experts** identify domain-specific opportunities
2. **Research Intelligence** synthesizes cross-domain patterns
3. **AI/ML Agents** explore ML-enhanced solutions
4. **Engineering Agents** assess implementation feasibility
5. **Orchestrator** coordinates breakthrough implementation

#### **Emergent Optimization Strategies**
- **Multi-Scale Optimization**: Agents operating at different scales (quantum → classical → system → infrastructure)
- **Cross-Paradigm Solutions**: Combining insights from different computational paradigms
- **Adaptive Agent Allocation**: Dynamic agent assignment based on discovery patterns

### Agent Specialization Matrix

| Domain | Core Agents | Supporting Agents | Innovation Potential |
|--------|-------------|-------------------|---------------------|
| **Quantum Computing** | `advanced-quantum-computing-expert` | `scientific-computing-master`, `jax-pro`, `neural-networks-master` | Quantum-ML hybrid systems |
| **Scientific ML** | `jax-pro`, `neural-networks-master` | `scientific-computing-master`, `ai-ml-specialist` | Physics-informed neural networks |
| **Soft Matter Research** | `neutron-soft-matter-expert`, `xray-soft-matter-expert` | `correlation-function-expert`, `data-professional` | Multi-technique analysis platforms |
| **Research Infrastructure** | `systems-architect`, `devops-security-engineer` | `database-workflow-engineer`, `command-systems-engineer` | Self-optimizing research environments |
| **Innovation Discovery** | `research-intelligence-master`, `multi-agent-orchestrator` | All 23 agents | Cross-domain breakthrough identification |

## Related Commands

**Prerequisites**: Commands to run before multi-agent analysis
- `/check-code-quality --auto-fix` - Fix basic quality issues first
- `/debug --auto-fix` - Resolve runtime issues before optimization
- `/generate-tests` - Ensure adequate test coverage for validation
- `/run-all-tests` - Establish baseline performance metrics

**Alternatives**: Single-agent approaches
- `/optimize` - Single-domain performance optimization
- `/think-ultra` - General multi-agent analysis (broader scope, research focus)
- `/refactor-clean` - Code structure and modernization refactoring
- `/adopt-code` - Technology migration and modernization

**Combinations**: Commands that work with multi-agent optimize
- `/generate-tests` - Create tests to validate optimization results
- `/double-check` - Verify multi-agent recommendations systematically
- `/commit --ai-message` - Commit optimization changes with detailed messages
- `/reflection` - Meta-analyze multi-agent optimization process
- `/update-docs` - Document optimization improvements

**Follow-up**: Commands to run after optimization
- `/run-all-tests --benchmark` - Validate optimization results with performance metrics
- `/ci-setup --type=enterprise` - Automate quality checks and performance monitoring
- `/reflection --type=scientific` - Analyze optimization effectiveness
- `/double-check "optimization verification"` - Comprehensive result validation

## Integration Patterns

### Escalation Strategy
```bash
# Start simple, escalate to multi-agent as needed
/optimize code.py                         # Try single-agent first
/multi-agent-optimize code.py --agents=core  # Escalate to multiple agents
/think-ultra "complex analysis" --agents=all # For hardest problems
```

### Quality Assurance Pipeline
```bash
# Comprehensive quality improvement
/check-code-quality --auto-fix
/multi-agent-optimize --mode=review --focus=quality --implement
/generate-tests --type=all --security
/run-all-tests --coverage=95
```

### Architecture Review Process
```bash
# Multi-agent architecture analysis
/multi-agent-optimize project/ --focus=architecture --agents=all --parallel
/refactor-clean project/ --scope=project --implement
/update-docs project/ --type=api
```

## 23-Agent System Requirements

### Personal Agent Library Access
- **Agent Repository**: `/Users/b80985/.claude/agents/` (23 specialized personal agents)
- **Agent Activation**: Dynamic loading and coordination of relevant agents
- **Multi-Agent Communication**: Inter-agent message passing and coordination protocols

### Computational Resources
- **Parallel Processing**: Support for concurrent agent execution (--parallel)
- **Memory Management**: Efficient resource allocation for 23-agent coordination
- **Storage**: Adequate space for multi-agent analysis results and coordination logs

### Advanced Capabilities
- **Orchestration Engine**: `multi-agent-orchestrator` for workflow coordination
- **Cross-Domain Synthesis**: Intelligent synthesis across all 23 agent domains
- **Emergent Intelligence**: Pattern recognition across agent collaboration
- **Innovation Discovery**: Breakthrough identification through agent synergies

### System Integration
- **Codebase Access**: Target codebase for comprehensive multi-agent analysis
- **Tool Integration**: Access to specialized tools and libraries for each agent domain
- **Validation Framework**: Multi-agent testing and verification capabilities
- **Documentation System**: Research-grade documentation generation and management

## 23-Agent System Capabilities Summary

### **🚀 Core Advantages**
- **Comprehensive Coverage**: 23 specialized agents cover all domains from quantum computing to DevOps
- **Emergent Intelligence**: Agent collaboration creates insights beyond individual agent capabilities
- **Cross-Domain Innovation**: Breakthrough discoveries through multi-domain agent synthesis
- **Research-Grade Analysis**: Publication-quality analysis and optimization recommendations

### **🎯 Optimal Use Cases**
- **Scientific Computing Research**: Multi-domain research projects requiring diverse expertise
- **AI/ML System Optimization**: Complex ML pipelines with cross-domain requirements
- **Legacy Modernization**: Large-scale code modernization with domain-specific expertise
- **Innovation Discovery**: Breakthrough research requiring cross-paradigm analysis
- **Enterprise Architecture**: Large-scale systems requiring multi-domain optimization

### **⚡ Performance Characteristics**
- **Scalable Coordination**: Efficient orchestration of up to 23 agents
- **Intelligent Load Balancing**: Optimal resource allocation across agent domains
- **Parallel Execution**: Concurrent agent analysis for maximum efficiency
- **Adaptive Optimization**: Dynamic agent selection based on codebase characteristics

**The 23-agent multi-agent-optimize system represents the pinnacle of personal agent orchestration, enabling breakthrough discoveries and optimizations that are impossible with single-agent or limited multi-agent approaches.**