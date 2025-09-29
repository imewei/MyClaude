---
title: "Think Ultra"
description: "Advanced analytical thinking engine with multi-agent collaboration for complex problems"
category: cognitive-intelligence
subcategory: meta-analysis
complexity: advanced
argument-hint: "[problem] [--depth=auto|comprehensive|ultra|quantum] [--mode=auto|systematic|discovery|hybrid] [--paradigm=auto|multi|cross|meta] [--agents=auto|core|scientific|engineering|domain-specific|all] [--priority=auto|implementation] [--recursive=false|true] [--export-insights] [--auto-fix=false|true] [--orchestrate] [--intelligent] [--breakthrough]"
allowed-tools: Read, Write, Grep, Glob, TodoWrite, Bash, WebSearch, WebFetch, MultiEdit
model: inherit
tags: analysis, multi-agent, research, implementation, cognitive-enhancement
dependencies: []
related: [optimize, multi-agent-optimize, debug, double-check, reflection, generate-tests, run-all-tests, check-code-quality, adopt-code, refactor-clean]
workflows: [analysis-to-implementation, research-workflow, meta-optimization]
version: "2.0"
last-updated: "2025-09-28"
---

# Advanced Analytical Thinking Engine

Multi-agent collaborative analysis for complex problems using specialized reasoning approaches and domain expertise.

## Purpose

Performs deep analytical thinking using multiple AI agents with different specializations. Scales from simple analysis to complex multi-domain research with real implementation.

## Quick Start

```bash
# Basic analysis
/think-ultra "How do I optimize this algorithm?"

# Research-grade analysis with all agents
/think-ultra "Design ML architecture for scientific computing" --depth=ultra --agents=all

# Implementation-focused analysis
/think-ultra "Deploy scalable ML system" --priority=implementation

# Auto-fix: Analysis + Implementation
/think-ultra "Optimize this Python script" --auto-fix --agents=scientific
```

## Usage

```bash
/think-ultra "[problem description]" [options]
```

**Parameters:**
- `problem description` - Question or challenge to analyze (in quotes)
- `options` - Analysis configuration and execution options

## Auto-Fix Implementation Mode

When `--auto-fix=true` is enabled, think-ultra transforms from analysis-only to analysis + execution:

**Standard Mode (--auto-fix=false):**
1. Perform multi-agent analysis
2. Generate detailed recommendations
3. Output analysis report
4. User manually implements suggestions

**Auto-Fix Mode (--auto-fix=true):**
1. Perform multi-agent analysis
2. Generate detailed recommendations
3. **Automatically execute recommendations using Claude tools**
4. Validate implementation success
5. Report execution results

**Auto-Fix Implementation Process:**
- **Recommendation Extraction** - Parse analysis for actionable items
- **Tool Planning** - Convert recommendations to Write/Edit/Bash commands
- **Safe Execution** - Execute changes with error handling and rollback
- **Validation** - Verify implementation success and functionality

**When to Use Auto-Fix:**
- Code optimization tasks where changes are well-defined
- File organization and refactoring projects
- Documentation generation and updates
- Performance improvements with clear implementation steps

**When NOT to Use Auto-Fix:**
- Exploratory analysis where recommendations need review
- Complex architectural decisions requiring human judgment
- High-risk changes to critical systems
- Research questions without clear implementation path

## Arguments

- **`problem`** - Question or challenge to analyze
- **`--depth`** - Analysis depth: auto, comprehensive (default), ultra, quantum
- **`--mode`** - Approach: auto, systematic, discovery, hybrid (default)
- **`--agents`** - Personal agent categories: auto, core, scientific, engineering, domain-specific, all
- **`--orchestrate`** - Enable intelligent agent orchestration and coordination
- **`--intelligent`** - Activate advanced reasoning and cross-agent synthesis
- **`--breakthrough`** - Focus on paradigm shifts and innovative discoveries
- **`--paradigm`** - Thinking style: auto, multi (default), cross, meta
- **`--priority`** - Focus: auto (default), implementation
- **`--recursive`** - Self-improving analysis: false (default), true
- **`--export-insights`** - Generate deliverable files
- **`--auto-fix`** - Execute recommendations: false (default), true

## Personal Agent Categories

**Core Agents** - Foundational reasoning, problem-solving, and cognitive enhancement
**Scientific Agents** - JAX/Julia/Python optimization, numerical methods, performance engineering
**Engineering Agents** - Architecture, development, deployment, quality assurance, DevOps
**Domain-Specific Agents** - Specialized expertise for targeted problem domains
**All Agents** - Complete 23-agent system with intelligent orchestration

### Agent Orchestration Options

**--orchestrate** - Intelligent coordination between agents for optimal collaboration
**--intelligent** - Advanced reasoning synthesis across multiple agent perspectives
**--breakthrough** - Focus on paradigm shifts and innovative breakthrough discovery

## Usage Patterns

```bash
# Core reasoning (foundational agents)
/think-ultra "your problem" --agents=core

# Technical optimization (scientific + engineering)
/think-ultra "optimization challenge" --agents=scientific,engineering --orchestrate

# Domain-specific analysis (targeted expertise)
/think-ultra "specialized problem" --agents=domain-specific --intelligent

# Maximum capability (all 23 agents)
/think-ultra "complex system design" --agents=all --breakthrough --priority=implementation
```

## Quick Agent Selection Guide

**🚀 New User? Start Here:**

| **Your Problem Type** | **Recommended Agents** | **Example Command** |
|----------------------|----------------------|-------------------|
| **General analysis** | `--agents=core` | `/think-ultra "analyze this approach" --agents=core` |
| **Code optimization** | `--agents=scientific` | `/think-ultra "optimize Python performance" --agents=scientific --orchestrate` |
| **System design** | `--agents=engineering` | `/think-ultra "design architecture" --agents=engineering --intelligent` |
| **Research questions** | `--agents=domain-specific` | `/think-ultra "research methodology" --agents=domain-specific` |
| **Complex projects** | `--agents=all` | `/think-ultra "complex problem" --agents=all --breakthrough` |

**🎯 Quick Decision Tree:**
- **Simple problem?** → Use `--agents=core`
- **Technical/scientific?** → Use `--agents=scientific`
- **Engineering/architecture?** → Use `--agents=engineering`
- **Research/documentation?** → Use `--agents=domain-specific`
- **Maximum insight needed?** → Use `--agents=all`

**⚡ Pro Tip**: Add `--orchestrate` for better coordination, `--intelligent` for enhanced reasoning, `--breakthrough` for innovation focus.

## When to Use

**Use think-ultra for:**
- Complex multi-dimensional problems requiring deep analysis
- Research and development projects
- Cross-domain synthesis and innovation
- High-stakes technical decisions
- Performance optimization challenges

**Use standard execution for:**
- Simple implementation tasks
- Well-defined problems with known solutions
- Time-sensitive quick answers
- Basic debugging or documentation

## Analysis Depth Levels

- **Comprehensive** - Thorough single-domain analysis with systematic methodology
- **Ultra** - Multi-domain analysis with cross-disciplinary insights
- **Quantum** - Maximum depth analysis with paradigm shift detection and breakthrough discovery

## Analysis Modes

- **Systematic** - Structured, methodical approach with rigorous validation
- **Discovery** - Innovation-focused with creative pattern recognition
- **Hybrid** - Balanced approach combining systematic rigor with creative insights

## Personal 23-Agent System

Our personal multi-agent system uses 23 specialized agents organized into strategic categories:

### Core Agents (6 agents)
**Meta-Cognitive Agent** - Higher-order thinking, self-reflection, cognitive optimization
**Strategic-Thinking Agent** - Long-term planning, decision frameworks, strategic analysis
**Creative-Innovation Agent** - Breakthrough thinking, paradigm shifts, novel connections
**Problem-Solving Agent** - Systematic analysis, solution generation, optimization
**Critical-Analysis Agent** - Logic validation, assumption testing, skeptical evaluation
**Synthesis Agent** - Integration, pattern recognition, holistic understanding

### Scientific Agents (5 agents)
**JAX-Performance Agent** - JAX optimization, XLA compilation, GPU acceleration
**Julia-Computing Agent** - Julia performance, distributed computing, numerical methods
**Python-Scientific Agent** - Scientific Python, NumPy/SciPy optimization, algorithms
**ML-Engineering Agent** - Machine learning systems, model optimization, production ML
**Data-Science Agent** - Statistical analysis, experimental design, data engineering

### Engineering Agents (6 agents)
**Architecture Agent** - System design, scalability, technical architecture
**Full-Stack Agent** - End-to-end development, integration, user experience
**DevOps Agent** - Infrastructure, deployment, automation, monitoring
**Security Agent** - Security analysis, vulnerability assessment, secure coding
**Quality-Assurance Agent** - Testing strategies, code quality, validation frameworks
**Performance-Engineering Agent** - Optimization, profiling, scalability engineering

### Domain-Specific Agents (6 agents)
**Research-Methodology Agent** - Research design, literature synthesis, peer review standards
**Documentation Agent** - Technical writing, API docs, knowledge management
**UI-UX Agent** - User interface design, user experience, accessibility
**Database Agent** - Data modeling, query optimization, database design
**Network-Systems Agent** - Distributed systems, networking, communication protocols
**Integration Agent** - Cross-domain synthesis, interdisciplinary connections

### Agent Coordination Patterns

**Intelligent Orchestration (--orchestrate)**
- Dynamic agent selection based on problem characteristics
- Adaptive workflow routing and task distribution
- Real-time coordination and conflict resolution
- Resource optimization and parallel processing

**Advanced Reasoning (--intelligent)**
- Cross-agent knowledge synthesis and validation
- Multi-perspective analysis and viewpoint integration
- Cognitive bias detection and mitigation
- Evidence triangulation and consensus building

**Breakthrough Discovery (--breakthrough)**
- Paradigm shift detection and exploration
- Innovation pathway identification
- Disruptive opportunity analysis
- Creative constraint relaxation and reframing

## Examples

```bash
# JAX Performance Optimization with Personal Agents
/think-ultra "Optimize JAX performance for 10B parameter model training" \
  --depth=ultra --agents=scientific --orchestrate --intelligent

# Research Paper Development with Domain Expertise
/think-ultra "Create transformer variant optimized for PDE solving" \
  --agents=scientific,domain-specific --paradigm=meta --breakthrough

# Production System Design with Full Agent Team
/think-ultra "Scalable ML inference architecture with <10ms latency" \
  --agents=all --priority=implementation --orchestrate

# Cross-Domain Innovation with Personal Agents
/think-ultra "Apply quantum computing principles to ML optimization" \
  --paradigm=cross --agents=core,scientific --breakthrough --intelligent

# Auto-Fix Examples with Personal Agent System
/think-ultra "Optimize this Python script for performance" --auto-fix --agents=scientific --orchestrate
/think-ultra "Refactor this codebase to improve maintainability" --auto-fix --agents=engineering --intelligent
/think-ultra "Fix code quality issues in this project" --auto-fix --agents=all --priority=implementation --orchestrate
```

## Integration with Other Commands

```bash
# Strategy + Implementation workflow with Personal Agents
/think-ultra "JAX optimization strategy" --agents=core,scientific --orchestrate
/jax-performance --technique=caching --gpu-accel

# Auto-fix workflow with Intelligent Coordination
/think-ultra "Optimize JAX code for performance" --auto-fix --agents=scientific --intelligent

# Research + Validation workflow with Domain Expertise
/think-ultra "experimental design" --agents=core,domain-specific --breakthrough
/run-all-tests --scientific --auto-fix

# Auto-fix + Verification workflow with Full Agent Team
/think-ultra "Fix code quality issues" --auto-fix --agents=engineering --orchestrate
/double-check --deep-analysis --auto-complete
```

## Output Framework

Analysis follows 8-phase structured framework:

1. **Problem Architecture** - Mathematical foundations, complexity analysis
2. **Multi-Dimensional Systems** - Stakeholder analysis, cross-domain integration
3. **Evidence Synthesis** - Literature integration, methodological framework
4. **Innovation Analysis** - Breakthrough opportunities, paradigm shifts
5. **Risk Assessment** - Technical uncertainties, mitigation strategies
6. **Alternatives Analysis** - Multi-paradigm approaches, trade-offs
7. **Implementation Strategy** - Roadmap, resource requirements, success metrics
8. **Future Considerations** - Sustainability, evolution pathways, broader impact

## Performance Expectations

- **Cognitive Enhancement**: Improved reasoning patterns and creative insights
- **Scientific Computing**: 10-50x performance optimization recommendations
- **Research Quality**: Publication-ready analysis with peer-review standards
- **Implementation**: Working prototypes and production deployment strategies
- **Cross-Domain**: Novel connections and breakthrough opportunities
- **Auto-Fix Execution**: Complete analysis-to-implementation workflow with validation

## Common Workflows

### Analysis → Implementation Pattern
```bash
# 1. Deep analysis with personal agents
/think-ultra "optimize ML training pipeline" --depth=ultra --agents=scientific,engineering --orchestrate

# 2. Apply optimizations with intelligent coordination
/optimize training/ --language=jax --implement
/jax-performance --technique=caching --gpu-accel

# 3. Verify with personal agent validation
/double-check "ML training optimization" --deep-analysis --auto-complete
```

### Research → Development Workflow
```bash
# 1. Research methodology with domain expertise
/think-ultra "design experiment framework" --agents=domain-specific,core --paradigm=meta --breakthrough

# 2. Generate implementation with engineering agents
/generate-tests research/ --type=scientific --framework=auto
/update-docs research/ --type=api --research

# 3. Meta-analysis with personal agent synthesis
/reflection --type=scientific --optimize=innovation --breakthrough-mode
```

### Problem-Solving Escalation
```bash
# Start simple → escalate with personal agents as needed
/optimize code.py                    # Try standard optimization first
/multi-agent-optimize code.py --agents=core,scientific        # Personal agents for complex analysis
/think-ultra "complex problem" --agents=all --orchestrate --breakthrough  # Full 23-agent system for hardest problems
```

## Related Commands

**Prerequisites**: Commands that provide input or context
- `/check-code-quality` - Assess current code quality before analysis
- `/debug --auto-fix` - Fix runtime issues before deep analysis
- `/explain-code` - Understand codebase structure before optimization analysis
- Clean working environment - Remove temporary files and ensure clean state

**Alternatives**: Similar functionality, different approaches
- `/multi-agent-optimize` - Multi-agent focus specifically on code optimization
- `/optimize --implement` - Single-domain performance optimization with implementation
- `/reflection --type=comprehensive` - Self-analysis and session improvement
- `/double-check --deep-analysis` - Verification-focused multi-angle analysis

**Combinations**: Commands that enhance think-ultra
- `/double-check --deep-analysis` - Systematically verify think-ultra recommendations
- `/generate-tests --coverage=95` - Implement comprehensive testing based on analysis
- `/adopt-code --optimize` - Modernize legacy code based on insights
- `/refactor-clean --implement` - Apply structural improvements from analysis
- `/commit --template=optimization` - Commit analysis-driven improvements

**Follow-up Workflows**: Common next steps
- Analysis → `/optimize --implement` → `/generate-tests` → `/double-check`
- Research → `/scientific-computing/*` → `/run-all-tests --scientific`
- Strategy → `/multi-agent-optimize --implement` → `/reflection --type=scientific`
- Implementation → `/run-all-tests --auto-fix` → `/commit --validate`

## Integration Patterns

### With Scientific Computing Commands
```bash
# JAX optimization workflow with personal agents
/think-ultra "JAX performance analysis" --agents=scientific --auto-fix --orchestrate
# Intelligently coordinates: /jax-performance, /jax-debug, /jax-essentials

# Julia development workflow with engineering integration
/think-ultra "Julia code modernization" --agents=scientific,engineering --intelligent
# Follow with: /julia-jit-like, /julia-ad-grad
```

### With Quality Assurance
```bash
# Code quality improvement with personal agents
/think-ultra "improve code quality" --auto-fix --agents=engineering --orchestrate
# Intelligently combines: /check-code-quality, /refactor-clean, /generate-tests
```

### With Documentation and CI/CD
```bash
# Project improvement with full personal agent team
/think-ultra "project optimization" --agents=all --priority=implementation --orchestrate --intelligent
# Coordinates: /update-docs, /ci-setup, /clean-codebase
```