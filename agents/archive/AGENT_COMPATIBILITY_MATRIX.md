# Agent Compatibility Matrix

**Version**: 1.0
**Last Updated**: 2025-09-29
**Purpose**: Define collaboration patterns, delegation strategies, and workflow sequences across 23 Claude Code agents

---

## Matrix Overview

This document maps agent relationships, compatibility patterns, and optimal delegation strategies. Use this guide to:
- **Identify complementary agents** for multi-agent workflows
- **Design delegation sequences** for complex tasks
- **Avoid redundant invocations** by understanding agent overlap
- **Optimize task routing** based on agent capabilities

---

## Compatibility Matrix (Quick Reference)

| Agent | Primary Collaborators | Delegates To | Escalates To | Avoids |
|-------|----------------------|--------------|--------------|--------|
| **systems-architect** | fullstack-developer, ai-systems-architect | fullstack-developer, database-workflow-engineer | - | implementation details |
| **fullstack-developer** | database-workflow-engineer, devops-security-engineer | systems-architect (planning), code-quality-master (testing) | systems-architect | architecture design |
| **ai-ml-specialist** | data-professional, neural-networks-master | ai-systems-architect (infrastructure), data-professional (preprocessing) | - | AI platform design |
| **ai-systems-architect** | ai-ml-specialist, systems-architect | ai-ml-specialist (models), devops-security-engineer (infrastructure) | - | model training |
| **neural-networks-master** | ai-ml-specialist, jax-pro | ai-ml-specialist (full workflow), jax-pro (JAX optimization) | - | data preprocessing |
| **scientific-computing-master** | jax-pro, jax-scientific-domains | domain specialists (physics), visualization-interface-master | - | pure JAX work |
| **jax-pro** | scientific-computing-master, neural-networks-master | jax-scientific-domains (domain apps), scientific-computing-master (classical preprocessing) | - | multi-language HPC |
| **jax-scientific-domains** | jax-pro, domain specialists | jax-pro (JAX optimization), domain specialists (physics validation) | - | non-JAX frameworks |
| **code-quality-master** | All agents | - | - | initial implementation |
| **devops-security-engineer** | fullstack-developer, ai-systems-architect | - | - | application logic |
| **database-workflow-engineer** | fullstack-developer, data-professional | data-professional (analytics), devops-security-engineer (deployment) | - | frontend work |
| **data-professional** | ai-ml-specialist, database-workflow-engineer | database-workflow-engineer (schema), visualization-interface-master (viz) | - | model training |
| **documentation-architect** | All agents | - | - | code implementation |
| **visualization-interface-master** | data-professional, domain specialists | - | - | data processing |
| **research-intelligence-master** | domain specialists, neural-networks-master | - | - | implementation |
| **multi-agent-orchestrator** | All agents | (coordinates all) | - | single-agent tasks |
| **command-systems-engineer** | code-quality-master | - | fullstack-developer | web UIs |
| **Domain Specialists** | scientific-computing-master, jax-scientific-domains | scientific-computing-master (computation), visualization-interface-master (viz) | - | general software |

---

## Collaboration Patterns

### Pattern 1: Strategic Planning → Implementation
**Flow**: Architect → Developer → Quality

```
systems-architect (design)
    ↓
fullstack-developer (implement)
    ↓
code-quality-master (validate)
    ↓
devops-security-engineer (deploy)
```

**Use Cases**:
- New application development
- Feature roadmap implementation
- System modernization projects

**Compatibility Score**: ⭐⭐⭐⭐⭐ (optimal sequence)

---

### Pattern 2: AI/ML Development Pipeline
**Flow**: Data → Model → Infrastructure → Deployment

```
data-professional (data prep)
    ↓
ai-ml-specialist (model training)
    ↓
neural-networks-master (architecture optimization) [optional]
    ↓
ai-systems-architect (platform integration)
    ↓
devops-security-engineer (deployment)
```

**Use Cases**:
- ML model development and deployment
- AI platform building
- Model optimization projects

**Compatibility Score**: ⭐⭐⭐⭐⭐ (optimal sequence)

---

### Pattern 3: Scientific Computing Workflow
**Flow**: Classical → JAX → Domain → Visualization

```
scientific-computing-master (preprocessing, classical methods)
    ↓
jax-pro (JAX acceleration)
    ↓
jax-scientific-domains (domain-specific algorithms)
    ↓
[domain specialist] (physics validation) [optional]
    ↓
visualization-interface-master (results visualization)
```

**Use Cases**:
- Scientific simulations
- Physics-based modeling
- Numerical computing projects

**Compatibility Score**: ⭐⭐⭐⭐⭐ (optimal sequence)

---

### Pattern 4: Database-Centric Application
**Flow**: Schema → Backend → Frontend → Workflows

```
database-workflow-engineer (schema design)
    ↓
fullstack-developer (API + UI)
    ↓
data-professional (analytics) [optional]
    ↓
visualization-interface-master (dashboards)
```

**Use Cases**:
- Data-driven applications
- Workflow automation systems
- Analytics platforms

**Compatibility Score**: ⭐⭐⭐⭐⭐ (optimal sequence)

---

### Pattern 5: Research to Production
**Flow**: Research → Implementation → Documentation

```
research-intelligence-master (literature review)
    ↓
[appropriate specialist agent] (implementation)
    ↓
code-quality-master (testing)
    ↓
documentation-architect (documentation)
```

**Use Cases**:
- Research implementation projects
- Algorithm translation from papers
- Novel method development

**Compatibility Score**: ⭐⭐⭐⭐ (good sequence)

---

## Delegation Decision Tree

### Starting Point: What is the primary task type?

#### 1. Architecture & Design
- **High-level system design** → systems-architect
- **AI infrastructure design** → ai-systems-architect
- **Database architecture** → database-workflow-engineer
- **Then delegate to**: Appropriate implementation agents

#### 2. Implementation
- **Full-stack feature** → fullstack-developer
- **ML model training** → ai-ml-specialist
- **Scientific computing** → scientific-computing-master or jax-pro
- **CLI tool** → command-systems-engineer
- **Then delegate to**: code-quality-master (testing), devops-security-engineer (deployment)

#### 3. Optimization & Quality
- **Code quality improvement** → code-quality-master
- **Neural architecture optimization** → neural-networks-master
- **JAX performance tuning** → jax-pro
- **Then delegate to**: Original implementation agent if major refactoring needed

#### 4. Specialized Domains
- **Quantum computing** → advanced-quantum-computing-expert
- **Neutron scattering** → neutron-soft-matter-expert
- **X-ray scattering** → xray-soft-matter-expert
- **Correlation functions** → correlation-function-expert
- **Non-equilibrium physics** → nonequilibrium-stochastic-expert
- **Legacy code modernization** → scientific-code-adoptor
- **Then delegate to**: scientific-computing-master or jax-pro for computational implementation

#### 5. Support Functions
- **Data preparation** → data-professional
- **Visualization** → visualization-interface-master
- **Documentation** → documentation-architect
- **Literature review** → research-intelligence-master
- **Multi-agent coordination** → multi-agent-orchestrator

---

## Anti-Patterns (Avoid These)

### ❌ Anti-Pattern 1: Wrong Phase Agent
**Problem**: Using implementation agent for architecture decisions
**Example**: fullstack-developer for system design → Should use systems-architect first
**Impact**: Poor architectural decisions, technical debt

### ❌ Anti-Pattern 2: Skipping Planning
**Problem**: Jumping directly to implementation without architecture
**Example**: ai-ml-specialist before ai-systems-architect for new AI platform
**Impact**: Scalability issues, infrastructure problems

### ❌ Anti-Pattern 3: Framework Mismatch
**Problem**: Using JAX agent for non-JAX work
**Example**: jax-pro for Julia-only project → Should use scientific-computing-master
**Impact**: Incorrect framework selection, wasted effort

### ❌ Anti-Pattern 4: Premature Optimization
**Problem**: Invoking optimization agents before basic implementation
**Example**: neural-networks-master before ai-ml-specialist has working model
**Impact**: Optimizing non-existent code

### ❌ Anti-Pattern 5: Over-Specification
**Problem**: Using domain specialist when general agent sufficient
**Example**: advanced-quantum-computing-expert for basic quantum gates → jax-scientific-domains sufficient
**Impact**: Unnecessary complexity

---

## Compatibility Scoring System

### ⭐⭐⭐⭐⭐ Excellent Compatibility (Sequential or Parallel)
- **Sequential**: Natural workflow progression with clear handoff
- **Parallel**: Complementary work without conflicts
- **Examples**: systems-architect → fullstack-developer, data-professional + ai-ml-specialist

### ⭐⭐⭐⭐ Good Compatibility (Some Overlap)
- **Coordination needed**: Slight overlap requiring clear boundaries
- **Clear benefits**: Each agent adds distinct value
- **Examples**: jax-pro + neural-networks-master, database-workflow-engineer + fullstack-developer

### ⭐⭐⭐ Moderate Compatibility (Requires Careful Orchestration)
- **Potential conflicts**: Overlapping responsibilities need management
- **Value exists**: Collaboration valuable with careful planning
- **Examples**: scientific-computing-master + jax-pro (language choice conflicts)

### ⭐⭐ Limited Compatibility (Rarely Used Together)
- **Minimal overlap**: Rarely need both for same task
- **Distinct domains**: Different problem spaces
- **Examples**: command-systems-engineer + visualization-interface-master

### ⭐ Incompatible (Do Not Use Together)
- **Conflicting goals**: Agents work at cross purposes
- **Redundant**: One agent fully contains the other's capabilities
- **Examples**: None in current agent set (all agents have distinct value)

---

## Common Multi-Agent Workflows

### Workflow 1: Build New Web Application
```
1. systems-architect (architecture design) - 2 hours
2. database-workflow-engineer (schema design) - 3 hours
3. fullstack-developer (implementation) - 40 hours
4. code-quality-master (testing) - 8 hours
5. devops-security-engineer (deployment) - 4 hours
6. documentation-architect (documentation) - 4 hours

Total: ~60 hours
Agents: 6
Compatibility: ⭐⭐⭐⭐⭐
```

### Workflow 2: Implement ML Model
```
1. research-intelligence-master (literature review) - 4 hours [optional]
2. data-professional (data pipeline) - 8 hours
3. ai-ml-specialist (model training) - 20 hours
4. neural-networks-master (architecture optimization) - 8 hours [optional]
5. ai-systems-architect (infrastructure) - 6 hours
6. devops-security-engineer (deployment) - 4 hours
7. documentation-architect (model docs) - 3 hours

Total: ~50 hours
Agents: 7
Compatibility: ⭐⭐⭐⭐⭐
```

### Workflow 3: Scientific Simulation Project
```
1. research-intelligence-master (paper analysis) - 3 hours
2. scientific-computing-master (algorithm design) - 8 hours
3. jax-pro (JAX implementation) - 12 hours
4. [domain specialist] (physics validation) - 4 hours
5. visualization-interface-master (results viz) - 6 hours
6. documentation-architect (methods documentation) - 4 hours

Total: ~37 hours
Agents: 6
Compatibility: ⭐⭐⭐⭐⭐
```

### Workflow 4: Legacy Code Modernization
```
1. scientific-code-adoptor (analysis & migration) - 20 hours
2. scientific-computing-master (modernization) - 12 hours
3. jax-pro (JAX acceleration) - 10 hours [if JAX target]
4. code-quality-master (testing) - 8 hours
5. documentation-architect (documentation) - 4 hours

Total: ~54 hours
Agents: 5
Compatibility: ⭐⭐⭐⭐⭐
```

### Workflow 5: Data Analytics Platform
```
1. systems-architect (platform design) - 3 hours
2. database-workflow-engineer (data architecture) - 6 hours
3. data-professional (ETL pipelines) - 15 hours
4. visualization-interface-master (dashboards) - 10 hours
5. devops-security-engineer (deployment) - 5 hours

Total: ~39 hours
Agents: 5
Compatibility: ⭐⭐⭐⭐⭐
```

---

## Agent Substitution Guide

When primary agent unavailable or task characteristics change:

### Architecture Tier
- **systems-architect** ↔ **ai-systems-architect** (if AI-centric)
- **database-workflow-engineer** ↔ **systems-architect** (for data architecture only)

### Implementation Tier
- **fullstack-developer** ↔ **command-systems-engineer** (if CLI instead of web)
- **ai-ml-specialist** ↔ **neural-networks-master** (if pure architecture focus)
- **scientific-computing-master** ↔ **jax-pro** (if JAX-only)

### Domain Specialists
- **jax-scientific-domains** ↔ **domain specialists** (for specific physics domains)
- **advanced-quantum-computing-expert** ↔ **jax-scientific-domains** (for basic quantum)

### Support Tier
- No direct substitutions (each support agent has unique capabilities)

---

## Escalation Paths

### When to Escalate UP (to Architecture)
- Implementation reveals architectural problems
- Scope expands beyond original design
- Performance requires architectural changes
- **Example**: fullstack-developer → systems-architect when database design inadequate

### When to Escalate DOWN (to Implementation)
- Architecture complete, need execution
- Proof of concept required
- Detailed technical work needed
- **Example**: systems-architect → fullstack-developer after design approval

### When to Escalate LATERAL (to Peer Specialist)
- Task requires complementary expertise
- Different framework more appropriate
- Domain-specific knowledge needed
- **Example**: jax-pro → jax-scientific-domains for quantum computing domain

### When to Escalate to SUPPORT
- Quality issues arise → code-quality-master
- Documentation needed → documentation-architect
- Visualization required → visualization-interface-master
- Research context needed → research-intelligence-master

---

## Parallel Execution Opportunities

### Safe Parallel Execution (No Conflicts)
- **systems-architect** + **research-intelligence-master** (design + literature review)
- **data-professional** + **database-workflow-engineer** (ETL + schema work on different datasets)
- **documentation-architect** + **code-quality-master** (docs + tests, independent)
- **visualization-interface-master** + **frontend work** (viz components + app structure)

### Sequential Required (Dependencies)
- **systems-architect** BEFORE **fullstack-developer** (architecture → implementation)
- **data-professional** BEFORE **ai-ml-specialist** (data → model)
- **implementation** BEFORE **code-quality-master** (code → tests)
- **code-quality-master** BEFORE **devops-security-engineer** (tests → deployment)

---

## Compatibility Matrix by Category

### Engineering Core ↔ Engineering Core
- **High compatibility**: All engineering agents work well together in sequence
- **Optimal flow**: architect → developer → quality → devops → documentation

### AI/ML Core ↔ AI/ML Core
- **Moderate overlap**: Clear differentiation needed (see AGENT_CATEGORIES.md)
- **Optimal flow**: ai-ml-specialist → neural-networks-master → ai-systems-architect

### Scientific Computing ↔ Scientific Computing
- **Language-based compatibility**: Choose based on framework (Julia vs JAX vs multi-language)
- **Optimal flow**: scientific-computing-master → jax-pro → jax-scientific-domains

### Domain Specialists ↔ Scientific Computing
- **Excellent compatibility**: Domain experts define physics, computing agents implement
- **Optimal flow**: [domain specialist] → scientific-computing-master/jax-pro

### Support ↔ Any Category
- **Universal compatibility**: Support agents enhance any workflow
- **Flexible integration**: Can be added at any stage

---

## Usage Guidelines

### 1. Single-Agent Tasks (No Collaboration Needed)
- Simple, focused tasks within one domain
- Clear boundaries, no architectural decisions
- Examples: Bug fix (code-quality-master), Simple visualization (visualization-interface-master)

### 2. Two-Agent Collaboration (Minimal Coordination)
- Tasks spanning 2 domains with clear handoff
- Examples: Architecture + Implementation (systems-architect → fullstack-developer)

### 3. Multi-Agent Workflows (3-5 agents, Orchestration Helpful)
- Complex projects with multiple phases
- Consider using multi-agent-orchestrator for coordination
- Examples: Full application development, ML pipeline implementation

### 4. Large-Scale Projects (5+ agents, Orchestration Required)
- Enterprise-scale projects with many moving parts
- **Must use multi-agent-orchestrator** to manage workflow
- Examples: Platform development, system modernization

---

## Maintenance

### Adding New Agents
1. Identify agent's category (Engineering/AI/Scientific/Domain/Support)
2. Map compatibility with existing agents in same category
3. Define delegation patterns (delegates to / escalates to / avoids)
4. Update this matrix with new agent relationships

### Updating Relationships
1. Review agent differentiation sections for changes
2. Update compatibility scores if agent scopes change
3. Verify workflow examples remain accurate
4. Maintain bidirectional cross-references

### Quarterly Review
- Validate common workflows reflect actual usage
- Update time estimates based on experience
- Refine anti-patterns based on observed issues
- Add new collaboration patterns as discovered

---

**Matrix Maintained By**: Agent Optimization Project
**Cross-Reference**: AGENT_CATEGORIES.md (category structure), AGENT_TEMPLATE.md (agent structure)
**Project Documentation**: README.md (consolidated project information)