# Claude Code Agent Categories

**Version**: 1.0
**Last Updated**: 2025-09-29
**Total Agents**: 23

---

## Category Overview

The Claude Code agent system is organized into 5 primary categories based on specialization and use cases:

1. **Engineering Core** (7 agents) - Software development, architecture, and quality
2. **AI/ML Core** (3 agents) - Artificial intelligence and machine learning
3. **Scientific Computing** (3 agents) - Numerical computing and scientific simulation
4. **Domain Specialists** (7 agents) - Physics, chemistry, and specialized scientific domains
5. **Support Specialists** (3 agents) - Data, documentation, and orchestration

---

## 1. Engineering Core (7 agents)

**Purpose**: General software engineering, architecture, and development practices

### systems-architect
- **Role**: High-level architecture strategy and technology evaluation
- **Scope**: Enterprise architecture, API strategy, legacy modernization
- **Use When**: Need architectural decisions, technology selection, system design patterns
- **Lines**: 447 (comprehensive multi-domain scope)

### fullstack-developer
- **Role**: End-to-end feature implementation from database to UI
- **Scope**: Full-stack development, REST APIs, authentication, deployment
- **Use When**: Building complete features across entire stack
- **Lines**: 305

### code-quality-master
- **Role**: Code quality, testing, and software craftsmanship
- **Scope**: Testing strategies, code review, refactoring, quality assurance
- **Use When**: Need test coverage, code quality improvement, refactoring guidance
- **Lines**: 353

### command-systems-engineer
- **Role**: CLI tools and command-line interface development
- **Scope**: Argument parsing, interactive prompts, terminal UX
- **Use When**: Building CLI applications or command-line tools
- **Lines**: 360

### devops-security-engineer
- **Role**: DevOps, CI/CD, infrastructure, and security
- **Scope**: Docker, Kubernetes, CI/CD pipelines, security hardening
- **Use When**: Need infrastructure automation, deployment pipelines, security audits
- **Lines**: 316

### database-workflow-engineer
- **Role**: Database design and workflow orchestration
- **Scope**: PostgreSQL optimization, database migrations, workflow engines
- **Use When**: Database architecture, complex workflows, data pipelines
- **Lines**: 368

### documentation-architect
- **Role**: Technical documentation strategy and generation
- **Scope**: API documentation, architecture docs, knowledge management
- **Use When**: Creating comprehensive documentation, knowledge bases
- **Lines**: 331

---

## 2. AI/ML Core (3 agents)

**Purpose**: Artificial intelligence, machine learning, and AI infrastructure

### ai-ml-specialist
- **Role**: Full ML lifecycle from data to deployment
- **Scope**: Model training, feature engineering, ML pipelines, MLOps
- **Use When**: Building ML applications, training models, ML deployment
- **Differentiation**: Hands-on ML development vs AI infrastructure
- **Lines**: 176

### ai-systems-architect
- **Role**: AI infrastructure and LLM system architecture
- **Scope**: LLM serving, MCP integration, multi-model orchestration, prompt engineering
- **Use When**: Designing AI platforms, LLM infrastructure, agent systems
- **Differentiation**: AI platform design vs ML model training
- **Lines**: 381

### neural-networks-master
- **Role**: Neural architecture research and optimization
- **Scope**: Novel network designs, architecture comparison, neural paradigms
- **Use When**: Designing custom neural networks, architecture optimization
- **Differentiation**: Neural architecture research vs full ML workflows
- **Lines**: 186

---

## 3. Scientific Computing (3 agents)

**Purpose**: Numerical computing, scientific simulation, and high-performance computing

### scientific-computing-master
- **Role**: Multi-language scientific computing and HPC
- **Scope**: Julia, Fortran, C++, classical numerical methods, MPI/OpenMP
- **Use When**: Multi-language HPC, classical methods, non-JAX workflows
- **Differentiation**: Multi-language expertise, classical methods
- **Lines**: 447 (comprehensive multi-language/multi-domain scope)

### jax-pro
- **Role**: JAX framework and functional programming expert
- **Scope**: JAX transformations (jit/vmap/pmap), Flax, Optax, Orbax
- **Use When**: General JAX development, neural networks with Flax
- **Differentiation**: JAX framework expertise vs domain applications
- **Lines**: 182

### jax-scientific-domains
- **Role**: Domain-specific JAX applications
- **Scope**: Quantum computing (Cirq), CFD, molecular dynamics, signal processing
- **Use When**: JAX + specialized domains (quantum/CFD/MD)
- **Differentiation**: JAX domain specializations vs general JAX
- **Lines**: 277

---

## 4. Domain Specialists (7 agents)

**Purpose**: Specialized scientific domains requiring deep subject matter expertise

### advanced-quantum-computing-expert
- **Role**: Advanced quantum algorithms and error correction
- **Scope**: VQE, QAOA, quantum error correction, quantum machine learning
- **Use When**: Complex quantum algorithms, research-level quantum computing
- **Lines**: 171

### correlation-function-expert
- **Role**: Correlation function analysis for soft matter physics
- **Scope**: Radial distribution functions, structure factors, spatial correlations
- **Use When**: Analyzing spatial correlations in molecular simulations
- **Lines**: 178

### neutron-soft-matter-expert
- **Role**: Neutron scattering analysis for soft matter
- **Scope**: SANS, NSLD calculations, neutron contrast matching
- **Use When**: Neutron scattering data analysis and experimental design
- **Lines**: 181

### xray-soft-matter-expert
- **Role**: X-ray scattering analysis for soft matter
- **Scope**: SAXS, WAXS, X-ray structure factors, scattering profiles
- **Use When**: X-ray scattering data analysis and interpretation
- **Lines**: 181

### nonequilibrium-stochastic-expert
- **Role**: Non-equilibrium statistical mechanics and stochastic processes
- **Scope**: Langevin dynamics, Fokker-Planck equations, master equations
- **Use When**: Non-equilibrium simulations, stochastic modeling
- **Lines**: 176

### scientific-code-adoptor
- **Role**: Legacy scientific code modernization
- **Scope**: Fortran/C/C++/MATLAB → Python/JAX/Julia migration
- **Use When**: Modernizing legacy scientific codebases
- **Lines**: 184

---

## 5. Support Specialists (3 agents)

**Purpose**: Cross-cutting concerns supporting all other agents

### data-professional
- **Role**: Data engineering, analytics, and data science
- **Scope**: ETL pipelines, data warehousing, analytics, data visualization
- **Use When**: Data infrastructure, analytics pipelines, BI systems
- **Lines**: 187

### visualization-interface-master
- **Role**: Data visualization and interactive interface design
- **Scope**: Scientific plotting, dashboards, web visualizations, interactive UIs
- **Use When**: Creating visualizations, dashboards, interactive interfaces
- **Lines**: 338

### research-intelligence-master
- **Role**: Literature review and research synthesis
- **Scope**: Paper analysis, methodology extraction, research workflows
- **Use When**: Literature reviews, research analysis, knowledge synthesis
- **Lines**: 306

### multi-agent-orchestrator
- **Role**: Coordinating multi-agent workflows
- **Scope**: Agent delegation, workflow orchestration, task decomposition
- **Use When**: Complex multi-agent tasks requiring coordination
- **Lines**: 324

---

## Agent Selection Guide

### By Development Phase

**Planning & Design**:
- systems-architect (software architecture)
- ai-systems-architect (AI infrastructure)
- database-workflow-engineer (data architecture)

**Implementation**:
- fullstack-developer (complete features)
- ai-ml-specialist (ML applications)
- jax-pro (JAX development)

**Quality & Maintenance**:
- code-quality-master (testing, refactoring)
- devops-security-engineer (deployment, security)
- documentation-architect (documentation)

### By Technical Domain

**Web Development**: fullstack-developer, systems-architect, database-workflow-engineer
**AI/ML**: ai-ml-specialist, neural-networks-master, ai-systems-architect
**Scientific Computing**: scientific-computing-master, jax-pro, jax-scientific-domains
**Physics/Chemistry**: Domain specialists (quantum, neutron, xray, correlation, nonequilibrium)
**Data Engineering**: data-professional, database-workflow-engineer, visualization-interface-master
**CLI Tools**: command-systems-engineer
**Research**: research-intelligence-master, domain specialists

### By Complexity

**High Complexity** (>400 lines):
- scientific-computing-master (447) - Multi-language HPC
- systems-architect (447) - Enterprise architecture

**Medium Complexity** (300-400 lines):
- database-workflow-engineer (368)
- command-systems-engineer (360)
- code-quality-master (353)
- visualization-interface-master (338)
- documentation-architect (331)
- multi-agent-orchestrator (324)
- devops-security-engineer (316)
- research-intelligence-master (306)
- fullstack-developer (305)

**Focused Specialists** (<200 lines):
- Domain experts (171-187 lines)
- AI/ML specialists (176-186 lines)
- JAX specialists (182-277 lines)

---

## Category Relationships

### Vertical Integration
```
Systems Architect → Fullstack Developer
    ↓                    ↓
AI Systems Arch → AI/ML Specialist → Neural Networks Master
    ↓                    ↓
Database Workflow ← Data Professional
```

### Horizontal Collaboration
```
Scientific Computing Master ↔ JAX Pro ↔ JAX Scientific Domains
                                ↓
                        Domain Specialists
                    (Quantum, Neutron, X-ray, etc.)
```

### Support Layer
```
All Agents → Documentation Architect
All Agents → Visualization Interface Master
All Agents → Research Intelligence Master
All Agents → Multi-Agent Orchestrator
```

---

## Performance Characteristics

### Fast Response (Simple queries, focused scope)
- Domain specialists: 171-187 lines, single-focus expertise
- JAX specialists: Well-optimized for specific frameworks

### Comprehensive Analysis (Complex systems, broad scope)
- scientific-computing-master: Multi-language, multi-domain
- systems-architect: Enterprise-wide architecture

### Iterative Development (Build-measure-learn cycles)
- fullstack-developer: Rapid feature iteration
- ai-ml-specialist: Experiment-train-deploy cycles

---

## Maintenance Notes

### Adding New Agents
1. Classify into one of 5 categories
2. Define clear differentiation from similar agents
3. Follow AGENT_TEMPLATE.md structure
4. Target 130-400 lines (exceptions for comprehensive scope)
5. Update this AGENT_CATEGORIES.md document

### Reviewing Agent Overlap
1. Check within-category differentiation
2. Verify cross-references are bidirectional
3. Ensure "Choose X over Y when Z" clarity
4. Update differentiation sections if roles evolve

### Quarterly Audits
- Verify line counts remain within acceptable ranges
- Update tool lists for deprecated/new tools
- Refresh cross-references for structural changes
- Validate agent descriptions match current capabilities

---

**Document Maintained By**: Agent Optimization Project
**Template Reference**: AGENT_TEMPLATE.md
**Project Documentation**: README.md (consolidated from optimization and completion reports)