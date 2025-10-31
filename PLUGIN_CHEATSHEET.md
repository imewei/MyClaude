# Claude Code Plugin Marketplace - Quick Reference

**Total Resources:** 31 Plugins | 73 Agents | 48 Commands | 110 Skills
**Version:** 1.0.1 | **Last Updated:** October 31, 2025

---

## Quick Navigation

- [Scientific Computing](#scientific-computing-2-plugins) (2) - Julia, HPC, JAX, Statistical Physics
- [Development](#development-7-plugins) (7) - Python, Backend, Frontend, Systems Programming
- [AI & Machine Learning](#ai--machine-learning-2-plugins) (2) - Deep Learning, ML Pipelines
- [DevOps & Infrastructure](#devops--infrastructure-2-plugins) (2) - CI/CD, Git Workflows
- [Code Quality & Testing](#code-quality--testing-4-plugins) (4) - Testing, Review, Cleanup
- [Documentation & Migration](#documentation--migration-4-plugins) (4) - Docs, Code Migration
- [Debugging & Tools](#debugging--tools-3-plugins) (3) - Debugging, CLI Design, Visualization
- [Orchestration & AI](#orchestration--ai-3-plugins) (3) - Agents, Reasoning, LLM Apps
- [Specialized](#specialized-6-plugins) (6) - Molecular Sim, Research, Observability

---

## Scientific Computing (2 plugins)

### julia-development
**Purpose:** Comprehensive Julia development with SciML, Bayesian inference, and scientific computing
**Key Agents:** julia-pro, sciml-pro, turing-pro, julia-developer (4 agents)
**Skills:** DifferentialEquations, ModelingToolkit, Turing MCMC, Performance tuning (20 skills)
**Use When:** Scientific simulations, differential equations, Bayesian modeling, high-performance numerics
**Install:** `/plugin install julia-development@scientific-computing-workflows`

### hpc-computing
**Purpose:** High-performance computing on clusters with parallel/GPU acceleration
**Key Agent:** hpc-numerical-coordinator (1 agent)
**Skills:** GPU acceleration, parallel computing, numerical methods, ecosystem selection (4 skills)
**Use When:** Large-scale simulations, cluster computing, GPU programming, parallel algorithms
**Install:** `/plugin install hpc-computing@scientific-computing-workflows`

---

## Development (7 plugins)

### python-development
**Purpose:** Modern Python development with async patterns, packaging, and performance optimization
**Key Agents:** python-pro, fastapi-pro, django-pro (3 agents)
**Skills:** Async patterns, uv package manager, performance optimization, testing patterns (5 skills)
**Use When:** Python projects, FastAPI/Django apps, async programming, package development
**Install:** `/plugin install python-development@scientific-computing-workflows`

### backend-development
**Purpose:** Backend API design, microservices architecture, and scalable systems
**Key Agents:** backend-architect, graphql-architect, tdd-orchestrator (3 agents)
**Skills:** API design, microservices patterns, authentication, error handling, SQL optimization (6 skills)
**Use When:** REST/GraphQL APIs, microservices, backend architecture, database optimization
**Install:** `/plugin install backend-development@scientific-computing-workflows`

### frontend-mobile-development
**Purpose:** Frontend and mobile app development with React, React Native
**Key Agents:** frontend-developer, mobile-developer (2 agents)
**Use When:** Web UIs, mobile apps, React development, cross-platform applications
**Install:** `/plugin install frontend-mobile-development@scientific-computing-workflows`

### javascript-typescript
**Purpose:** Modern JavaScript/TypeScript development with advanced patterns
**Key Agents:** javascript-pro, typescript-pro (2 agents)
**Skills:** Modern JS patterns, advanced TypeScript types, testing, monorepo management, Node.js backends (5 skills)
**Use When:** JavaScript projects, TypeScript type systems, Node.js backends, monorepos
**Install:** `/plugin install javascript-typescript@scientific-computing-workflows`

### systems-programming
**Purpose:** Systems-level programming in Rust, C, C++, Go
**Key Agents:** rust-pro, c-pro, cpp-pro, golang-pro (4 agents)
**Skills:** Systems programming patterns (1 skill)
**Use When:** Performance-critical code, systems programming, low-level development, embedded systems
**Install:** `/plugin install systems-programming@scientific-computing-workflows`

### multi-platform-apps
**Purpose:** Cross-platform application development (mobile, web, desktop)
**Key Agents:** flutter-expert, ios-developer, mobile-developer, frontend-developer, backend-architect, ui-ux-designer (6 agents)
**Skills:** Flutter development, iOS best practices, React Native, multi-platform architecture (4 skills)
**Use When:** Multi-platform apps, Flutter projects, iOS apps, unified codebases
**Install:** `/plugin install multi-platform-apps@scientific-computing-workflows`

### llm-application-dev
**Purpose:** LLM-powered application development with RAG and agents
**Key Agents:** ai-engineer, prompt-engineer (2 agents)
**Skills:** RAG implementation, prompt engineering, LangChain architecture, LLM evaluation (4 skills)
**Use When:** AI-powered apps, chatbots, RAG systems, prompt optimization
**Install:** `/plugin install llm-application-dev@scientific-computing-workflows`

---

## AI & Machine Learning (2 plugins)

### deep-learning
**Purpose:** Neural network architectures, training workflows, and deep learning research
**Key Agents:** neural-architecture-engineer, neural-network-master (2 agents)
**Skills:** Neural architectures, training diagnostics, model optimization, research implementation (6 skills)
**Use When:** Deep learning models, neural network design, training optimization, research papers
**Install:** `/plugin install deep-learning@scientific-computing-workflows`

### machine-learning
**Purpose:** Machine learning pipelines, MLOps, and production ML systems
**Key Agents:** data-scientist, ml-engineer, mlops-engineer (3 agents)
**Skills:** ML essentials, advanced ML systems, data wrangling, model deployment, ML pipelines (8 skills)
**Use When:** ML workflows, model deployment, MLOps, data science, production ML
**Install:** `/plugin install machine-learning@scientific-computing-workflows`

---

## DevOps & Infrastructure (2 plugins)

### cicd-automation
**Purpose:** CI/CD pipelines, deployment automation, cloud infrastructure, Kubernetes
**Key Agents:** deployment-engineer, kubernetes-architect, cloud-architect, terraform-specialist, devops-troubleshooter (5 agents)
**Commands:** `/fix-commit-errors` (1 command)
**Skills:** GitHub Actions, GitLab CI, deployment pipelines, Kubernetes, Terraform, secrets management (6 skills)
**Use When:** CI/CD setup, Kubernetes deployments, cloud infrastructure, IaC, pipeline debugging
**Install:** `/plugin install cicd-automation@scientific-computing-workflows`

### git-pr-workflows
**Purpose:** Git workflows, pull request automation, code review
**Key Agent:** code-reviewer (1 agent)
**Commands:** `/commit`, `/git-workflow` (2 commands)
**Skills:** Git best practices (1 skill)
**Use When:** Git commits, PR creation, code reviews, version control workflows
**Install:** `/plugin install git-pr-workflows@scientific-computing-workflows`

---

## Code Quality & Testing (4 plugins)

### unit-testing
**Purpose:** Comprehensive test automation with pytest, Jest, property-based testing
**Key Agents:** test-automator, debugger (2 agents)
**Commands:** `/run-all-tests`, `/test-generate` (2 commands)
**Skills:** Testing patterns (1 skill)
**Use When:** Writing tests, test automation, property-based testing, test coverage
**Install:** `/plugin install unit-testing@scientific-computing-workflows`

### comprehensive-review
**Purpose:** Multi-dimensional code review (architecture, security, performance)
**Key Agents:** code-reviewer, architect-review, security-auditor (3 agents)
**Skills:** Comprehensive review framework (1 skill)
**Use When:** Code reviews, security audits, architecture validation, pre-deployment checks
**Install:** `/plugin install comprehensive-review@scientific-computing-workflows`

### codebase-cleanup
**Purpose:** Code maintenance, refactoring, technical debt reduction
**Key Agents:** code-reviewer, test-automator (2 agents)
**Use When:** Refactoring, removing dead code, dependency updates, codebase maintenance
**Install:** `/plugin install codebase-cleanup@scientific-computing-workflows`

### quality-engineering
**Purpose:** Quality assurance, plugin validation, comprehensive validation frameworks
**Key Resources:** Plugin syntax validator, comprehensive validation (2 skills)
**Commands:** `/lint-plugins`, `/double-check` (2 commands)
**Use When:** Quality gates, plugin validation, pre-release checks, validation automation
**Install:** `/plugin install quality-engineering@scientific-computing-workflows`

---

## Documentation & Migration (4 plugins)

### code-documentation
**Purpose:** Automated documentation generation, architecture guides, tutorials
**Key Agents:** docs-architect, tutorial-engineer, code-reviewer (3 agents)
**Commands:** `/code-explain`, `/update-docs`, `/update-claudemd` (3 commands)
**Use When:** Generating docs, API documentation, README updates, architecture diagrams
**Install:** `/plugin install code-documentation@scientific-computing-workflows`

### code-migration
**Purpose:** Legacy code modernization, cross-language migration (Fortran/C/MATLAB → Python/Julia)
**Key Agent:** scientific-code-adoptor (1 agent)
**Commands:** `/adopt-code` (1 command)
**Use When:** Modernizing legacy code, cross-language migration, adopting external codebases
**Install:** `/plugin install code-migration@scientific-computing-workflows`

### framework-migration
**Purpose:** Framework upgrades, dependency updates, migration strategies
**Key Agents:** legacy-modernizer, architect-review (2 agents)
**Commands:** `/legacy-modernize` (1 command)
**Skills:** Angular migration, React modernization, database migration, dependency upgrades (4 skills)
**Use When:** Framework upgrades, library migrations, breaking change handling
**Install:** `/plugin install framework-migration@scientific-computing-workflows`

### research-methodology
**Purpose:** Research workflows, literature analysis, methodology design
**Key Agent:** research-intelligence (1 agent)
**Skills:** Research methodology (1 skill)
**Use When:** Research projects, literature reviews, experimental design, methodology planning
**Install:** `/plugin install research-methodology@scientific-computing-workflows`

---

## Debugging & Tools (3 plugins)

### debugging-toolkit
**Purpose:** AI-assisted debugging, root cause analysis, observability
**Key Agents:** debugger, dx-optimizer (2 agents)
**Skills:** AI-assisted debugging, debugging strategies, observability/SRE (3 skills)
**Use When:** Bug investigation, performance debugging, observability setup, developer experience
**Install:** `/plugin install debugging-toolkit@scientific-computing-workflows`

### cli-tool-design
**Purpose:** Command-line tool design and automation
**Key Agent:** command-systems-engineer (1 agent)
**Skills:** CLI tool design, scripting languages (2 skills)
**Use When:** Building CLI tools, command automation, developer tooling
**Install:** `/plugin install cli-tool-design@scientific-computing-workflows`

### data-visualization
**Purpose:** Scientific data visualization with Python, Julia, D3.js
**Key Agent:** visualization-interface (1 agent)
**Skills:** Python/Julia visualization, scientific visualization, UX design (3 skills)
**Use When:** Data visualization, plotting, dashboards, scientific graphics
**Install:** `/plugin install data-visualization@scientific-computing-workflows`

---

## Orchestration & AI (3 plugins)

### agent-orchestration
**Purpose:** Multi-agent system coordination, workflow optimization
**Key Agents:** multi-agent-orchestrator, context-manager (2 agents)
**Commands:** `/multi-agent-optimize` (1 command)
**Skills:** Multi-agent coordination, agent performance optimization (2 skills)
**Use When:** Complex multi-agent workflows, agent optimization, distributed AI systems
**Install:** `/plugin install agent-orchestration@scientific-computing-workflows`

### ai-reasoning
**Purpose:** Structured reasoning, reflection frameworks, AI problem-solving
**Key Agents:** Multiple reasoning agents (4 agents)
**Commands:** `/ultra-think`, `/reflection` (2 commands)
**Skills:** Structured reasoning, meta-cognitive reflection, comprehensive reflection (3 skills)
**Use When:** Complex problem-solving, structured analysis, AI reasoning, decision-making
**Install:** `/plugin install ai-reasoning@scientific-computing-workflows`

### full-stack-orchestration
**Purpose:** Full-stack application orchestration and coordination
**Key Agents:** deployment-engineer, performance-engineer, security-auditor, test-automator (4 agents)
**Use When:** Full-stack projects, end-to-end orchestration, deployment coordination
**Install:** `/plugin install full-stack-orchestration@scientific-computing-workflows`

---

## Specialized (6 plugins)

### jax-implementation
**Purpose:** JAX for numerical computing, physics-informed ML, optimization
**Key Agents:** jax-pro, jax-scientist, nlsq-pro, numpyro-pro (4 agents)
**Skills:** JAX core programming, physics applications, NLSQ, NumPyro (4 skills)
**Use When:** JAX projects, physics simulations, Bayesian inference with JAX, optimization
**Install:** `/plugin install jax-implementation@scientific-computing-workflows`

### molecular-simulation
**Purpose:** Molecular dynamics, multiscale modeling, trajectory analysis
**Key Agent:** simulation-expert (1 agent)
**Skills:** MD simulation, ML force fields, multiscale modeling, trajectory analysis (4 skills)
**Use When:** Molecular simulations, atomistic modeling, materials science, computational chemistry
**Install:** `/plugin install molecular-simulation@scientific-computing-workflows`

### statistical-physics
**Purpose:** Statistical physics modeling, correlation functions, non-equilibrium systems
**Key Agents:** correlation-function-expert, non-equilibrium-expert (2 agents)
**Skills:** Correlation functions, active matter, non-equilibrium theory, stochastic dynamics (8 skills)
**Use When:** Statistical physics, correlation analysis, active matter, non-equilibrium systems
**Install:** `/plugin install statistical-physics@scientific-computing-workflows`

### observability-monitoring
**Purpose:** System monitoring, observability, performance engineering, database optimization
**Key Agents:** observability-engineer, performance-engineer, database-optimizer, network-engineer (4 agents)
**Skills:** Prometheus, Grafana, distributed tracing, SLO implementation, Airflow workflows (5 skills)
**Use When:** Monitoring setup, observability, performance tuning, database optimization
**Install:** `/plugin install observability-monitoring@scientific-computing-workflows`

---

## Common Workflows

### Scientific Research Pipeline
```bash
# 1. Setup Julia environment
/plugin install julia-development@scientific-computing-workflows

# 2. High-performance computing
/plugin install hpc-computing@scientific-computing-workflows

# 3. Data visualization
/plugin install data-visualization@scientific-computing-workflows

# 4. Research methodology
/plugin install research-methodology@scientific-computing-workflows
```

### Full-Stack Web Development
```bash
# 1. Backend development
/plugin install backend-development@scientific-computing-workflows

# 2. Frontend development
/plugin install frontend-mobile-development@scientific-computing-workflows

# 3. Testing
/plugin install unit-testing@scientific-computing-workflows

# 4. CI/CD
/plugin install cicd-automation@scientific-computing-workflows
```

### AI/ML Development
```bash
# 1. Machine learning
/plugin install machine-learning@scientific-computing-workflows

# 2. Deep learning
/plugin install deep-learning@scientific-computing-workflows

# 3. LLM applications
/plugin install llm-application-dev@scientific-computing-workflows

# 4. JAX for optimization
/plugin install jax-implementation@scientific-computing-workflows
```

### DevOps & Infrastructure
```bash
# 1. CI/CD automation
/plugin install cicd-automation@scientific-computing-workflows

# 2. Git workflows
/plugin install git-pr-workflows@scientific-computing-workflows

# 3. Observability
/plugin install observability-monitoring@scientific-computing-workflows

# 4. Debugging
/plugin install debugging-toolkit@scientific-computing-workflows
```

### Code Quality & Maintenance
```bash
# 1. Comprehensive review
/plugin install comprehensive-review@scientific-computing-workflows

# 2. Testing
/plugin install unit-testing@scientific-computing-workflows

# 3. Quality engineering
/plugin install quality-engineering@scientific-computing-workflows

# 4. Codebase cleanup
/plugin install codebase-cleanup@scientific-computing-workflows
```

---

## Installation Options

### Option 1: Install All Plugins
```bash
# Clone repository
git clone https://github.com/imewei/MyClaude.git
cd MyClaude

# Enable all 31 plugins
make plugin-enable-all
```

### Option 2: Install by Category
```bash
# Scientific Computing
/plugin install julia-development@scientific-computing-workflows
/plugin install hpc-computing@scientific-computing-workflows

# Development
/plugin install python-development@scientific-computing-workflows
/plugin install backend-development@scientific-computing-workflows

# AI/ML
/plugin install machine-learning@scientific-computing-workflows
/plugin install deep-learning@scientific-computing-workflows

# DevOps
/plugin install cicd-automation@scientific-computing-workflows
/plugin install git-pr-workflows@scientific-computing-workflows
```

### Option 3: Add Marketplace First
```bash
# Add marketplace
/plugin marketplace add imewei/MyClaude

# Browse and install via UI
# Select "Browse and install plugins" → "scientific-computing-workflows" → Select plugin → "Install now"
```

---

## Quick Reference by Use Case

| Use Case | Recommended Plugins |
|----------|-------------------|
| **Scientific Computing** | julia-development, hpc-computing, jax-implementation, statistical-physics |
| **Web Development** | backend-development, frontend-mobile-development, python-development |
| **Mobile Apps** | multi-platform-apps, frontend-mobile-development |
| **AI/ML Projects** | machine-learning, deep-learning, llm-application-dev |
| **Data Science** | python-development, machine-learning, data-visualization |
| **DevOps** | cicd-automation, observability-monitoring, git-pr-workflows |
| **Code Quality** | comprehensive-review, unit-testing, quality-engineering, codebase-cleanup |
| **Legacy Migration** | code-migration, framework-migration |
| **Research** | research-methodology, julia-development, jax-implementation |
| **Systems Programming** | systems-programming, debugging-toolkit |

---

## Additional Resources

- **Full Documentation:** https://myclaude.readthedocs.io/en/latest/
- **GitHub Repository:** https://github.com/imewei/MyClaude
- **Version:** 1.0.1
- **Last Updated:** October 31, 2025
- **Total Plugins:** 31 | **Agents:** 73 | **Commands:** 48 | **Skills:** 110

---

## Notes

- All plugins follow consistent naming conventions (kebab-case)
- Agent references use `plugin:agent` format (single colon)
- Commands use `/command-name` format
- Skills are organized in `skills/skill-name/SKILL.md`
- Full validation completed with 100% success rate (v1.0.1)

---

**Quick Tip:** Use `/quality-engineering:lint-plugins` to validate plugin structure and references across your installation.
