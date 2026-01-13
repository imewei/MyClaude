# Claude Code Plugin Marketplace - Quick Reference

**Total Resources:** 31 Plugins | 74 Agents | 49 Commands | 114 Skills
**Version:** 1.0.7 | **Last Updated:** January 12, 2026

---

## Quick Navigation

- [Scientific Computing](#scientific-computing-8-plugins) (8) - Julia, HPC, JAX, Statistical Physics, Molecular Sim
- [Development](#development-10-plugins) (10) - Python, Backend, Frontend, Systems, Multi-Platform
- [AI & Machine Learning](#ai--machine-learning-2-plugins) (2) - Deep Learning, ML Pipelines
- [DevOps & Infrastructure](#devops--infrastructure-3-plugins) (3) - CI/CD, Git Workflows, Observability
- [Quality & Testing](#quality--testing-4-plugins) (4) - Testing, Review, Cleanup, Quality Engineering
- [Tools & Migration](#tools--migration-4-plugins) (4) - Documentation, Migration, Debugging

---

## Scientific Computing (8 plugins)

### julia-development (4 agents, 4 commands, 21 skills)
**Purpose:** Comprehensive Julia development with SciML, Bayesian inference, and scientific computing
**Agents:** julia-pro, sciml-pro, turing-pro, julia-developer
**Commands:** `/julia-optimize`, `/julia-package-ci`, `/julia-scaffold`, `/sciml-setup`
**Use When:** Scientific simulations, differential equations, Bayesian modeling, high-performance numerics

### jax-implementation (4 agents, 4 skills)
**Purpose:** JAX for numerical computing, physics-informed ML, optimization
**Agents:** jax-pro, jax-scientist, nlsq-pro, numpyro-pro
**Use When:** JAX projects, physics simulations, Bayesian inference with JAX, NLSQ optimization

### hpc-computing (1 agent, 4 skills)
**Purpose:** High-performance computing on clusters with parallel/GPU acceleration
**Agent:** hpc-numerical-coordinator
**Use When:** Large-scale simulations, cluster computing, GPU programming, parallel algorithms

### molecular-simulation (1 agent, 4 skills)
**Purpose:** Molecular dynamics, multiscale modeling, trajectory analysis
**Agent:** simulation-expert
**Use When:** Molecular simulations, atomistic modeling, materials science, computational chemistry

### statistical-physics (2 agents, 8 skills)
**Purpose:** Statistical physics modeling, correlation functions, non-equilibrium systems
**Agents:** correlation-function-expert, non-equilibrium-expert
**Use When:** Statistical physics, correlation analysis, active matter, non-equilibrium systems

### data-visualization (1 agent, 3 skills)
**Purpose:** Scientific data visualization with Python, Julia, D3.js
**Agent:** visualization-interface
**Use When:** Data visualization, plotting, dashboards, scientific graphics

### research-methodology (1 agent, 1 skill)
**Purpose:** Research workflows, literature analysis, methodology design
**Agent:** research-intelligence
**Use When:** Research projects, literature reviews, experimental design, methodology planning

### deep-learning (2 agents, 6 skills)
**Purpose:** Neural network architectures, training workflows, and deep learning research
**Agents:** neural-architecture-engineer, neural-network-master
**Use When:** Deep learning models, neural network design, training optimization, research papers

---

## Development (10 plugins)

### python-development (3 agents, 1 command, 5 skills)
**Purpose:** Modern Python development with async patterns, packaging, and performance optimization
**Agents:** python-pro, fastapi-pro, django-pro
**Command:** `/python-scaffold`
**Use When:** Python projects, FastAPI/Django apps, async programming, package development

### backend-development (3 agents, 1 command, 6 skills)
**Purpose:** Backend API design, microservices architecture, and scalable systems
**Agents:** backend-architect, graphql-architect, tdd-orchestrator
**Command:** `/feature-development`
**Use When:** REST/GraphQL APIs, microservices, backend architecture, database optimization

### frontend-mobile-development (2 agents, 1 command)
**Purpose:** Frontend and mobile app development with React, React Native
**Agents:** frontend-developer, mobile-developer
**Command:** `/component-scaffold`
**Use When:** Web UIs, mobile apps, React development, cross-platform applications

### javascript-typescript (2 agents, 1 command, 5 skills)
**Purpose:** Modern JavaScript/TypeScript development with advanced patterns
**Agents:** javascript-pro, typescript-pro
**Command:** `/typescript-scaffold`
**Use When:** JavaScript projects, TypeScript type systems, Node.js backends, monorepos

### systems-programming (4 agents, 3 commands, 1 skill)
**Purpose:** Systems-level programming in Rust, C, C++, Go
**Agents:** rust-pro, c-pro, cpp-pro, golang-pro
**Commands:** `/c-project`, `/profile-performance`, `/rust-project`
**Use When:** Performance-critical code, systems programming, low-level development, embedded systems

### multi-platform-apps (6 agents, 1 command, 4 skills)
**Purpose:** Cross-platform application development (mobile, web, desktop)
**Agents:** flutter-expert, ios-developer, mobile-developer, frontend-developer, backend-architect, ui-ux-designer
**Command:** `/multi-platform`
**Use When:** Multi-platform apps, Flutter projects, iOS apps, unified codebases

### llm-application-dev (2 agents, 3 commands, 4 skills)
**Purpose:** LLM-powered application development with RAG and agents
**Agents:** ai-engineer, prompt-engineer
**Commands:** `/ai-assistant`, `/langchain-agent`, `/prompt-optimize`
**Use When:** AI-powered apps, chatbots, RAG systems, prompt optimization

### cli-tool-design (1 agent, 2 skills)
**Purpose:** Command-line tool design and automation
**Agent:** command-systems-engineer
**Use When:** Building CLI tools, command automation, developer tooling

### full-stack-orchestration (4 agents, 1 command)
**Purpose:** Full-stack application orchestration and coordination
**Agents:** deployment-engineer, performance-engineer, security-auditor, test-automator
**Command:** `/full-stack-feature`
**Use When:** Full-stack projects, end-to-end orchestration, deployment coordination

### agent-orchestration (2 agents, 2 commands, 2 skills)
**Purpose:** Multi-agent system coordination, workflow optimization
**Agents:** multi-agent-orchestrator, context-manager
**Commands:** `/improve-agent`, `/multi-agent-optimize`
**Use When:** Complex multi-agent workflows, agent optimization, distributed AI systems

---

## AI & Machine Learning (2 plugins)

### machine-learning (4 agents, 1 command, 7 skills)
**Purpose:** Machine learning pipelines, MLOps, and production ML systems
**Agents:** data-scientist, data-engineer, ml-engineer, mlops-engineer
**Command:** `/ml-pipeline`
**Use When:** ML workflows, model deployment, MLOps, data science, production ML

### ai-reasoning (2 commands, 3 skills)
**Purpose:** Structured reasoning, reflection frameworks, AI problem-solving
**Commands:** `/ultra-think`, `/reflection`
**Use When:** Complex problem-solving, structured analysis, AI reasoning, decision-making

---

## DevOps & Infrastructure (3 plugins)

### cicd-automation (5 agents, 2 commands, 6 skills)
**Purpose:** CI/CD pipelines, deployment automation, cloud infrastructure, Kubernetes
**Agents:** deployment-engineer, kubernetes-architect, cloud-architect, terraform-specialist, devops-troubleshooter
**Commands:** `/fix-commit-errors`, `/workflow-automate`
**Use When:** CI/CD setup, Kubernetes deployments, cloud infrastructure, IaC, pipeline debugging

### git-pr-workflows (1 agent, 5 commands, 1 skill)
**Purpose:** Git workflows, pull request automation, code review
**Agent:** code-reviewer
**Commands:** `/commit`, `/git-workflow`, `/merge-all`, `/onboard`, `/pr-enhance`
**Use When:** Git commits, PR creation, code reviews, version control workflows, branch consolidation

### observability-monitoring (4 agents, 2 commands, 5 skills)
**Purpose:** System monitoring, observability, performance engineering, database optimization
**Agents:** observability-engineer, performance-engineer, database-optimizer, network-engineer
**Commands:** `/monitor-setup`, `/slo-implement`
**Use When:** Monitoring setup, observability, performance tuning, database optimization

---

## Quality & Testing (4 plugins)

### unit-testing (2 agents, 2 commands, 1 skill)
**Purpose:** Comprehensive test automation with pytest, Jest, property-based testing
**Agents:** test-automator, debugger
**Commands:** `/run-all-tests`, `/test-generate`
**Use When:** Writing tests, test automation, property-based testing, test coverage

### comprehensive-review (3 agents, 2 commands, 1 skill)
**Purpose:** Multi-dimensional code review (architecture, security, performance)
**Agents:** code-reviewer, architect-reviewer, security-auditor
**Commands:** `/full-review`, `/pr-enhance`
**Use When:** Code reviews, security audits, architecture validation, pre-deployment checks

### codebase-cleanup (2 agents, 4 commands)
**Purpose:** Code maintenance, refactoring, technical debt reduction
**Agents:** code-reviewer, test-automator
**Commands:** `/deps-audit`, `/fix-imports`, `/refactor-clean`, `/tech-debt`
**Use When:** Refactoring, removing dead code, dependency updates, codebase maintenance

### quality-engineering (2 commands, 2 skills)
**Purpose:** Quality assurance, plugin validation, comprehensive validation frameworks
**Commands:** `/double-check`, `/lint-plugins`
**Use When:** Quality gates, plugin validation, pre-release checks, validation automation

---

## Tools & Migration (4 plugins)

### code-documentation (3 agents, 4 commands)
**Purpose:** Automated documentation generation, architecture guides, tutorials
**Agents:** docs-architect, tutorial-engineer, code-reviewer
**Commands:** `/code-explain`, `/doc-generate`, `/update-claudemd`, `/update-docs`
**Use When:** Generating docs, API documentation, README updates, architecture diagrams

### code-migration (1 agent, 1 command)
**Purpose:** Legacy code modernization, cross-language migration (Fortran/C/MATLAB → Python/Julia)
**Agent:** scientific-code-adoptor
**Command:** `/adopt-code`
**Use When:** Modernizing legacy code, cross-language migration, adopting external codebases

### framework-migration (2 agents, 3 commands, 4 skills)
**Purpose:** Framework upgrades, dependency updates, migration strategies
**Agents:** legacy-modernizer, architect-reviewer
**Commands:** `/code-migrate`, `/deps-upgrade`, `/legacy-modernize`
**Use When:** Framework upgrades, library migrations, breaking change handling

### debugging-toolkit (2 agents, 1 command, 3 skills)
**Purpose:** AI-assisted debugging, root cause analysis, observability
**Agents:** debugger, dx-optimizer
**Command:** `/smart-debug`
**Use When:** Bug investigation, performance debugging, observability setup, developer experience

---

## Common Workflows

### Scientific Research Pipeline
```bash
/plugin install julia-development@scientific-computing-workflows
/plugin install hpc-computing@scientific-computing-workflows
/plugin install jax-implementation@scientific-computing-workflows
/plugin install data-visualization@scientific-computing-workflows
```

### Full-Stack Web Development
```bash
/plugin install backend-development@scientific-computing-workflows
/plugin install frontend-mobile-development@scientific-computing-workflows
/plugin install unit-testing@scientific-computing-workflows
/plugin install cicd-automation@scientific-computing-workflows
```

### AI/ML Development
```bash
/plugin install machine-learning@scientific-computing-workflows
/plugin install deep-learning@scientific-computing-workflows
/plugin install llm-application-dev@scientific-computing-workflows
/plugin install jax-implementation@scientific-computing-workflows
```

### DevOps & Infrastructure
```bash
/plugin install cicd-automation@scientific-computing-workflows
/plugin install git-pr-workflows@scientific-computing-workflows
/plugin install observability-monitoring@scientific-computing-workflows
/plugin install debugging-toolkit@scientific-computing-workflows
```

### Code Quality & Maintenance
```bash
/plugin install comprehensive-review@scientific-computing-workflows
/plugin install unit-testing@scientific-computing-workflows
/plugin install quality-engineering@scientific-computing-workflows
/plugin install codebase-cleanup@scientific-computing-workflows
```

---

## Installation Options

### Option 1: Install All Plugins
```bash
git clone https://github.com/imewei/MyClaude.git
cd MyClaude
make plugin-enable-all
```

### Option 2: Add Marketplace and Browse
```bash
/plugin marketplace add imewei/MyClaude
# Select "Browse and install plugins" → "scientific-computing-workflows" → Select plugin
```

### Option 3: Install Individual Plugins
```bash
/plugin install python-development@scientific-computing-workflows
/plugin install backend-development@scientific-computing-workflows
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

## Statistics Summary

| Category | Plugins | Agents | Commands |
|----------|---------|--------|----------|
| Scientific Computing | 8 | 18 | 4 |
| Development | 10 | 24 | 14 |
| AI & ML | 2 | 6 | 3 |
| DevOps & Infrastructure | 3 | 10 | 9 |
| Quality & Testing | 4 | 7 | 10 |
| Tools & Migration | 4 | 8 | 9 |
| **Total** | **31** | **74** | **49** |

---

## Additional Resources

- **Full Documentation:** [https://myclaude.readthedocs.io/en/latest/](https://myclaude.readthedocs.io/en/latest/)
- **GitHub Repository:** [https://github.com/imewei/MyClaude](https://github.com/imewei/MyClaude)
- **Agents List:** [AGENTS_LIST.md](AGENTS_LIST.md)
- **Commands List:** [COMMANDS_LIST.md](COMMANDS_LIST.md)

---

*All plugins enhanced with nlsq-pro template pattern (v1.0.7) including Pre-Response Validation Framework and Constitutional AI principles.*
