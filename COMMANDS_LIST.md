# Claude Code Plugin Marketplace - Complete Command List

**Total Commands:** 48 across 23 plugins
**Format:** `/plugin:command` (slash prefix for execution)
**Version:** 1.0.4 | **Last Updated:** December 3, 2025

---

## Scientific Computing (4 commands)

### julia-development (4 commands)
- **/julia-development:julia-optimize** - Profile Julia code and provide optimization recommendations. Analyzes type stability, memory allocations, identifies bottlenecks, and suggests parallelization strategies
- **/julia-development:julia-package-ci** - Generate GitHub Actions CI/CD workflows for Julia packages. Configures testing matrices across Julia versions and platforms, coverage reporting, and documentation deployment
- **/julia-development:julia-scaffold** - Bootstrap new Julia package with proper structure following PkgTemplates.jl conventions. Creates Project.toml, testing infrastructure, documentation framework, and CI/CD
- **/julia-development:sciml-setup** - Interactive SciML project scaffolding with auto-detection of problem types (ODE, PDE, SDE, optimization). Generates template code with callbacks, ensemble simulations, and sensitivity analysis

---

## Development (14 commands)

### backend-development (1 command)
- **/backend-development:feature-development** - Orchestrate end-to-end feature development from requirements to production deployment with 3 execution modes (quick/standard/enterprise)

### frontend-mobile-development (1 command)
- **/frontend-mobile-development:component-scaffold** - Orchestrate production-ready React/React Native component generation with multi-mode execution (quick: analysis, standard: full implementation)

### javascript-typescript (1 command)
- **/javascript-typescript:typescript-scaffold** - Production-ready TypeScript project scaffolding with modern tooling, automated setup, and comprehensive configuration

### llm-application-dev (3 commands)
- **/llm-application-dev:ai-assistant** - Build AI assistants with LLMs, RAG, and conversational AI patterns for production-ready applications
- **/llm-application-dev:langchain-agent** - Create LangChain agents with tools, memory, and complex reasoning with LangGraph orchestration
- **/llm-application-dev:prompt-optimize** - Optimize prompts for better LLM performance through CoT, few-shot learning, and constitutional AI techniques

### multi-platform-apps (1 command)
- **/multi-platform-apps:multi-platform** - Build and deploy features across web, mobile, and desktop platforms with API-first architecture and 3 execution modes

### python-development (1 command)
- **/python-development:python-scaffold** - Scaffold production-ready Python projects with modern tooling, 3 execution modes (quick: 1-2h, standard: 3-6h, enterprise: 1-2d)

### systems-programming (3 commands)
- **/systems-programming:c-project** - Scaffold production-ready C projects with 3 execution modes (quick: CLI tool, standard: production app with Makefile/CMake/testing)
- **/systems-programming:profile-performance** - Comprehensive performance profiling workflow with 3 execution modes (quick: hotspot identification, standard: full analysis, comprehensive: optimization)
- **/systems-programming:rust-project** - Scaffold production-ready Rust projects with 3 execution modes (quick: simple binary/library, standard: production crate with async Tokio)

### full-stack-orchestration (1 command)
- **/full-stack-orchestration:full-stack-feature** - Orchestrate end-to-end full-stack feature development with multi-mode execution (quick: architecture, standard: implementation, deep: complete)

### agent-orchestration (2 commands)
- **/agent-orchestration:improve-agent** - Systematic agent improvement through 4-phase methodology (analysis, prompt engineering, testing, deployment) with A/B testing
- **/agent-orchestration:multi-agent-optimize** - Multi-agent code optimization with executable workflows (scan/analyze/apply modes) and graceful fallback

---

## AI & Machine Learning (3 commands)

### ai-reasoning (2 commands)
- **/ai-reasoning:ultra-think** - Advanced structured reasoning engine with step-by-step thought processing, branching logic, and dynamic adaptation for complex problem-solving
- **/ai-reasoning:reflection** - Advanced reflection engine for AI reasoning, session analysis, and research optimization with multi-agent orchestration

### machine-learning (1 command)
- **/machine-learning:ml-pipeline** - Design and implement production-ready ML pipelines with multi-agent MLOps orchestration and 3 execution modes

---

## DevOps & Infrastructure (8 commands)

### cicd-automation (2 commands)
- **/cicd-automation:fix-commit-errors** - Intelligent GitHub Actions failure resolution with 5-agent system, pattern matching across 100+ error types, and Bayesian confidence scoring
- **/cicd-automation:workflow-automate** - Automated CI/CD workflow generation with intelligent platform selection (GitHub Actions, GitLab CI, Terraform) and technology stack detection

### git-pr-workflows (4 commands)
- **/git-pr-workflows:commit** - Create intelligent atomic commits with automated analysis, quality validation, and conventional commit format
- **/git-pr-workflows:git-workflow** - Implement and optimize Git workflows with multi-agent orchestration, branching strategies (trunk-based, Git Flow, GitHub Flow), and merge patterns
- **/git-pr-workflows:onboard** - Onboard new team members with comprehensive 30-60-90 day plans, role-specific playbooks, and mentor guides
- **/git-pr-workflows:pr-enhance** - Enhance pull requests with automated analysis, review best practices, comprehensive templates, and quality guidelines

### observability-monitoring (2 commands)
- **/observability-monitoring:monitor-setup** - Set up comprehensive monitoring and observability stack with Prometheus, Grafana, and distributed tracing with 3 execution modes
- **/observability-monitoring:slo-implement** - Implement SLO/SLA monitoring, error budgets, and burn rate alerting with comprehensive governance framework

---

## Quality & Testing (10 commands)

### codebase-cleanup (4 commands)
- **/codebase-cleanup:deps-audit** - Comprehensive dependency security scanning and vulnerability analysis with multi-language support
- **/codebase-cleanup:fix-imports** - Systematically fix broken imports with intelligent resolution and session continuity
- **/codebase-cleanup:refactor-clean** - Refactor code for quality, maintainability, and SOLID principles with measurable improvements
- **/codebase-cleanup:tech-debt** - Analyze, prioritize, and create remediation plans for technical debt with ROI calculations

### comprehensive-review (2 commands)
- **/comprehensive-review:full-review** - Orchestrate comprehensive multi-dimensional code review using specialized agents with execution modes
- **/comprehensive-review:pr-enhance** - Create high-quality pull requests with comprehensive descriptions, automated review checks, and best practices

### quality-engineering (2 commands)
- **/quality-engineering:double-check** - Comprehensive multi-dimensional validation with 3 execution modes (quick: 30min-1h, standard: 2-4h, enterprise: 1-2d)
- **/quality-engineering:lint-plugins** - Validate Claude Code plugin syntax and structure with 3 execution modes (quick: syntax, standard: full, enterprise: deep)

### unit-testing (2 commands)
- **/unit-testing:run-all-tests** - Iteratively run and fix all tests with 3 execution modes (quick: max 3 iterations, standard: max 10 with AI-RCA, enterprise: full coverage)
- **/unit-testing:test-generate** - Generate comprehensive test suites with 3 execution modes (quick: single module, standard: package suite, enterprise: entire project)

---

## Tools & Migration (9 commands)

### code-documentation (4 commands)
- **/code-documentation:code-explain** - Detailed explanation of code structure, functionality, and design patterns with scientific computing support
- **/code-documentation:doc-generate** - Generate comprehensive, maintainable documentation from code with AI-powered analysis
- **/code-documentation:update-claudemd** - Automatically update CLAUDE.md file based on recent code changes
- **/code-documentation:update-docs** - Comprehensively update and optimize Sphinx docs, README, and related codebase documentation with AST-based content extraction

### code-migration (1 command)
- **/code-migration:adopt-code** - Comprehensive scientific code adoption and modernization workflow with systematic analysis and validation

### debugging-toolkit (1 command)
- **/debugging-toolkit:smart-debug** - AI-assisted debugging with automated root cause analysis, error pattern recognition, and hypothesis generation

### framework-migration (3 commands)
- **/framework-migration:code-migrate** - Orchestrate systematic code migration between frameworks with test-first discipline and multi-mode execution (quick/standard/deep)
- **/framework-migration:deps-upgrade** - Safe dependency upgrade orchestration with security-first prioritization, breaking change management, and incremental strategies
- **/framework-migration:legacy-modernize** - Comprehensive legacy modernization with Strangler Fig pattern, multi-agent orchestration, and zero-downtime migration

---

## Quick Reference by Plugin

| Plugin | Commands | Command Names |
|--------|----------|---------------|
| julia-development | 4 | julia-optimize, julia-package-ci, julia-scaffold, sciml-setup |
| git-pr-workflows | 4 | commit, git-workflow, onboard, pr-enhance |
| code-documentation | 4 | code-explain, doc-generate, update-claudemd, update-docs |
| codebase-cleanup | 4 | deps-audit, fix-imports, refactor-clean, tech-debt |
| llm-application-dev | 3 | ai-assistant, langchain-agent, prompt-optimize |
| systems-programming | 3 | c-project, profile-performance, rust-project |
| framework-migration | 3 | code-migrate, deps-upgrade, legacy-modernize |
| agent-orchestration | 2 | improve-agent, multi-agent-optimize |
| ai-reasoning | 2 | reflection, ultra-think |
| cicd-automation | 2 | fix-commit-errors, workflow-automate |
| comprehensive-review | 2 | full-review, pr-enhance |
| observability-monitoring | 2 | monitor-setup, slo-implement |
| quality-engineering | 2 | double-check, lint-plugins |
| unit-testing | 2 | run-all-tests, test-generate |
| backend-development | 1 | feature-development |
| code-migration | 1 | adopt-code |
| debugging-toolkit | 1 | smart-debug |
| frontend-mobile-development | 1 | component-scaffold |
| full-stack-orchestration | 1 | full-stack-feature |
| javascript-typescript | 1 | typescript-scaffold |
| machine-learning | 1 | ml-pipeline |
| multi-platform-apps | 1 | multi-platform |
| python-development | 1 | python-scaffold |

**Total: 48 commands across 23 plugins**

---

## Usage Examples

### Execute a command
```bash
# Basic command execution
/ai-reasoning:ultra-think "How do we optimize our database queries?"

# Command with arguments
/unit-testing:run-all-tests tests/ --fix --coverage

# Command with execution mode
/quality-engineering:double-check --mode=standard --security --performance
```

### Common Workflow Patterns

#### Feature Development
```bash
/backend-development:feature-development "user authentication with OAuth2"
/unit-testing:test-generate src/auth/
/comprehensive-review:full-review --security-focus
/git-pr-workflows:commit --split
```

#### Code Quality
```bash
/codebase-cleanup:deps-audit
/codebase-cleanup:fix-imports
/codebase-cleanup:tech-debt
/quality-engineering:double-check --mode=standard
```

#### Scientific Computing
```bash
/julia-development:sciml-setup "Solve stiff ODEs for chemical reactions"
/julia-development:julia-optimize src/solver.jl
/unit-testing:test-generate --scientific
```

#### CI/CD & DevOps
```bash
/cicd-automation:workflow-automate
/cicd-automation:fix-commit-errors --auto-fix
/observability-monitoring:monitor-setup
/observability-monitoring:slo-implement
```

---

## Execution Modes

Most commands support three execution modes:

| Mode | Time Estimate | Description |
|------|---------------|-------------|
| **quick** | 30min - 2h | Fast analysis, syntax checking, basic scaffolding |
| **standard** | 2-6h | Full implementation with testing and documentation |
| **enterprise/comprehensive** | 1-2d | Complete solution with advanced features and compliance |

Use `--mode=<mode>` to specify the execution mode:
```bash
/quality-engineering:double-check --mode=quick    # Fast validation
/python-development:python-scaffold --mode=standard  # Full project setup
```

---

## Command Categories Summary

| Category | Commands | Primary Use Cases |
|----------|----------|------------------|
| **Scientific Computing** | 4 | Julia development, SciML, optimization, package scaffolding |
| **Development** | 14 | Project scaffolding, backend/frontend, LLM apps, systems programming |
| **AI/ML** | 3 | Machine learning pipelines, structured reasoning, reflection |
| **DevOps** | 8 | CI/CD automation, git workflows, monitoring, SLO implementation |
| **Quality & Testing** | 10 | Test automation, code review, cleanup, quality validation |
| **Tools & Migration** | 9 | Documentation, code migration, framework upgrades, debugging |

---

## Resources

- **Plugin Cheatsheet:** [PLUGIN_CHEATSHEET.md](PLUGIN_CHEATSHEET.md)
- **Agent List:** [AGENTS_LIST.md](AGENTS_LIST.md)
- **Full Documentation:** [https://myclaude.readthedocs.io/en/latest/](https://myclaude.readthedocs.io/en/latest/)

**Marketplace Stats:** 31 plugins | 74 agents | 48 commands | 114 skills

---

*Generated from v1.0.4 validated marketplace data. All commands follow consistent naming conventions and include comprehensive functional descriptions.*
