# Plugin Installation Reference

Complete list of all 31 plugins with installation commands.

## Quick Install Commands

### Install All Plugins at Once

```bash
# Option 1: Use the make command (recommended)
make plugin-enable-all

# Option 2: Use Python script directly
python3 scripts/enable-all-plugins.py
```

### Install Individual Plugins

Copy and paste the commands for the plugins you want:

## AI/ML (5 plugins)

```bash
/plugin install agent-orchestration@scientific-computing-workflows
/plugin install ai-reasoning@scientific-computing-workflows
/plugin install deep-learning@scientific-computing-workflows
/plugin install jax-implementation@scientific-computing-workflows
/plugin install machine-learning@scientific-computing-workflows
```

## Development (8 plugins)

```bash
/plugin install backend-development@scientific-computing-workflows
/plugin install debugging-toolkit@scientific-computing-workflows
/plugin install frontend-mobile-development@scientific-computing-workflows
/plugin install javascript-typescript@scientific-computing-workflows
/plugin install llm-application-dev@scientific-computing-workflows
/plugin install multi-platform-apps@scientific-computing-workflows
/plugin install python-development@scientific-computing-workflows
/plugin install systems-programming@scientific-computing-workflows
```

## Scientific Computing (4 plugins)

```bash
/plugin install hpc-computing@scientific-computing-workflows
/plugin install julia-development@scientific-computing-workflows
/plugin install molecular-simulation@scientific-computing-workflows
/plugin install statistical-physics@scientific-computing-workflows
```

## Tools & Utilities (11 plugins)

```bash
/plugin install cicd-automation@scientific-computing-workflows
/plugin install cli-tool-design@scientific-computing-workflows
/plugin install code-documentation@scientific-computing-workflows
/plugin install code-migration@scientific-computing-workflows
/plugin install codebase-cleanup@scientific-computing-workflows
/plugin install comprehensive-review@scientific-computing-workflows
/plugin install framework-migration@scientific-computing-workflows
/plugin install git-pr-workflows@scientific-computing-workflows
/plugin install observability-monitoring@scientific-computing-workflows
/plugin install quality-engineering@scientific-computing-workflows
/plugin install unit-testing@scientific-computing-workflows
```

## Research & Visualization (3 plugins)

```bash
/plugin install data-visualization@scientific-computing-workflows
/plugin install research-methodology@scientific-computing-workflows
/plugin install full-stack-orchestration@scientific-computing-workflows
```

## Install All 31 Plugins via CLI

If you prefer to install all plugins using CLI commands:

```bash
/plugin install agent-orchestration@scientific-computing-workflows
/plugin install ai-reasoning@scientific-computing-workflows
/plugin install backend-development@scientific-computing-workflows
/plugin install cicd-automation@scientific-computing-workflows
/plugin install cli-tool-design@scientific-computing-workflows
/plugin install code-documentation@scientific-computing-workflows
/plugin install code-migration@scientific-computing-workflows
/plugin install codebase-cleanup@scientific-computing-workflows
/plugin install comprehensive-review@scientific-computing-workflows
/plugin install data-visualization@scientific-computing-workflows
/plugin install debugging-toolkit@scientific-computing-workflows
/plugin install deep-learning@scientific-computing-workflows
/plugin install framework-migration@scientific-computing-workflows
/plugin install frontend-mobile-development@scientific-computing-workflows
/plugin install full-stack-orchestration@scientific-computing-workflows
/plugin install git-pr-workflows@scientific-computing-workflows
/plugin install hpc-computing@scientific-computing-workflows
/plugin install jax-implementation@scientific-computing-workflows
/plugin install javascript-typescript@scientific-computing-workflows
/plugin install julia-development@scientific-computing-workflows
/plugin install llm-application-dev@scientific-computing-workflows
/plugin install machine-learning@scientific-computing-workflows
/plugin install molecular-simulation@scientific-computing-workflows
/plugin install multi-platform-apps@scientific-computing-workflows
/plugin install observability-monitoring@scientific-computing-workflows
/plugin install python-development@scientific-computing-workflows
/plugin install quality-engineering@scientific-computing-workflows
/plugin install research-methodology@scientific-computing-workflows
/plugin install statistical-physics@scientific-computing-workflows
/plugin install systems-programming@scientific-computing-workflows
/plugin install unit-testing@scientific-computing-workflows
```

## Plugin Descriptions

### AI/ML
- **agent-orchestration** - Multi-agent system optimization and context management
- **ai-reasoning** - Advanced cognitive tools for problem-solving and meta-analysis
- **deep-learning** - Neural network architecture design and training diagnostics
- **jax-implementation** - JAX programming with Flax NNX and physics applications
- **machine-learning** - Data science, statistical analysis, and MLOps workflows

### Development
- **backend-development** - Backend API design, GraphQL architecture, TDD
- **debugging-toolkit** - Interactive debugging and developer experience optimization
- **frontend-mobile-development** - Frontend UI and mobile app implementation
- **javascript-typescript** - Modern JavaScript/TypeScript with ES6+ and Node.js
- **llm-application-dev** - LLM applications with prompt engineering and RAG
- **multi-platform-apps** - Cross-platform web, iOS, Android, desktop apps
- **python-development** - Python with FastAPI, Django, async patterns
- **systems-programming** - Rust, C, C++, and Go development

### Scientific Computing
- **hpc-computing** - High-performance computing and numerical methods
- **julia-development** - Julia ecosystem with SciML and Bayesian inference
- **molecular-simulation** - Molecular dynamics with LAMMPS and GROMACS
- **statistical-physics** - Correlation function analysis and FFT

### Tools & Utilities
- **cicd-automation** - CI/CD pipelines with GitHub Actions and GitLab CI
- **cli-tool-design** - CLI tool development and automation
- **code-documentation** - Documentation generation and technical writing
- **code-migration** - Legacy code modernization and migration
- **codebase-cleanup** - Technical debt reduction and refactoring
- **comprehensive-review** - Multi-perspective code analysis and security
- **framework-migration** - Framework updates and architectural transformation
- **git-pr-workflows** - Git workflow automation and pull request enhancement
- **observability-monitoring** - Metrics, logging, tracing, and SLO implementation
- **quality-engineering** - QA, validation, and correctness verification
- **unit-testing** - Test automation, generation, and execution

### Research & Visualization
- **data-visualization** - Scientific data visualization and AR/VR interfaces
- **research-methodology** - Research intelligence and literature analysis
- **full-stack-orchestration** - End-to-end full-stack feature delivery

## After Installation

**Remember to restart Claude Code** after installing plugins for changes to take effect.

### Verify Installation

```bash
# Check available plugins
make plugin-list

# Check plugin statistics
make plugin-count

# View all available commands
make help
```

### Using Installed Plugins

Once installed, you can:

- **Use specialized agents**: `@python-pro`, `@julia-pro`, `@jax-pro`, `@ml-engineer`
- **Run commands**: `/ultra-think`, `/double-check`, `/full-stack-feature`
- **Access skills**: Skills are automatically loaded and available

Example usage:
```
Ask Claude: "@python-pro help me optimize this async function"
Ask Claude: "@julia-pro implement this differential equation using SciML"
Ask Claude: "/ultra-think analyze the architecture of this system"
```
