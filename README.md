# MyClaude

[![Plugins](https://img.shields.io/badge/Plugins-31-blue.svg)](https://myclaude.readthedocs.io/en/latest/plugins/)
[![Agents](https://img.shields.io/badge/Agents-74-green.svg)](AGENTS_LIST.md)
[![Commands](https://img.shields.io/badge/Commands-49-orange.svg)](COMMANDS_LIST.md)
[![Skills](https://img.shields.io/badge/Skills-117-purple.svg)](PLUGIN_CHEATSHEET.md)
[![Tools](https://img.shields.io/badge/Tools-16-teal.svg)](docs/tools-reference.rst)
[![Version](https://img.shields.io/badge/Version-1.0.7-red.svg)](https://github.com/imewei/MyClaude)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/Docs-ReadTheDocs-brightgreen.svg)](https://myclaude.readthedocs.io/en/latest/)

Claude Code plugin marketplace with **31 specialized plugins**, **74 expert agents**, **49 slash commands**, **117 skills**, and **16 tools** for AI-powered development, scientific computing, and research workflows.

## Full Documentation

**[View Complete Plugin Documentation](https://myclaude.readthedocs.io/en/latest/)**

For comprehensive documentation including detailed plugin guides, integration patterns, quick-start tutorials, and API references, visit the full Sphinx documentation.

```bash
# Build documentation locally
cd docs/
make html
open _build/html/index.html
```

## Overview

The MyClaude plugin ecosystem provides production-ready tools for:

- **AI-Powered Development**: Advanced reasoning, multi-agent orchestration, and LLM application development
- **Full-Stack Engineering**: Backend APIs, frontend/mobile apps, multi-platform development
- **Scientific Computing**: HPC, molecular simulation, JAX/Julia workflows, deep learning
- **Quality Engineering**: Comprehensive testing, security auditing, code review
- **DevOps & Infrastructure**: CI/CD automation, observability, Kubernetes orchestration
- **Code Intelligence**: Documentation generation, migration, cleanup, debugging

## What's New in v1.0.7

**Optimization & Documentation Release** (All 31 plugins updated)

All plugins synchronized to version 1.0.7:

- **Plugin Optimization**: 40-76% token reduction across all plugins for faster loading
- **New Command**: Added `/merge-all` to git-pr-workflows for branch consolidation
- **API Documentation**: New `docs/api/` reference for Python tools
- **Enhanced CI/CD**: Documentation workflow with coverage and linkcheck
- **49 Commands**: Total commands increased from 48 to 49

## Statistics

| Metric | Count |
|--------|-------|
| Plugins | 31 |
| Agents | 74 |
| Commands | 49 |
| Skills | 117 |
| Tools | 16 |
| Categories | 6 |

### Category Breakdown

| Category | Plugins | Agents | Commands | Skills |
|----------|---------|--------|----------|--------|
| Scientific Computing | 8 | 18 | 4 | 54 |
| Development | 10 | 24 | 14 | 30 |
| AI & Machine Learning | 2 | 6 | 3 | 10 |
| DevOps & Infrastructure | 3 | 10 | 8 | 12 |
| Quality & Testing | 4 | 7 | 10 | 3 |
| Tools & Migration | 4 | 9 | 9 | 7 |

## Plugin Categories

### Scientific Computing (8 plugins)

| Plugin | Version | Agents | Commands | Description |
|--------|---------|--------|----------|-------------|
| [julia-development](plugins/julia-development/) | v1.0.7 | 4 | 4 | Julia ecosystem with SciML, Turing.jl, and package development |
| [jax-implementation](plugins/jax-implementation/) | v1.0.7 | 4 | - | JAX with NumPyro, Flax NNX, NLSQ optimization |
| [hpc-computing](plugins/hpc-computing/) | v1.0.7 | 1 | - | High-performance computing and numerical methods |
| [molecular-simulation](plugins/molecular-simulation/) | v1.0.7 | 1 | - | MD with LAMMPS, GROMACS, HOOMD-blue |
| [statistical-physics](plugins/statistical-physics/) | v1.0.7 | 2 | - | Non-equilibrium systems and correlation functions |
| [deep-learning](plugins/deep-learning/) | v1.0.7 | 2 | - | Neural networks with systematic frameworks |
| [data-visualization](plugins/data-visualization/) | v1.0.7 | 1 | - | Scientific plots with Matplotlib, Plotly, Makie |
| [research-methodology](plugins/research-methodology/) | v1.0.7 | 1 | - | Research intelligence and literature analysis |

### Development (10 plugins)

| Plugin | Version | Agents | Commands | Description |
|--------|---------|--------|----------|-------------|
| [python-development](plugins/python-development/) | v1.0.7 | 3 | 1 | Python 3.12+ with FastAPI, Django, async patterns |
| [backend-development](plugins/backend-development/) | v1.0.7 | 3 | 1 | REST/GraphQL/gRPC APIs, microservices, TDD |
| [frontend-mobile-development](plugins/frontend-mobile-development/) | v1.0.7 | 2 | 1 | React 19, Next.js 15, React Native, Flutter |
| [javascript-typescript](plugins/javascript-typescript/) | v1.0.7 | 2 | 1 | Modern JS/TS with ES2024 and Node.js |
| [systems-programming](plugins/systems-programming/) | v1.0.7 | 4 | 3 | C, C++, Rust, Go systems programming |
| [multi-platform-apps](plugins/multi-platform-apps/) | v1.0.7 | 6 | 1 | Cross-platform web, iOS, Android, desktop apps |
| [llm-application-dev](plugins/llm-application-dev/) | v1.0.7 | 2 | 3 | LLM apps with prompt engineering and RAG |
| [cli-tool-design](plugins/cli-tool-design/) | v1.0.7 | 1 | - | CLI tool design and developer automation |
| [full-stack-orchestration](plugins/full-stack-orchestration/) | v1.0.7 | 4 | 1 | End-to-end feature delivery with multi-agent coordination |
| [agent-orchestration](plugins/agent-orchestration/) | v1.0.7 | 2 | 2 | Multi-agent workflow coordination and context management |

### AI & Machine Learning (2 plugins)

| Plugin | Version | Agents | Commands | Description |
|--------|---------|--------|----------|-------------|
| [machine-learning](plugins/machine-learning/) | v1.0.7 | 4 | 1 | MLOps with data engineering and ML pipelines |
| [ai-reasoning](plugins/ai-reasoning/) | v1.0.7 | - | 2 | Advanced reasoning with ultra-think and reflection |

### DevOps & Infrastructure (3 plugins)

| Plugin | Version | Agents | Commands | Description |
|--------|---------|--------|----------|-------------|
| [cicd-automation](plugins/cicd-automation/) | v1.0.7 | 5 | 2 | CI/CD with intelligent error resolution |
| [git-pr-workflows](plugins/git-pr-workflows/) | v1.0.7 | 1 | 4 | Git workflows and PR enhancement |
| [observability-monitoring](plugins/observability-monitoring/) | v1.0.7 | 4 | 2 | Prometheus, Grafana, distributed tracing |

### Quality & Testing (4 plugins)

| Plugin | Version | Agents | Commands | Description |
|--------|---------|--------|----------|-------------|
| [unit-testing](plugins/unit-testing/) | v1.0.7 | 2 | 2 | Test automation with AI-powered debugging |
| [comprehensive-review](plugins/comprehensive-review/) | v1.0.7 | 3 | 2 | Multi-agent code review with security auditing |
| [codebase-cleanup](plugins/codebase-cleanup/) | v1.0.7 | 2 | 4 | Technical debt reduction and refactoring |
| [quality-engineering](plugins/quality-engineering/) | v1.0.7 | - | 2 | Comprehensive validation frameworks |

### Tools & Migration (4 plugins)

| Plugin | Version | Agents | Commands | Description |
|--------|---------|--------|----------|-------------|
| [code-documentation](plugins/code-documentation/) | v1.0.7 | 3 | 4 | AI-powered documentation with AST analysis |
| [code-migration](plugins/code-migration/) | v1.0.7 | 1 | 1 | Scientific code modernization (Fortran/MATLAB to Python/JAX) |
| [framework-migration](plugins/framework-migration/) | v1.0.7 | 2 | 3 | Framework upgrades with strangler fig patterns |
| [debugging-toolkit](plugins/debugging-toolkit/) | v1.0.7 | 2 | 1 | AI-assisted debugging with RCA frameworks |

## Quick Start

### Installation

#### Step 1: Add the Marketplace

In Claude Code, add this marketplace:

```bash
/plugin marketplace add imewei/MyClaude
```

#### Step 2: Install Plugins

**Option A: Browse and Install via UI**

1. Select "Browse and install plugins"
2. Select "scientific-computing-workflows"
3. Select the plugin you want to install
4. Select "Install now"

**Option B: Install Specific Plugins via CLI**

```bash
# Install a single plugin
/plugin install plugin-name@scientific-computing-workflows

# Examples:
/plugin install ai-reasoning@scientific-computing-workflows
/plugin install quality-engineering@scientific-computing-workflows
/plugin install backend-development@scientific-computing-workflows
```

**Option C: Install All 31 Plugins at Once**

```bash
# Clone the repository first
git clone https://github.com/imewei/MyClaude.git
cd MyClaude

# Enable all plugins
make plugin-enable-all

# Or use the Python script directly
python3 tools/enable-all-plugins.py
```

**Note:** After installation, restart Claude Code for changes to take effect.

#### Verify Installation

```bash
# From the cloned repository
make plugin-count   # Show plugin statistics
make plugin-list    # List all plugins with versions
```

### Using Plugins

Once installed, plugins provide agents, commands, and skills that are automatically available:

**Using Specialized Agents**
```
Ask Claude: "@python-pro help me optimize this async function"
Ask Claude: "@julia-pro implement this differential equation using SciML"
Ask Claude: "@jax-pro optimize this neural network training loop"
Ask Claude: "@rust-pro refactor this code for better memory safety"
```

**Running Commands**
```bash
/ai-reasoning:ultra-think "Analyze the architecture of this system" --depth=deep
/quality-engineering:double-check my-feature --mode=standard
/unit-testing:run-all-tests --fix --coverage
/cicd-automation:fix-commit-errors workflow-123 --auto-fix
```

**Accessing Skills**
Skills are automatically loaded based on file context and your requests.

## Tools

The marketplace includes 16 Python utilities for plugin management, validation, and profiling:

| Tool | Purpose | Target |
|------|---------|--------|
| `activation-profiler.py` | Measure agent activation time | <50ms |
| `load-profiler.py` | Measure plugin load time | <100ms |
| `memory-analyzer.py` | Profile memory consumption | <5MB |
| `metadata-validator.py` | Validate plugin.json schema | 100% |
| `skill-validator.py` | Test skill pattern matching | <5% over-trigger |
| `plugin-review-script.py` | Comprehensive validation | All pass |

See [Tools Reference](docs/tools-reference.rst) for complete documentation.

## Documentation

- **[Plugin Cheatsheet](PLUGIN_CHEATSHEET.md)** - Quick reference for all plugins
- **[Complete Agents List](AGENTS_LIST.md)** - Catalog of all 74 agents
- **[Complete Commands List](COMMANDS_LIST.md)** - Catalog of all 48 commands
- **[Tools Reference](docs/tools-reference.rst)** - 16 utility scripts documentation
- **[API Reference](docs/api/index.rst)** - Python tools API documentation
- **[Full Documentation](https://myclaude.readthedocs.io/en/latest/)** - Comprehensive guides and API references

### LaTeX Reference Documents

Printable reference documents are available in `docs/guides/`:

- `AGENTS_LIST.tex` - Complete 74-agent reference guide
- `COMMANDS_LIST.tex` - Complete 48-command reference guide
- `agents-reference.tex` - Concise agent reference
- `commands-reference.tex` - Concise command reference
- `plugin-cheatsheet.tex` - Quick reference cheatsheet (landscape)

To compile LaTeX documents:

```bash
cd docs/guides/
pdflatex AGENTS_LIST.tex
pdflatex COMMANDS_LIST.tex
```

## Development Commands

```bash
# Build Sphinx documentation
make docs                    # Build to docs/_build/html/
make docs-live               # Live server with auto-reload

# Quality checks
make lint                    # Run ruff and mypy
make format                  # Format with black and ruff
make validate                # Validate plugin metadata

# Testing
make test                    # Run pytest
make test-coverage           # Run with coverage report

# Plugin management
make plugin-count            # Show plugin statistics
make plugin-list             # List all plugins with versions
make plugin-enable-all       # Enable all plugins in Claude Code

# Cleanup
make clean                   # Clean Python artifacts and cache
make clean-all               # Deep clean including docs
```

## Version History

### v1.0.7 (Current - 2026-01-12)

**Optimization & Documentation Release** - All 31 plugins synchronized to 1.0.7:

- Plugin Optimization: 40-76% token reduction across all plugins
- New Command: Added /merge-all for branch consolidation
- API Documentation: New docs/api/ reference for Python tools
- Theme Migration: Switched Sphinx from RTD to Furo theme
- Enhanced CI/CD: Documentation workflow with coverage and linkcheck

### v1.0.4 (2025-12-03)

**Agent Optimization Release** - All 74 agents enhanced with nlsq-pro template:

- Pre-Response Validation Framework: 5 mandatory self-checks + 5 quality gates
- When to Invoke Sections: USE FOR / DELEGATE TO tables and Decision Trees
- Enhanced Constitutional AI Principles: Target %, Self-Check Questions, Anti-Patterns, Metrics
- Version Consistency: All agents bumped to v1.0.4

### v1.0.3 (2025-11-08)

**Command Optimization Release** - 21 plugins updated:

- 25-73% token reduction across plugins with hub-and-spoke architecture
- Execution modes: quick/standard/comprehensive with time estimates
- ~30,000+ lines of external documentation across plugins
- Enhanced YAML frontmatter for better command discovery

### v1.0.2 (2025-10-31)

- jax-implementation v1.0.2: Enhanced NumPyro with ArviZ integration

### v1.0.1 (2025-10-31)

- Documentation improvements and enhanced installation guide
- Updated statistics (74 agents, 48 commands, 114 skills)

### v1.0.0 (2025-10-29)

- Initial release with 31 specialized plugins
- Comprehensive coverage for scientific computing and software development

See [changelog](https://myclaude.readthedocs.io/en/latest/changelog.html) for detailed version history.

## Contributing

1. Fork and create a branch
2. Make changes to plugins
3. Test thoroughly with Claude Code
4. Submit PR with detailed description

See [contribution guidelines](https://myclaude.readthedocs.io/en/latest/contributing.html) for details.

## License

MIT License (see [LICENSE](LICENSE))

---

**Built by Wei Chen** | [Documentation](https://myclaude.readthedocs.io/en/latest/) | [GitHub](https://github.com/imewei/MyClaude)
