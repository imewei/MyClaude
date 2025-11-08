# MyClaude - Production-Grade Claude Code Plugins

[![Plugins](https://img.shields.io/badge/Plugins-31-blue.svg)](https://myclaude.readthedocs.io/en/latest/plugins/)
[![Agents](https://img.shields.io/badge/Agents-75-green.svg)](AGENTS_LIST.md)
[![Commands](https://img.shields.io/badge/Commands-60+-orange.svg)](COMMANDS_LIST.md)
[![Version](https://img.shields.io/badge/Version-1.0.3-red.svg)](https://github.com/imewei/MyClaude)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/Docs-ReadTheDocs-brightgreen.svg)](https://myclaude.readthedocs.io/en/latest/)

Comprehensive Claude Code plugin marketplace with **31 specialized plugins**, **75 expert agents**, and **60+ slash commands** for AI-powered development, scientific computing, and research workflows.

## Full Documentation

**[View Complete Plugin Documentation →](https://myclaude.readthedocs.io/en/latest/)**

For comprehensive documentation including detailed plugin guides, integration patterns, quick-start tutorials, and API references, visit the full Sphinx documentation.

To build documentation locally:

```bash
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

### What's New in v1.0.3

**Major Optimization Release** (21 plugins updated)

- **Command Optimization**: 25-73% token reduction across plugins with hub-and-spoke architecture
- **Execution Modes**: All commands now support quick/standard/comprehensive (or enterprise) modes with time estimates
- **External Documentation**: ~30,000+ lines of comprehensive reference guides across all plugins
- **Enhanced YAML Frontmatter**: Structured metadata for better command discovery and user experience
- **100% Backward Compatibility**: All optimizations maintain full compatibility with previous versions

**Key Highlights by Plugin:**

- **ai-reasoning**: 46% command optimization (reflection, ultra-think)
- **code-documentation**: 54% reduction with AST-based analysis
- **cicd-automation**: 62% optimization with multi-agent error resolution
- **quality-engineering**: 8 external docs (~6,455 lines) for validation frameworks
- **systems-programming**: 8 external docs (~5,602 lines) for C/C++/Rust/Go
- **unit-testing**: 50% reduction with execution modes and external docs (~5,858 lines)
- **backend-development**: Enhanced feature-development with 3 execution modes
- **machine-learning**: Enhanced ml-pipeline with data-engineer agent

## Statistics

- **Total Plugins**: 31
- **Total Agents**: 75
- **Total Commands**: 60+
- **Total Skills**: 110+
- **Categories**: 6 major categories

### Version Distribution

- **v1.0.3**: 21 plugins (latest optimization release)
- **v1.0.2**: 1 plugin (cli-tool-design)
- **v1.0.2**: 1 plugin (jax-implementation)
- **v1.0.1**: 8 plugins (scientific computing)

## Plugin Categories

### 🤖 AI & Reasoning (2 plugins)

- **[ai-reasoning](plugins/ai-reasoning/)** (v1.0.3) - Advanced structured reasoning with ultra-think and reflection engines. 46% command optimization with execution modes
- **[agent-orchestration](plugins/agent-orchestration/)** (v1.0.3) - Multi-agent workflow coordination and context management with version consolidation

### 💻 Development & Engineering (10 plugins)

- **[backend-development](plugins/backend-development/)** (v1.0.3) - REST/GraphQL/gRPC APIs, microservices, TDD orchestration
- **[frontend-mobile-development](plugins/frontend-mobile-development/)** (v1.0.3) - React 19, Next.js 15, React Native, Flutter development
- **[full-stack-orchestration](plugins/full-stack-orchestration/)** (v1.0.3) - End-to-end feature delivery with multi-agent coordination
- **[multi-platform-apps](plugins/multi-platform-apps/)** (v1.0.3) - Cross-platform web, iOS, Android, desktop apps
- **[python-development](plugins/python-development/)** (v1.0.3) - Python 3.12+ with FastAPI, Django, async patterns
- **[javascript-typescript](plugins/javascript-typescript/)** (v1.0.3) - Modern JS/TS with 25% command optimization
- **[systems-programming](plugins/systems-programming/)** (v1.0.3) - C, C++, Rust, Go with 8 external docs (~5,602 lines)
- **[julia-development](plugins/julia-development/)** (v1.0.3) - Julia ecosystem with SciML and Bayesian inference
- **[llm-application-dev](plugins/llm-application-dev/)** (v1.0.3) - LLM apps with prompt engineering and RAG
- **[cli-tool-design](plugins/cli-tool-design/)** (v1.0.2) - CLI tool design and developer automation

### 🔬 Scientific Computing (8 plugins)

- **[jax-implementation](plugins/jax-implementation/)** (v1.0.2) - JAX with NumPyro, Flax NNX, NLSQ optimization
- **[hpc-computing](plugins/hpc-computing/)** (v1.0.1) - High-performance computing and numerical methods
- **[molecular-simulation](plugins/molecular-simulation/)** (v1.0.1) - MD with LAMMPS, GROMACS, HOOMD-blue
- **[statistical-physics](plugins/statistical-physics/)** (v1.0.1) - Non-equilibrium systems and correlation functions
- **[deep-learning](plugins/deep-learning/)** (v1.0.1) - Neural networks with systematic frameworks
- **[data-visualization](plugins/data-visualization/)** (v1.0.1) - Scientific plots with Matplotlib, Plotly, Makie
- **[research-methodology](plugins/research-methodology/)** (v1.0.1) - Research intelligence and literature analysis
- **[machine-learning](plugins/machine-learning/)** (v1.0.3) - MLOps with data-engineer agent and ml-pipeline enhancements

### ✅ Quality & Testing (4 plugins)

- **[quality-engineering](plugins/quality-engineering/)** (v1.0.3) - Comprehensive validation with 8 external docs (~6,455 lines)
- **[unit-testing](plugins/unit-testing/)** (v1.0.3) - Test automation with 50% command optimization
- **[comprehensive-review](plugins/comprehensive-review/)** (v1.0.3) - Multi-agent code review with execution modes
- **[debugging-toolkit](plugins/debugging-toolkit/)** (v1.0.3) - AI-assisted debugging with RCA frameworks

### 🔧 Code Maintenance & Migration (4 plugins)

- **[codebase-cleanup](plugins/codebase-cleanup/)** (v1.0.3) - Technical debt reduction with 9 external docs
- **[code-documentation](plugins/code-documentation/)** (v1.0.3) - 54% optimization with AST-based analysis
- **[code-migration](plugins/code-migration/)** (v1.0.3) - Scientific code modernization (Fortran/MATLAB → Python/JAX)
- **[framework-migration](plugins/framework-migration/)** (v1.0.3) - Framework upgrades with strangler fig patterns

### ⚙️ DevOps & Infrastructure (3 plugins)

- **[cicd-automation](plugins/cicd-automation/)** (v1.0.3) - 62% optimization with intelligent error resolution
- **[git-pr-workflows](plugins/git-pr-workflows/)** (v1.0.3) - Git workflows and PR enhancement
- **[observability-monitoring](plugins/observability-monitoring/)** (v1.0.3) - Prometheus, Grafana, distributed tracing

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
python3 scripts/enable-all-plugins.py
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
/ultra-think "Analyze the architecture of this system" --depth=deep
/double-check my-feature --mode=standard
/run-all-tests --fix --coverage
/fix-commit-errors workflow-123 --auto-fix
```

**Accessing Skills**
Skills are automatically loaded based on file context and your requests.

## Documentation

- **[Plugin Cheatsheet](PLUGIN_CHEATSHEET.md)** - Quick reference for all plugins
- **[Complete Agents List](AGENTS_LIST.md)** - Catalog of all 75 agents
- **[Complete Commands List](COMMANDS_LIST.md)** - Catalog of all 60+ commands
- **[Full Documentation](https://myclaude.readthedocs.io/en/latest/)** - Comprehensive guides and API references

## Version History

### v1.0.3 (Current - 2025-11-08)

**Major Optimization Release** - 21 plugins updated with significant enhancements:

**Command Optimization Highlights:**
- **ai-reasoning**: 46.5% token reduction (reflection.md 1704→695 lines, ultra-think.md 1288→906 lines)
- **cicd-automation**: 62.1% reduction (2,391→906 lines) with multi-agent error analysis
- **code-documentation**: 54.1% reduction (2,495→1,146 lines) with AST-based analysis
- **unit-testing**: 50.6% reduction (2,112→1,044 lines) with execution modes
- **codebase-cleanup**: 25% reduction (2,608→1,965 lines) with 9 external docs

**New Features:**
- **Execution Modes**: All commands support quick/standard/comprehensive (or enterprise) modes with time estimates
- **External Documentation**: ~30,000+ lines of comprehensive guides across plugins
- **Enhanced YAML Frontmatter**: Structured metadata for better command discovery
- **100% Backward Compatibility**: All optimizations maintain full compatibility

**Plugin Updates:**
- agent-orchestration (v1.0.3): Version consolidation
- backend-development (v1.0.3): Enhanced feature-development with 3 execution modes
- machine-learning (v1.0.3): Added data-engineer agent, ml-pipeline enhancements
- quality-engineering (v1.0.3): 8 external docs (~6,455 lines)
- systems-programming (v1.0.3): 8 external docs (~5,602 lines)
- And 16 more plugins with similar enhancements

### v1.0.2 (2025-10-31)

- **jax-implementation v1.0.2**: Enhanced NumPyro with ArviZ integration, Consensus Monte Carlo, and response verification

### v1.0.1 (2025-10-31)

- Documentation improvements and enhanced installation guide
- Updated statistics (75 agents, 60+ commands, 110+ skills)
- Fixed Sphinx warnings

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
