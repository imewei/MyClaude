# Scientific Computing Workflows Marketplace

Comprehensive Claude Code marketplace with 31 specialized plugins for scientific computing, software development, and research workflows.

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

This marketplace provides:
- **31 specialized plugins** across 9 categories
- **Comprehensive coverage** for scientific computing and software development
- **Independent modification** without affecting source plugins
- **Git-based version control** for all customizations
- **Integrated ecosystem** with cross-plugin collaboration

## Statistics

- **Total Plugins:** 31
- **Categories:** 9 (Scientific Computing, Development, DevOps, AI/ML, Tools, Orchestration, Quality)
- **Total Agents:** 73 | **Commands:** 48 | **Skills:** 110

## Categories

- **Scientific Computing (2)**: HPC, molecular simulation, numerical methods
- **Development (7)**: Full-stack, backend, frontend/mobile, Python, JavaScript/TypeScript, CLI
- **DevOps (2)**: CI/CD automation, observability & monitoring
- **AI/ML (2)**: Deep learning, machine learning pipelines
- **Tools (14)**: Code quality, documentation, migration, testing, debugging
- **Orchestration (1)**: Multi-agent workflow coordination
- **Quality (1)**: Quality engineering and testing

## Featured Plugins

- **julia-development**: Comprehensive Julia ecosystem support with SciML
- **jax-implementation**: JAX with NumPyro Bayesian inference, ArviZ diagnostics, and response verification (v1.0.2)
- **python-development**: Modern Python patterns and async workflows
- **comprehensive-review**: Multi-perspective code analysis
- **agent-orchestration**: Multi-agent system optimization

[Browse all plugins →](https://myclaude.readthedocs.io/en/latest/plugins/)

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

Install individual plugins using the CLI:

```bash
# Install a single plugin
/plugin install plugin-name@scientific-computing-workflows

# Examples:
/plugin install python-development@scientific-computing-workflows
/plugin install julia-development@scientific-computing-workflows
/plugin install deep-learning@scientific-computing-workflows
```

**Option C: Install All 31 Plugins at Once**

Use the convenience script to enable all plugins:

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

### Quick Installation Reference

#### Install Plugins by Category

**AI/ML (5 plugins)**
```bash
/plugin install agent-orchestration@scientific-computing-workflows
/plugin install ai-reasoning@scientific-computing-workflows
/plugin install deep-learning@scientific-computing-workflows
/plugin install jax-implementation@scientific-computing-workflows
/plugin install machine-learning@scientific-computing-workflows
```

**Development (8 plugins)**
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

**Scientific Computing (4 plugins)**
```bash
/plugin install hpc-computing@scientific-computing-workflows
/plugin install julia-development@scientific-computing-workflows
/plugin install molecular-simulation@scientific-computing-workflows
/plugin install statistical-physics@scientific-computing-workflows
```

**Tools & Utilities (11 plugins)**
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

**Research & Visualization (3 plugins)**
```bash
/plugin install data-visualization@scientific-computing-workflows
/plugin install research-methodology@scientific-computing-workflows
/plugin install full-stack-orchestration@scientific-computing-workflows
```

### Available Plugins

<details>
<summary><b>All 31 Plugins by Category (Click to expand)</b></summary>

#### AI/ML (5)
- **agent-orchestration** - Multi-agent system optimization and context management
- **ai-reasoning** - Advanced cognitive tools for problem-solving and meta-analysis
- **deep-learning** - Neural network architecture design and training diagnostics
- **jax-implementation** - JAX with NumPyro Bayesian inference, ArviZ integration, Consensus Monte Carlo, and response verification protocols
- **machine-learning** - Data science, statistical analysis, and MLOps workflows

#### Development (8)
- **backend-development** - Backend API design, GraphQL architecture, TDD
- **debugging-toolkit** - Interactive debugging and developer experience optimization
- **frontend-mobile-development** - Frontend UI and mobile app implementation
- **javascript-typescript** - Modern JavaScript/TypeScript with ES6+ and Node.js
- **llm-application-dev** - LLM applications with prompt engineering and RAG
- **multi-platform-apps** - Cross-platform web, iOS, Android, desktop apps
- **python-development** - Python with FastAPI, Django, async patterns
- **systems-programming** - Rust, C, C++, and Go development

#### Scientific Computing (4)
- **hpc-computing** - High-performance computing and numerical methods
- **julia-development** - Julia ecosystem with SciML and Bayesian inference
- **molecular-simulation** - Molecular dynamics with LAMMPS and GROMACS
- **statistical-physics** - Correlation function analysis and FFT

#### Tools & Utilities (11)
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

#### Research & Visualization (3)
- **data-visualization** - Scientific data visualization and AR/VR interfaces
- **research-methodology** - Research intelligence and literature analysis
- **full-stack-orchestration** - End-to-end full-stack feature delivery

</details>

### Using Plugins

Once installed, plugins provide agents, commands, and skills that are automatically available in Claude Code:

**Using Specialized Agents**
```
Ask Claude: "@python-pro help me optimize this async function"
Ask Claude: "@julia-pro implement this differential equation using SciML"
Ask Claude: "@jax-pro optimize this neural network training loop"
Ask Claude: "@ml-engineer build a scikit-learn pipeline for this dataset"
Ask Claude: "@rust-pro refactor this code for better memory safety"
```

**Running Commands**
```bash
/ultra-think "Analyze the architecture of this system"
/double-check "Validate this implementation"
/run-all-tests --fix  # Run and fix all tests
/commit "Add new feature"  # Intelligent git commit
```

**Accessing Skills**
Skills are automatically loaded and available to Claude based on file context and your requests.

### Integration Examples

See [full documentation](https://myclaude.readthedocs.io/en/latest/guides/) for detailed workflow guides:
- Scientific computing (Julia + HPC + GPU)
- Development (Python + API + Testing)
- DevOps (Docker + Kubernetes + CI/CD)
- Infrastructure (Terraform + Cloud + Monitoring)

## Maintenance

```bash
# Commit changes
git add .
git commit -m "Update plugins and configurations"

# Create version tag
git tag -a v1.0.0 -m "Release v1.0.0"
```

## Contributing

1. Fork and create a branch
2. Make changes to plugins
3. Test thoroughly with Claude Code
4. Submit PR with detailed description

See [contribution guidelines](https://myclaude.readthedocs.io/en/latest/contributing.html) for details.

## Documentation Links

- **Full Documentation:** [myclaude.readthedocs.io](https://myclaude.readthedocs.io/en/latest/)
- **Plugin Guides:** [myclaude.readthedocs.io/plugins/](https://myclaude.readthedocs.io/en/latest/plugins/)
- **Quick-Start Guides:** [myclaude.readthedocs.io/guides/](https://myclaude.readthedocs.io/en/latest/guides/)
- **Integration Patterns:** [myclaude.readthedocs.io/integration-map.html](https://myclaude.readthedocs.io/en/latest/integration-map.html)
- **Technical Glossary:** [myclaude.readthedocs.io/glossary.html](https://myclaude.readthedocs.io/en/latest/glossary.html)

## Version History

### v1.0.2 (Current - 2025-10-31)
- **jax-implementation v1.0.2**: Enhanced NumPyro capabilities with major improvements
  - **ArviZ Integration**: Comprehensive Bayesian visualization with 15+ diagnostic plot types, model comparison (LOO/WAIC), and publication-quality outputs
  - **Consensus Monte Carlo**: Large-scale distributed Bayesian inference for datasets with N > 1M observations
  - **Response Quality Verification**: Mandatory 26-point pre-delivery checklist across 5 categories (statistical correctness, code quality, inference validity, completeness, documentation)
  - **Self-Critique Loop**: 5-question validation protocol before delivery
  - **Anti-Pattern Prevention**: 4 documented common mistakes with WRONG/RIGHT code examples
  - **Enhanced Skill Discoverability**: Expanded descriptions from 500 to 1,300+ characters with 20+ specific trigger scenarios
  - **NLSQ Agent Enhancement**: +995 lines of comprehensive framework improvements
  - **Version Consistency**: All components aligned at v1.0.2 across agents, skills, and documentation

### v1.0.1 (2025-10-31)
- **Documentation improvements**: Consolidated PLUGIN_INSTALLATION.md into README.md
- **Enhanced installation guide**: Added category-based installation commands for easier plugin selection
- **Improved usage examples**: Concrete agent and command examples with real-world scenarios
- **Accurate resource counts**: Updated statistics (73 agents, 48 commands, 110 skills)
- **Fixed Sphinx warnings**: Eliminated toctree duplication warnings
- **Added plugin changelog links**: Comprehensive links to all 31 plugin changelogs in main documentation

### v1.0.0 (2025-10-29)
- Initial release of customized plugin marketplace
- 31 specialized plugins across 9 categories
- Comprehensive coverage for scientific computing and software development
- Includes HPC computing, Julia/JAX development, deep learning, and full-stack development
- Standardized documentation and author information
- Complete Sphinx documentation with Read the Docs integration

See [changelog](https://myclaude.readthedocs.io/en/latest/changelog.html) for detailed version history and individual plugin changelogs.

## License

MIT License (see [LICENSE](LICENSE))
