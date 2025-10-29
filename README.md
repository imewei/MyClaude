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
- **Total Agents:** 60+ | **Commands:** 40+ | **Skills:** 100+

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
- **jax-implementation**: JAX optimization and physics-informed ML
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

### Available Plugins

<details>
<summary><b>All 31 Plugins by Category (Click to expand)</b></summary>

#### AI/ML (5)
- `agent-orchestration` - Multi-agent system optimization
- `ai-reasoning` - Advanced cognitive tools and structured reasoning
- `deep-learning` - Neural network architecture and training
- `jax-implementation` - JAX programming and physics applications
- `machine-learning` - Data science and ML engineering

#### Development (8)
- `backend-development` - API design and GraphQL architecture
- `debugging-toolkit` - Interactive debugging and DX optimization
- `frontend-mobile-development` - Frontend UI and mobile apps
- `javascript-typescript` - Modern JS/TS development
- `llm-application-dev` - LLM apps with RAG and LangChain
- `multi-platform-apps` - Cross-platform development
- `python-development` - FastAPI, Django, async patterns
- `systems-programming` - Rust, C, C++, Go development

#### Scientific Computing (4)
- `hpc-computing` - High-performance computing
- `julia-development` - Julia ecosystem with SciML
- `molecular-simulation` - Molecular dynamics simulations
- `statistical-physics` - Correlation function analysis

#### Tools & Utilities (11)
- `cicd-automation` - CI/CD pipeline configuration
- `cli-tool-design` - CLI tool development
- `code-documentation` - Doc generation and technical writing
- `code-migration` - Legacy code modernization
- `codebase-cleanup` - Technical debt reduction
- `comprehensive-review` - Multi-perspective code analysis
- `framework-migration` - Framework updates and migrations
- `git-pr-workflows` - Git workflow automation
- `observability-monitoring` - Metrics and monitoring
- `quality-engineering` - QA and validation
- `unit-testing` - Test automation and generation

#### Research & Visualization (3)
- `data-visualization` - Scientific data visualization
- `research-methodology` - Research intelligence and analysis
- `full-stack-orchestration` - End-to-end feature delivery

</details>

### Using Plugins

Once installed, plugins provide agents, commands, and skills that are automatically available in Claude Code:

```bash
# Use an agent from a plugin
Ask Claude to use specialized agents like @python-pro or @rust-pro

# Run a command
/ultra-think "Analyze this architecture..."
/double-check "Validate this implementation"

# Access skills
Skills are automatically loaded and available to Claude
```

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

### v1.0.0 (Current)
- Initial release of customized plugin marketplace
- 31 specialized plugins across 9 categories
- Comprehensive coverage for scientific computing and software development
- Includes HPC computing, Julia/JAX development, deep learning, and full-stack development
- Standardized documentation and author information
- 60+ agents, 40+ commands, 100+ skills

See [changelog](https://myclaude.readthedocs.io/en/latest/changelog.html) for detailed version history.

## License

MIT License (see [LICENSE](LICENSE))
